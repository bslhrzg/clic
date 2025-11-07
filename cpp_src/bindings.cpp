#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/stl_bind.h>

#include "ci_core.h"
#include "slater_condon.h"
#include "hamiltonian.h"
#include "ed_tools.h"
#include "applyH.h"

#include <complex>
#include <vector>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;
using namespace ci;
using cx = std::complex<double>;


// ---------------------- helpers: SciPy CSR constructors ----------------------

static py::object scipy_csr_from_CSR(const CSR& csr)
{
    py::module_ sp_sparse = py::module_::import("scipy.sparse");
    auto np_data   = py::array_t<cx>(csr.data.size(), csr.data.data());
    auto np_indices= py::array_t<int32_t>(csr.indices.size(), csr.indices.data());
    auto np_indptr = py::array_t<int64_t>(csr.indptr.size(), csr.indptr.data());
    py::tuple shape = py::make_tuple((py::int_)csr.n_rows, (py::int_)csr.n_cols);
    return sp_sparse.attr("csr_matrix")(py::make_tuple(np_data, np_indices, np_indptr),
                                        py::arg("shape")=shape);
}

static py::object scipy_csr_from_triplets(
    int dim, const std::vector<Triplet>& triplets)
{
    py::module_ sp_sparse = py::module_::import("scipy.sparse");
    std::vector<int> rows, cols;
    std::vector<cx> vals;
    rows.reserve(triplets.size());
    cols.reserve(triplets.size());
    vals.reserve(triplets.size());
    for (const auto& t : triplets) {
        rows.push_back(t.row);
        cols.push_back(t.col);
        vals.push_back(t.value);
    }
    auto np_data = py::array_t<cx>(vals.size(), vals.data());
    auto np_rows = py::array_t<int>(rows.size(), rows.data());
    auto np_cols = py::array_t<int>(cols.size(), cols.data());
    py::tuple ij = py::make_tuple(np_rows, np_cols);
    py::tuple data_ij = py::make_tuple(np_data, ij);
    py::tuple shape = py::make_tuple(dim, dim);
    return sp_sparse.attr("csr_matrix")(data_ij, py::arg("shape")=shape);
}

// --------------------------- wrap ED tools -----------------------------------

py::object get_annihilation_operator_py(int num_spin_orbitals, int orbital_index_1based)
{
    int i0 = orbital_index_1based - 1;
    auto triplets = get_annihilation_operator_sparse(num_spin_orbitals, i0);
    int dim = 1 << num_spin_orbitals;
    return scipy_csr_from_triplets(dim, triplets);
}

py::object get_creation_operator_py(int num_spin_orbitals, int orbital_index_1based)
{
    int i0 = orbital_index_1based - 1;
    auto triplets = get_creation_operator_sparse(num_spin_orbitals, i0);
    int dim = 1 << num_spin_orbitals;
    return scipy_csr_from_triplets(dim, triplets);
}

// ------------------- helpers: array views -----------------

static H1View make_H1(py::array H) {
    py::buffer_info buf = H.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("H must be a square 2D NumPy array.");
    return H1View{reinterpret_cast<const cx*>(buf.ptr), (size_t)buf.shape[0]};
}

static ERIView make_ERI(py::array V) {
    py::buffer_info buf = V.request();
    if (buf.ndim != 4 || buf.shape[0] != buf.shape[1] || buf.shape[0] != buf.shape[2] || buf.shape[0] != buf.shape[3])
        throw std::runtime_error("V must be a 4D NumPy array of shape (K,K,K,K).");
    return ERIView{reinterpret_cast<const cx*>(buf.ptr), (size_t)buf.shape[0]};
}

// -------------------------- Hamiltonian wrappers -----------------------------

py::object build_ham_naive_py(const std::vector<SlaterDeterminant>& basis,
                              py::array H, py::array V, double tol)
{
    if (basis.empty()) return py::none();
    CSR csr = build_hamiltonian_naive(basis, make_H1(H), make_ERI(V), tol);
    return scipy_csr_from_CSR(csr);
}

py::object build_ham_openmp_py(const std::vector<SlaterDeterminant>& basis,
                               py::array H, py::array V,
                               bool enable_magnetic, double tol)
{
    if (basis.empty()) return py::none();
    CSR csr = build_hamiltonian_openmp(basis, make_H1(H), make_ERI(V), enable_magnetic, tol);
    return scipy_csr_from_CSR(csr);
}


// --------------------------------- Module Definition ------------------------------------
PYBIND11_MODULE(clic_clib, m) {
m.doc() = R"doc(
    High-performance C++ backend for Configuration Interaction (CI) calculations.

    This library provides core data structures for representing Slater determinants
    and wavefunctions, efficient C++ kernels for applying operators and building
    Hamiltonian matrices using Slater-Condon rules, and tools for dynamic
    operator application.
)doc";

    // --- Enums ---
    py::enum_<Spin>(m, "Spin")
        .value("Alpha", Spin::Alpha)
        .value("Beta", Spin::Beta)
        .export_values();

    py::enum_<SpinOrbitalOrder>(m, "SpinOrbitalOrder")
        .value("AlphaFirst", SpinOrbitalOrder::AlphaFirst)
        .value("Interleaved", SpinOrbitalOrder::Interleaved)
        .export_values();
        
    // --- Result Struct for Operators ---
    py::class_<SlaterDeterminant::OpResult>(m, "SlaterDeterminantOpResult")
        .def_readonly("det", &SlaterDeterminant::OpResult::det)
        .def_readonly("sign", &SlaterDeterminant::OpResult::sign)
        .def("__repr__", [](const SlaterDeterminant::OpResult &res) {
            return "<SlaterDeterminantOpResult: sign=" + std::to_string(res.sign) + ">";
        });

    // --- Core Classes ---
    py::class_<SlaterDeterminant>(m, "SlaterDeterminant","Represents a Slater determinant with separate alpha and beta spin strings.")
        .def(py::init<std::size_t, const std::vector<int>&, const std::vector<int>&>(),
             py::arg("n_spatial"), py::arg("occ_alpha0"), py::arg("occ_beta0"),"Constructs a determinant for M spatial orbitals with given occupations.")
        .def("alpha_occupied_indices", [](const SlaterDeterminant& det) {
            std::vector<int> occ;
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) occ.push_back(*it);
            return occ;
        })
        .def("beta_occupied_indices", [](const SlaterDeterminant& det) {
            std::vector<int> occ;
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) occ.push_back(*it);
            return occ;
        })
        .def_static("create", &SlaterDeterminant::create, py::arg("s"), py::arg("i0"), py::arg("spin"), 
                    py::arg("order") = SpinOrbitalOrder::AlphaFirst)
        .def_static("annihilate", &SlaterDeterminant::annihilate, py::arg("s"), py::arg("i0"), py::arg("spin"),
                    py::arg("order") = SpinOrbitalOrder::AlphaFirst)
        .def_static("apply_excitation_single", &SlaterDeterminant::apply_excitation_single)
        .def_static("apply_excitation_double", &SlaterDeterminant::apply_excitation_double)
        .def_property_readonly("n_spatial", &SlaterDeterminant::num_spatial_orbitals)
        .def("get_occupied_spin_orbitals", [](const SlaterDeterminant& det) {
            std::vector<int> occ;
            const auto M = det.num_spatial_orbitals();
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) occ.push_back(*it);
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) occ.push_back(*it + static_cast<int>(M));
            std::sort(occ.begin(), occ.end()); // Return in sorted order
            return occ;
        }, "Returns the combined list of occupied spin-orbital indices (AlphaFirst ordering).")
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::hash(py::self))
        .def("__repr__", [](const SlaterDeterminant& det) {
            py::list alpha, beta;
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) alpha.append(*it);
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) beta.append(*it);
            return "<SlaterDeterminant α=" + std::string(py::str(alpha)) + " β=" + std::string(py::str(beta)) + ">";
        });
    
    py::class_<Wavefunction>(m, "Wavefunction")
        .def(py::init<std::size_t>(), py::arg("n_spatial"))

        .def(py::init<std::size_t, const Wavefunction::Data&>(),
         py::arg("n_spatial"), py::arg("data"))
         
        // Constructor from basis and amplitudes
        .def(py::init<std::size_t, const std::vector<SlaterDeterminant>&, const std::vector<Wavefunction::Coeff>&, bool>(),
         py::arg("n_spatial"), py::arg("basis"), py::arg("amplitudes"), py::arg("keep_zeros") = true)


        .def("add_term", &Wavefunction::add_term, py::arg("det"), py::arg("coeff"), py::arg("tol") = 0)

        .def("add_wavefunction", &Wavefunction::add_wavefunction, 
         py::arg("other"), py::arg("scale") = cx{1.0, 0.0},
         "Adds another wavefunction scaled by a coefficient, modifying this one in-place.")

        .def("data", &Wavefunction::data, py::return_value_policy::reference_internal)
        .def("prune", &Wavefunction::prune, py::arg("threshold"))
        .def("normalize", &Wavefunction::normalize, py::arg("tol") = 1e-15)
        .def("dot", &Wavefunction::dot, py::arg("other"), "Computes dot product <self|other>")

        .def("__iadd__", [](Wavefunction &wf, const Wavefunction &other) { wf.add_wavefunction(other); return wf; }, py::is_operator())

        .def("__add__", [](const Wavefunction &wf, const Wavefunction &other) { Wavefunction result = wf; result.add_wavefunction(other); return result; }, py::is_operator())
        .def("__sub__", [](const Wavefunction &wf, const Wavefunction &other) { Wavefunction result = wf; result.add_wavefunction(other, {-1.0, 0.0}); return result; }, py::is_operator())
        .def("__mul__", [](const Wavefunction &wf, cx scale) { Wavefunction result(wf.num_spatial_orbitals()); for (const auto& [det, coeff] : wf.data()) { result.add_term(det, coeff * scale); } return result; }, py::is_operator())
        .def("__rmul__", [](const Wavefunction &wf, cx scale) { Wavefunction result(wf.num_spatial_orbitals()); for (const auto& [det, coeff] : wf.data()) { result.add_term(det, coeff * scale); } return result; }, py::is_operator())
        .def("get_basis", &Wavefunction::basis_sorted, "Returns a sorted list of SlaterDeterminant objects in the wavefunction.")
        .def("get_amplitudes", [](Wavefunction& wf) {
            auto basis = wf.basis_sorted();
            auto coeffs_vec = wf.coeffs_sorted(basis);
            return py::array_t<cx>(coeffs_vec.size(), coeffs_vec.data());
        }, "Returns a NumPy array of amplitudes, sorted according to the basis.")
        .def_property_readonly("n_spatial", &Wavefunction::num_spatial_orbitals)
        .def("__repr__", [](const Wavefunction& wf) {
            return "<Wavefunction with " + std::to_string(wf.data().size()) + " terms>";
        });

    m.def("apply_creation", &apply_creation, py::arg("wf"), py::arg("i0"), py::arg("spin"),
          py::arg("order") = SpinOrbitalOrder::AlphaFirst);
    m.def("apply_annihilation", &apply_annihilation, py::arg("wf"), py::arg("i0"), py::arg("spin"),
          py::arg("order") = SpinOrbitalOrder::AlphaFirst);

    // Some useful accesses 
    // Return coeff if present, else 0+0j
    m.attr("Wavefunction").cast<py::class_<Wavefunction>>()
        .def("amplitude", [](const Wavefunction& wf, const SlaterDeterminant& det) {
            const auto& M = wf.data();                // const ref to unordered_map
            auto it = M.find(det);
            return (it == M.end()) ? cx{0.0, 0.0} : it->second;
        }, py::arg("det"), R"doc(Returns ⟨det|ψ⟩, or 0 if absent.)doc")


        // Pythonic containment test: `det in wf`
        .def("__contains__", [](const Wavefunction& wf, const SlaterDeterminant& det) {
            return wf.data().find(det) != wf.data().end();
        }, py::arg("det"));

    // --- Wavefunction Operator Functions (MODIFIED SECTION) ---
    // Use lambdas to convert Python list of tuples to C++ vector of structs
    m.def("apply_one_body_operator",
          [](const Wavefunction& wf, const std::vector<std::tuple<size_t, size_t, Spin, Spin, Wavefunction::Coeff>>& term_tuples) {
              std::vector<OneBodyTerm> terms;
              terms.reserve(term_tuples.size());
              for (const auto& t : term_tuples) {
                  terms.emplace_back(OneBodyTerm{
                      std::get<0>(t), std::get<1>(t),
                      std::get<2>(t), std::get<3>(t),
                      std::get<4>(t)
                  });
              }
              return apply_one_body_operator(wf, terms);
          },
          py::arg("wf"), py::arg("terms"));

    m.def("apply_two_body_operator",
          [](const Wavefunction& wf, const std::vector<std::tuple<size_t, size_t, size_t, size_t, Spin, Spin, Spin, Spin, Wavefunction::Coeff>>& term_tuples) {
              std::vector<TwoBodyTerm> terms;
              terms.reserve(term_tuples.size());
              for (const auto& t : term_tuples) {
                  terms.emplace_back(TwoBodyTerm{
                      std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t),
                      std::get<4>(t), std::get<5>(t), std::get<6>(t), std::get<7>(t),
                      std::get<8>(t)
                  });
              }
              return apply_two_body_operator(wf, terms);
          },
          py::arg("wf"), py::arg("terms"));

    // --- Basis Connectivity Functions (MODIFIED SECTION) ---
    m.def("get_connections_one_body",
          [](const std::vector<SlaterDeterminant>& basis, const std::vector<std::tuple<size_t, size_t, Spin, Spin, Wavefunction::Coeff>>& term_tuples) {
              std::vector<OneBodyTerm> terms;
              terms.reserve(term_tuples.size());
              for (const auto& t : term_tuples) {
                  terms.emplace_back(OneBodyTerm{
                      std::get<0>(t), std::get<1>(t),
                      std::get<2>(t), std::get<3>(t),
                      std::get<4>(t)
                  });
              }
              return get_connections_one_body(basis, terms);
          },
          py::arg("basis"), py::arg("terms"));

    m.def("get_connections_two_body",
          [](const std::vector<SlaterDeterminant>& basis, const std::vector<std::tuple<size_t, size_t, size_t, size_t, Spin, Spin, Spin, Spin, Wavefunction::Coeff>>& term_tuples) {
              std::vector<TwoBodyTerm> terms;
              terms.reserve(term_tuples.size());
              for (const auto& t : term_tuples) {
                  terms.emplace_back(TwoBodyTerm{
                      std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t),
                      std::get<4>(t), std::get<5>(t), std::get<6>(t), std::get<7>(t),
                      std::get<8>(t)
                  });
              }
              return get_connections_two_body(basis, terms);
          },
          py::arg("basis"), py::arg("terms"));


    // --- Table-driven Hamiltonian Application (NEW, CORRECTED SECTION) ---
    py::class_<ScreenedHamiltonian>(m, "ScreenedHamiltonian", "Pre-computed tables for efficient H|psi> application.")
        .def(py::init<>())
        .def("__repr__", [](const ScreenedHamiltonian& sh) {
            return "<ScreenedHamiltonian for K=" + std::to_string(sh.n_spin_orbitals) + " spin-orbitals>";
        });

    m.def("build_screened_hamiltonian",
        [](py::array H, py::array V, double tol) {
            size_t K = H.request().shape[0];
            return build_screened_hamiltonian(K, make_H1(H), make_ERI(V), tol);
        },
        py::arg("H"), py::arg("V"), py::arg("tol"),
        R"doc(
        Builds pre-computed screening tables for fast Hamiltonian application based on integral values.
        This is a direct C++ translation of the proven Python `build_*` functions.
        )doc");

    m.def("apply_hamiltonian",
        [](const Wavefunction& psi, const ScreenedHamiltonian& sh, py::array H, py::array V, double tol_element) {
            return apply_hamiltonian(psi, sh, make_H1(H), make_ERI(V), tol_element);
        },
        py::arg("psi"), py::arg("screened_H"), py::arg("H"), py::arg("V"), py::arg("tol_element"),
        R"doc(
        Applies the Hamiltonian to a wavefunction using pre-built screening tables.
        This is a direct C++ translation of the proven Python `apply_H_on_det` logic,
        applied to each determinant in the input wavefunction.
        )doc");

    // --- fixed-basis: build fixed-basis tables ---
    m.def("build_fixed_basis_tables",
        [](const ScreenedHamiltonian& sh_full,
           const std::vector<SlaterDeterminant>& basis,
           size_t M)
        {
            return build_fixed_basis_tables(sh_full, basis, M);
        },
        py::arg("screened_full"),
        py::arg("basis"),
        py::arg("M"),
        R"doc(
        Build screening tables restricted to a fixed determinant basis.

        This function takes:
          * `screened_full`: the full screening tables (from build_screened_hamiltonian)
          * `basis`: a list of SlaterDeterminant objects defining the fixed CI basis
          * `M`: number of spatial orbitals

        It removes all excitations involving spin-orbitals that never appear
        in the union of occupied orbitals across the basis.

        The resulting ScreenedHamiltonian is typically much smaller and allows
        very fast application of H restricted to the fixed basis.
        )doc"
    );

    // --- fixed-basis H application ---
    m.def("apply_hamiltonian_fixed_basis",
        [](const Wavefunction& psi,
           const ScreenedHamiltonian& sh_fb,
           const std::vector<SlaterDeterminant>& basis,
           py::array H,
           py::array V,
           double tol_element)
        {
            return apply_hamiltonian_fixed_basis(
                psi,
                sh_fb,
                basis,
                make_H1(H),
                make_ERI(V),
                tol_element
            );
        },
        py::arg("psi"),
        py::arg("screened_H_fixed_basis"),
        py::arg("basis"),
        py::arg("H"),
        py::arg("V"),
        py::arg("tol_element"),
        R"doc(
        Apply the Hamiltonian to a wavefunction restricted to a fixed determinant basis.

        Only determinants present in `basis` are ever produced, and only excitations
        allowed by the filtered screening tables are considered.

        This is the fixed-basis analogue of `apply_hamiltonian`, and should be used
        together with `build_fixed_basis_tables`.
        )doc"
    );


    // Expose FixedBasisCSR and builders
    py::class_<ci::FixedBasisCSR>(m, "FixedBasisCSR")
        .def_property_readonly("N", [](const ci::FixedBasisCSR& A){ return A.N; })
        .def_property_readonly("nnz", [](const ci::FixedBasisCSR& A){ return (int64_t)A.data.size(); })
        .def_property_readonly("indptr", [](const ci::FixedBasisCSR& A){
            return py::array_t<int64_t>(A.indptr.size(), A.indptr.data());
        })
        .def_property_readonly("indices", [](const ci::FixedBasisCSR& A){
            return py::array_t<int32_t>(A.indices.size(), A.indices.data());
        })
        .def_property_readonly("data", [](const ci::FixedBasisCSR& A){
            return py::array_t<std::complex<double>>(A.data.size(), reinterpret_cast<const std::complex<double>*>(A.data.data()));
        });

    m.def("build_fixed_basis_csr",
        [](const ci::ScreenedHamiltonian& sh_fb,
        const std::vector<ci::SlaterDeterminant>& basis,
        py::array H, py::array V)
        {
            return ci::build_fixed_basis_csr(sh_fb, basis, make_H1(H), make_ERI(V));
        },
        py::arg("screened_H_fixed_basis"),
        py::arg("basis"),
        py::arg("H"), py::arg("V"));

    m.def("csr_matvec",
        [](const ci::FixedBasisCSR& A,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> x)
        {
            if ((size_t)x.size() != A.N) throw std::runtime_error("x has wrong length");
            py::array_t<std::complex<double>> y(A.N);
            ci::csr_matvec(A,
                reinterpret_cast<const cx*>(x.data()),
                reinterpret_cast<cx*>(y.mutable_data()));
            return y;
        },
        py::arg("A"), py::arg("x"));


    m.def("build_fixed_basis_csr_full",
    [](const std::vector<ci::SlaterDeterminant>& basis,
       size_t M,
       py::array H,
       py::array V,
       double tol_tables,
       double drop_tol)
    {
        return ci::build_fixed_basis_csr_full(
            basis, M, make_H1(H), make_ERI(V), tol_tables, drop_tol
        );
    },
    py::arg("basis"),
    py::arg("M"),
    py::arg("H"),
    py::arg("V"),
    py::arg("tol_tables") = 1e-12,
    py::arg("drop_tol")   = 0.0,
    R"doc(
    Build the projected Hamiltonian on a fixed determinant basis as a CSR matrix.

    Steps:
      1) build_screened_hamiltonian(2*M, H, V, tol_tables)
      2) build_fixed_basis_tables(...)
      3) build_fixed_basis_csr(...)

    Returns a FixedBasisCSR with (indptr, indices, data).
    Optionally prunes |Hij| <= drop_tol after compression.
    )doc");


    // --- ED Tools & Hamiltonian Construction ---
    m.def("get_annihilation_operator", &get_annihilation_operator_py,
          py::arg("num_spin_orbitals"), py::arg("orbital_index_1based"));
    m.def("get_creation_operator", &get_creation_operator_py,
          py::arg("num_spin_orbitals"), py::arg("orbital_index_1based"));

    m.def("build_hamiltonian_naive", &build_ham_naive_py,
      py::arg("basis"), py::arg("H"), py::arg("V"), py::arg("tol") = 1e-16);

    m.def("build_hamiltonian_openmp", &build_ham_openmp_py,
          py::arg("basis"), py::arg("H"), py::arg("V"),
          py::arg("enable_magnetic") = true, py::arg("tol") = 1e-16,R"doc(
        Builds the sparse Hamiltonian matrix for a given basis using OpenMP.

        This is the main high-performance function for constructing the CI matrix.
        It expects integrals in the 'AlphaFirst' basis ordering.

        Args:
            basis (list[SlaterDeterminant]): A list of SlaterDeterminant objects defining the basis.
            H (np.ndarray): The (K, K) one-electron integral matrix (complex128, C-contiguous).
            V (np.ndarray): The (K, K, K, K) two-electron integral tensor (complex128, C-contiguous).
            enable_magnetic (bool): Flag to include magnetic interactions (default: True).
            tol (float): Tolerance below which matrix elements are considered zero.

        Returns:
            scipy.sparse.csr_matrix: The Hamiltonian matrix in sparse CSR format.
    )doc");

    m.def("KL", [](const std::vector<int>& occ1, const std::vector<int>& occ2, size_t K,
                   py::array H, py::array V) {
        Determinant D1(K, occ1);
        Determinant D2(K, occ2);
        return KL(D1, D2, make_H1(H), make_ERI(V));
    }, py::arg("occ1"), py::arg("occ2"), py::arg("K"), py::arg("H"), py::arg("V"));
}
