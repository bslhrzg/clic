#pragma once
// hamiltonian.hpp
// -----------------------------------------------------------------------------
// Sparse Hamiltonian construction mirroring your Julia algorithm exactly:
// - Precompute Csa/Cda/Cma (and β analogues) on unique spin strings
// - Loops: diagonal, α-connected, β-connected, α×β singles, magnetic path
// - OpenMP parallel outer loop; optional MPI variant with Gatherv
// - Returns CSR arrays (indices:int32, indptr:int64 for safety)
// -----------------------------------------------------------------------------

#include "ci_core.h"
#include "slater_condon.h"
#include <complex>
#include <vector>
#include <cstdint>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef CI_USE_MPI
#include <mpi.h>
#endif

namespace ci {

using cx = std::complex<double>;

// CSR container to pass back to Python cleanly.
struct CSR {
    std::vector<cx>        data;
    std::vector<int32_t>   indices;   // column indices
    std::vector<int64_t>   indptr;    // row pointer (size N+1)
    std::size_t            n_rows{0}, n_cols{0};
};

// Build Hamiltonian (naive O(N^2)): Hij = KL(Di, Dj) for all i,j
CSR build_hamiltonian_naive(const std::vector<SlaterDeterminant>& basis,
                            const H1View& H, const ERIView& V,
                            double tol = 1e-16);

// Build Hamiltonian (OpenMP). Basis is a vector of Slater determinants.
// H and V are dense, row-major, complex arrays with shapes (K,K) and (K,K,K,K).
CSR build_hamiltonian_openmp(const std::vector<SlaterDeterminant>& basis,
                             const H1View& H, const ERIView& V,
                             bool enable_magnetic = true,
                             double tol = 1e-16);

// Optional MPI builder. Rank 0 returns the matrix; others return empty CSR.
// Hybrid OpenMP+MPI is supported (OpenMP inside each rank).
#ifdef CI_USE_MPI
CSR build_hamiltonian_mpi(const std::vector<SlaterDeterminant>& basis,
                          const H1View& H, const ERIView& V,
                          MPI_Comm comm,
                          bool enable_magnetic = true,
                          double tol = 1e-16);
#endif


// y = H @ x over a fixed CI basis, with all structure cached once
class FixedBasisMatvec {
public:
    using cx = std::complex<double>;

    FixedBasisMatvec(const std::vector<SlaterDeterminant>& basis,
                     const H1View& H, const ERIView& V,
                     bool enable_magnetic, double tol);

    // y and x must have length size()
    void apply(const cx* x, cx* y) const;

    std::size_t size() const { return N_; }

private:
    // sizes and views
    std::size_t N_{0};
    std::size_t M_{0};
    H1View H_;
    ERIView V_;

    // cached per row
    std::vector<Determinant> D_;                // combined determinants Di for all rows
    std::vector<cx>          Hii_;              // diagonal elements
    std::vector<std::vector<int32_t>> cols_;    // neighbor column indices, deduped, no diagonal
};


} // namespace ci