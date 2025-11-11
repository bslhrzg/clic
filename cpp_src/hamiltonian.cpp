// hamiltonian.cpp
#include "hamiltonian.h"

#include <bit>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <cassert>

// Note: We keep AlphaFirst ordering consistently.

namespace ci {

// -------------------------- utilities: hashing & keys ------------------------

struct SDPairHash {
    std::size_t operator()(const SlaterDeterminant& s) const noexcept {
        return std::hash<SlaterDeterminant>{}(s);
    }
};

struct SDPairEq {
    bool operator()(const SlaterDeterminant& a, const SlaterDeterminant& b) const noexcept {
        return a == b;
    }
};

// Build combined determinant (K=2M) directly from α/β sets (faster than interleave()).
static inline void occ_list_alpha_beta(const SlaterDeterminant& sd, std::vector<int>& occK) {
    occK.clear();
    const std::size_t M = sd.num_spatial_orbitals();
    for (auto it = sd.alpha().begin_occ(); it != sd.alpha().end_occ(); ++it) {
        occK.push_back(*it);
    }
    for (auto it = sd.beta().begin_occ(); it != sd.beta().end_occ(); ++it) {
        occK.push_back(static_cast<int>(M + *it));
    }
}

// Construct combined Determinant (K=2M) from SlaterDeterminant quickly.
static inline Determinant combined_from(const SlaterDeterminant& sd) {
    const std::size_t M = sd.num_spatial_orbitals();
    Determinant D(2*M);
    for (auto it = sd.alpha().begin_occ(); it != sd.alpha().end_occ(); ++it)
        D.set(static_cast<std::size_t>(*it));
    for (auto it = sd.beta().begin_occ(); it != sd.beta().end_occ(); ++it)
        D.set(static_cast<std::size_t>(M + *it));
    return D;
}

// XOR popcount for SpinDeterminant pair (same length).
static inline int xor_popcount(const SpinDeterminant& a, const SpinDeterminant& b) {
    const auto& wa = a.raw().words();
    const auto& wb = b.raw().words();
    std::size_t c = 0;
    for (std::size_t i = 0; i < wa.size(); ++i) c += std::popcount(wa[i] ^ wb[i]);
    return static_cast<int>(c);
}

// --------------------------- Caa / Cd / Cmag --------------------------------

struct Connections {
    // maps: key SpinDeterminant -> vector of connected SpinDeterminant
    std::unordered_map<SpinDeterminant, std::vector<SpinDeterminant>> Cs; // single
    std::unordered_map<SpinDeterminant, std::vector<SpinDeterminant>> Cd; // double
    std::unordered_map<SpinDeterminant, std::vector<SpinDeterminant>> Cmag; // magnetic
};

static Connections Caa(const std::vector<SpinDeterminant>& Ia, bool is_mag) {
    Connections res;
    res.Cs.reserve(Ia.size());
    res.Cd.reserve(Ia.size());
    res.Cmag.reserve(Ia.size());

    // Group by electron count to match your Julia condition "count_ones(ei)==count_ones(ej)"
    std::unordered_map<int, std::vector<SpinDeterminant>> byNe;
    for (const auto& e : Ia) byNe[static_cast<int>(e.count_electrons())].push_back(e);

    // Pre-fill maps
    for (const auto& e : Ia) {
        res.Cs.emplace(e, std::vector<SpinDeterminant>{});
        res.Cd.emplace(e, std::vector<SpinDeterminant>{});
        res.Cmag.emplace(e, std::vector<SpinDeterminant>{});
    }

    // Singles + doubles within fixed electron count
    for (auto& kv : byNe) {
        auto& vec = kv.second;
        const std::size_t n = vec.size();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const int diff = xor_popcount(vec[i], vec[j]);
                if (diff == 2)      res.Cs[vec[i]].push_back(vec[j]);
                else if (diff == 4) res.Cd[vec[i]].push_back(vec[j]);
            }
        }
    }

    // Magnetic connections (allowed only if requested)
    if (is_mag) {
        // Compare across Ne±1 with xor popcount == 1
        for (const auto& e : Ia) {
            const int Ne = static_cast<int>(e.count_electrons());
            auto it1 = byNe.find(Ne+1);
            auto it2 = byNe.find(Ne-1);
            if (it1 != byNe.end()) {
                for (const auto& f : it1->second) {
                    if (xor_popcount(e, f) == 1) res.Cmag[e].push_back(f);
                }
            }
            if (it2 != byNe.end()) {
                for (const auto& f : it2->second) {
                    if (xor_popcount(e, f) == 1) res.Cmag[e].push_back(f);
                }
            }
        }
    }

    return res;
}

// ------------------------------ basis indexing -------------------------------

struct BasisIndex {
    // Keyed by combined Determinant; value = position in basis
    std::unordered_map<Determinant, int, std::hash<Determinant>> map;
};

static BasisIndex make_basis_index(const std::vector<SlaterDeterminant>& basis) {
    BasisIndex bi;
    bi.map.reserve(basis.size()*2);
    for (std::size_t i = 0; i < basis.size(); ++i) {
        Determinant D = combined_from(basis[i]);
        bi.map.emplace(std::move(D), static_cast<int>(i));
    }
    return bi;
}

// Build unique α/β sets from basis.
static std::pair<std::vector<SpinDeterminant>, std::vector<SpinDeterminant>>
unique_alpha_beta(const std::vector<SlaterDeterminant>& basis)
{
    std::unordered_set<SpinDeterminant> SA, SB;
    SA.reserve(basis.size()); SB.reserve(basis.size());
    for (const auto& s : basis) { SA.insert(s.alpha()); SB.insert(s.beta()); }
    return { std::vector<SpinDeterminant>(SA.begin(), SA.end()),
             std::vector<SpinDeterminant>(SB.begin(), SB.end()) };
}

// ------------------------------ COO buffer -----------------------------------

struct COO {
    std::vector<int32_t> row;
    std::vector<int32_t> col;
    std::vector<cx>      val;

    void push(int32_t r, int32_t c, cx v, double tol) {
        if (std::abs(v) < tol) return;
        row.push_back(r); col.push_back(c); val.push_back(v);
    }
};

static CSR coo_to_csr(std::size_t N, std::size_t M, COO&& coo) {
    CSR out;
    out.n_rows = N; out.n_cols = M;
    out.indptr.assign(N+1, 0);

    const std::size_t nnz = coo.val.size();
    // Count per-row
    for (std::size_t e = 0; e < nnz; ++e) {
        ++out.indptr[static_cast<std::size_t>(coo.row[e]) + 1];
    }
    // Prefix sum
    for (std::size_t i = 0; i < N; ++i) out.indptr[i+1] += out.indptr[i];

    out.indices.resize(nnz);
    out.data.resize(nnz);

    // Temporary write cursors (copy of indptr)
    std::vector<int64_t> cursor = out.indptr;

    for (std::size_t e = 0; e < nnz; ++e) {
        const int32_t r = coo.row[e];
        const int64_t pos = cursor[static_cast<std::size_t>(r)]++;
        out.indices[static_cast<std::size_t>(pos)] = coo.col[e];
        out.data   [static_cast<std::size_t>(pos)] = coo.val[e];
    }

    // (Optional) we could sort columns within each row and sum duplicates.
    // For now, algorithm produces each (i,j) at most once, mirroring Julia.

    return out;
}

// ------------------------------ main builder ---------------------------------

CSR build_hamiltonian_openmp(const std::vector<SlaterDeterminant>& basis,
                             const H1View& H, const ERIView& V,
                             bool enable_magnetic,
                             double tol)
{
    const std::size_t N = basis.size();
    if (N == 0) return CSR{};

    const std::size_t M = basis[0].num_spatial_orbitals();
    const std::size_t K = 2*M;
    (void)K; // shapes already validated by caller typically

    // Precompute unique α/β sets and connections
    auto [Ia, Ib] = unique_alpha_beta(basis);
    Connections Ca = Caa(Ia, enable_magnetic);
    Connections Cb = Caa(Ib, enable_magnetic);

    // Basis index on combined determinants
    BasisIndex BI = make_basis_index(basis);

    // Thread-local COO buffers
    const int nthreads =
    #ifdef _OPENMP
        omp_get_max_threads();
    #else
        1;
    #endif

    std::vector<COO> tl_coo(static_cast<std::size_t>(nthreads));

    // Parallel outer loop
    #pragma omp parallel for schedule(dynamic)
    for (std::int64_t irow = 0; irow < static_cast<std::int64_t>(N); ++irow) {
        const int tid =
        #ifdef _OPENMP
            omp_get_thread_num();
        #else
            0;
        #endif
        COO& coo = tl_coo[static_cast<std::size_t>(tid)];

        const SlaterDeterminant& S = basis[static_cast<std::size_t>(irow)];
        const SpinDeterminant& Sa = S.alpha();
        const SpinDeterminant& Sb = S.beta();

        // Precompute combined determinant and occ list for S
        Determinant D = combined_from(S);
        std::vector<int> occK; occ_list_alpha_beta(S, occK);

        // --- Diagonal ---
        {
            cx h = OO(H, V, occK);
            if (std::abs(h) >= tol) coo.push(static_cast<int32_t>(irow), static_cast<int32_t>(irow), h, tol);
        }

        // --- single/double alpha ---
        {
            auto it1 = Ca.Cs.find(Sa);
            auto it2 = Ca.Cd.find(Sa);
            if (it1 != Ca.Cs.end()) {
                for (const auto& ca : it1->second) {
                    SlaterDeterminant Sd(M, std::vector<int>{}, std::vector<int>{}); // scratch, but we need a real object
                    // Build combined determinant ND directly: α=ca, β=Sb
                    Determinant ND(2*M);
                    for (auto it = ca.begin_occ(); it != ca.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                    for (auto it = Sb.begin_occ(); it != Sb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));

                    auto itb = BI.map.find(ND);
                    if (itb != BI.map.end()) {
                        const int jcol = itb->second;
                        cx val = KL(D, ND, H, V);
                        if (std::abs(val) >= tol) coo.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                    }
                }
            }
            if (it2 != Ca.Cd.end()) {
                for (const auto& ca : it2->second) {
                    Determinant ND(2*M);
                    for (auto it = ca.begin_occ(); it != ca.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                    for (auto it = Sb.begin_occ(); it != Sb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                    auto itb = BI.map.find(ND);
                    if (itb != BI.map.end()) {
                        const int jcol = itb->second;
                        cx val = KL(D, ND, H, V);
                        if (std::abs(val) >= tol) coo.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                    }
                }
            }
        }

        // --- single/double beta ---
        {
            auto it1 = Cb.Cs.find(Sb);
            auto it2 = Cb.Cd.find(Sb);
            if (it1 != Cb.Cs.end()) {
                for (const auto& cb : it1->second) {
                    Determinant ND(2*M);
                    for (auto it = Sa.begin_occ(); it != Sa.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                    for (auto it = cb.begin_occ(); it != cb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                    auto itb = BI.map.find(ND);
                    if (itb != BI.map.end()) {
                        const int jcol = itb->second;
                        cx val = KL(D, ND, H, V);
                        if (std::abs(val) >= tol) coo.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                    }
                }
            }
            if (it2 != Cb.Cd.end()) {
                for (const auto& cb : it2->second) {
                    Determinant ND(2*M);
                    for (auto it = Sa.begin_occ(); it != Sa.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                    for (auto it = cb.begin_occ(); it != cb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                    auto itb = BI.map.find(ND);
                    if (itb != BI.map.end()) {
                        const int jcol = itb->second;
                        cx val = KL(D, ND, H, V);
                        if (std::abs(val) >= tol) coo.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                    }
                }
            }
        }

        // --- double = single alpha + single beta ---
        {
            auto itSa = Ca.Cs.find(Sa);
            auto itSb = Cb.Cs.find(Sb);
            if (itSa != Ca.Cs.end() && itSb != Cb.Cs.end()) {
                const auto& Va = itSa->second;
                const auto& Vb = itSb->second;
                for (const auto& ca : Va) {
                    for (const auto& cb : Vb) {
                        Determinant ND(2*M);
                        for (auto it = ca.begin_occ(); it != ca.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                        for (auto it = cb.begin_occ(); it != cb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                        auto itb = BI.map.find(ND);
                        if (itb != BI.map.end()) {
                            const int jcol = itb->second;
                            cx val = KL(D, ND, H, V);
                            if (std::abs(val) >= tol) coo.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                        }
                    }
                }
            }
        }

        // --- magnetic interactions ---
        if (enable_magnetic) {
            auto itMa = Ca.Cmag.find(Sa);
            auto itMb = Cb.Cmag.find(Sb);
            if (itMa != Ca.Cmag.end() && itMb != Cb.Cmag.end()) {
                const auto& Ma = itMa->second;
                const auto& Mb = itMb->second;
                for (const auto& ca : Ma) {
                    for (const auto& cb : Mb) {
                        Determinant ND(2*M);
                        for (auto it = ca.begin_occ(); it != ca.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                        for (auto it = cb.begin_occ(); it != cb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));

                        // Mirror your final guard:
                        // if nd!=0 && count_ones(xor(nd, od)) == 2
                        // (nd!=0 is implied here by having at least something changed)
                        const int ham = [&]{
                            const auto& wa = D.bits().words();
                            const auto& wb = ND.bits().words();
                            std::size_t c = 0;
                            for (std::size_t w = 0; w < wa.size(); ++w) c += std::popcount(wa[w] ^ wb[w]);
                            return static_cast<int>(c);
                        }();
                        if (ham == 2) {
                            auto itb = BI.map.find(ND);
                            if (itb != BI.map.end()) {
                                const int jcol = itb->second;
                                cx val = KL(D, ND, H, V);
                                if (std::abs(val) >= tol) coo.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                            }
                        }
                    }
                }
            }
        }
    }

    // Concatenate thread-local COO
    COO all;
    std::size_t total = 0;
    for (const auto& c : tl_coo) total += c.val.size();
    all.row.reserve(total); all.col.reserve(total); all.val.reserve(total);
    for (auto& c : tl_coo) {
        all.row.insert(all.row.end(), c.row.begin(), c.row.end());
        all.col.insert(all.col.end(), c.col.begin(), c.col.end());
        all.val.insert(all.val.end(), c.val.begin(), c.val.end());
    }

    return coo_to_csr(N, N, std::move(all));
}

#ifdef CI_USE_MPI
CSR build_hamiltonian_mpi(const std::vector<SlaterDeterminant>& basis,
                          const H1View& H, const ERIView& V,
                          MPI_Comm comm,
                          bool enable_magnetic,
                          double tol)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const std::size_t N = basis.size();
    if (N == 0) return CSR{};

    const std::size_t M = basis[0].num_spatial_orbitals();

    // Precompute unique α/β and connections on rank 0, broadcast
    // For simplicity and determinism, recompute on all ranks (as in your broadcast path).
    auto [Ia, Ib] = unique_alpha_beta(basis);
    Connections Ca = Caa(Ia, enable_magnetic);
    Connections Cb = Caa(Ib, enable_magnetic);
    BasisIndex BI = make_basis_index(basis);

    // Partition rows
    std::size_t rows_per = (N + static_cast<std::size_t>(size) - 1) / static_cast<std::size_t>(size);
    std::size_t i0 = static_cast<std::size_t>(rank) * rows_per;
    std::size_t i1 = std::min(N, i0 + rows_per);

    COO local;

    for (std::size_t irow = i0; irow < i1; ++irow) {
        const SlaterDeterminant& S = basis[irow];
        const SpinDeterminant& Sa = S.alpha();
        const SpinDeterminant& Sb = S.beta();

        Determinant D = combined_from(S);
        std::vector<int> occK; occ_list_alpha_beta(S, occK);

        // Diagonal
        {
            cx h = OO(H, V, occK);
            if (std::abs(h) >= tol) local.push(static_cast<int32_t>(irow), static_cast<int32_t>(irow), h, tol);
        }

        // α connections
        {
            auto it1 = Ca.Cs.find(Sa);
            auto it2 = Ca.Cd.find(Sa);
            if (it1 != Ca.Cs.end()) {
                for (const auto& ca : it1->second) {
                    Determinant ND(2*M);
                    for (auto it = ca.begin_occ(); it != ca.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                    for (auto it = Sb.begin_occ(); it != Sb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                    auto itb = BI.map.find(ND);
                    if (itb != BI.map.end()) {
                        const int jcol = itb->second;
                        cx val = KL(D, ND, H, V);
                        if (std::abs(val) >= tol) local.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                    }
                }
            }
            if (it2 != Ca.Cd.end()) {
                for (const auto& ca : it2->second) {
                    Determinant ND(2*M);
                    for (auto it = ca.begin_occ(); it != ca.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                    for (auto it = Sb.begin_occ(); it != Sb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                    auto itb = BI.map.find(ND);
                    if (itb != BI.map.end()) {
                        const int jcol = itb->second;
                        cx val = KL(D, ND, H, V);
                        if (std::abs(val) >= tol) local.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                    }
                }
            }
        }

        // β connections
        {
            auto it1 = Cb.Cs.find(Sb);
            auto it2 = Cb.Cd.find(Sb);
            if (it1 != Cb.Cs.end()) {
                for (const auto& cb : it1->second) {
                    Determinant ND(2*M);
                    for (auto it = Sa.begin_occ(); it != Sa.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                    for (auto it = cb.begin_occ(); it != cb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                    auto itb = BI.map.find(ND);
                    if (itb != BI.map.end()) {
                        const int jcol = itb->second;
                        cx val = KL(D, ND, H, V);
                        if (std::abs(val) >= tol) local.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                    }
                }
            }
            if (it2 != Cb.Cd.end()) {
                for (const auto& cb : it2->second) {
                    Determinant ND(2*M);
                    for (auto it = Sa.begin_occ(); it != Sa.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                    for (auto it = cb.begin_occ(); it != cb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                    auto itb = BI.map.find(ND);
                    if (itb != BI.map.end()) {
                        const int jcol = itb->second;
                        cx val = KL(D, ND, H, V);
                        if (std::abs(val) >= tol) local.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                    }
                }
            }
        }

        // α×β singles
        {
            auto itSa = Ca.Cs.find(Sa);
            auto itSb = Cb.Cs.find(Sb);
            if (itSa != Ca.Cs.end() && itSb != Cb.Cs.end()) {
                for (const auto& ca : itSa->second) {
                    for (const auto& cb : itSb->second) {
                        Determinant ND(2*M);
                        for (auto it = ca.begin_occ(); it != ca.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                        for (auto it = cb.begin_occ(); it != cb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));
                        auto itb = BI.map.find(ND);
                        if (itb != BI.map.end()) {
                            const int jcol = itb->second;
                            cx val = KL(D, ND, H, V);
                            if (std::abs(val) >= tol) local.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                        }
                    }
                }
            }
        }

        // magnetic
        if (enable_magnetic) {
            auto itMa = Ca.Cmag.find(Sa);
            auto itMb = Cb.Cmag.find(Sb);
            if (itMa != Ca.Cmag.end() && itMb != Cb.Cmag.end()) {
                for (const auto& ca : itMa->second) {
                    for (const auto& cb : itMb->second) {
                        Determinant ND(2*M);
                        for (auto it = ca.begin_occ(); it != ca.end_occ(); ++it) ND.set(static_cast<std::size_t>(*it));
                        for (auto it = cb.begin_occ(); it != cb.end_occ(); ++it) ND.set(static_cast<std::size_t>(M + *it));

                        // Guard: xor popcount == 2
                        const int ham = [&]{
                            const auto& wa = D.bits().words();
                            const auto& wb = ND.bits().words();
                            std::size_t c = 0;
                            for (std::size_t w = 0; w < wa.size(); ++w) c += std::popcount(wa[w] ^ wb[w]);
                            return static_cast<int>(c);
                        }();
                        if (ham == 2) {
                            auto itb = BI.map.find(ND);
                            if (itb != BI.map.end()) {
                                const int jcol = itb->second;
                                cx val = KL(D, ND, H, V);
                                if (std::abs(val) >= tol) local.push(static_cast<int32_t>(irow), static_cast<int32_t>(jcol), val, tol);
                            }
                        }
                    }
                }
            }
        }
    }

    // Gather to rank 0
    std::int64_t local_nnz = static_cast<std::int64_t>(local.val.size());
    std::vector<std::int64_t> counts(size);
    MPI_Allgather(&local_nnz, 1, MPI_LONG_LONG, counts.data(), 1, MPI_LONG_LONG, comm);

    std::int64_t total_nnz = 0;
    std::vector<std::int64_t> displs(size, 0);
    for (int r = 0; r < size; ++r) { displs[r] = total_nnz; total_nnz += counts[r]; }

    // Prepare recv buffers on root
    std::vector<int32_t> R_row, R_col;
    std::vector<cx>      R_val;
    if (rank == 0) {
        R_row.resize(static_cast<std::size_t>(total_nnz));
        R_col.resize(static_cast<std::size_t>(total_nnz));
        R_val.resize(static_cast<std::size_t>(total_nnz));
    }

    // Gatherv rows/cols/vals
    MPI_Gatherv(local.row.data(), static_cast<int>(local.row.size()), MPI_INT,
                R_row.data(), reinterpret_cast<int*>(counts.data()),
                reinterpret_cast<int*>(displs.data()), MPI_INT, 0, comm);
    MPI_Gatherv(local.col.data(), static_cast<int>(local.col.size()), MPI_INT,
                R_col.data(), reinterpret_cast<int*>(counts.data()),
                reinterpret_cast<int*>(displs.data()), MPI_INT, 0, comm);
    MPI_Gatherv(local.val.data(), static_cast<int>(local.val.size()), MPI_CXX_DOUBLE_COMPLEX,
                R_val.data(), reinterpret_cast<int*>(counts.data()),
                reinterpret_cast<int*>(displs.data()), MPI_CXX_DOUBLE_COMPLEX, 0, comm);

    if (rank == 0) {
        COO all;
        all.row = std::move(R_row);
        all.col = std::move(R_col);
        all.val = std::move(R_val);
        return coo_to_csr(N, N, std::move(all));
    } else {
        return CSR{};
    }
}
#endif

CSR build_hamiltonian_naive(const std::vector<SlaterDeterminant>& basis,
                            const H1View& H, const ERIView& V,
                            double tol)
{
    const std::size_t N = basis.size();
    if (N == 0) return CSR{};

    // Precompute combined determinants once
    std::vector<Determinant> Dlist; Dlist.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
        Dlist.push_back(combined_from(basis[i]));

    // Thread-local COO buffers
    const int nthreads =
    #ifdef _OPENMP
        omp_get_max_threads();
    #else
        1;
    #endif
    std::vector<COO> tl_coo(static_cast<std::size_t>(nthreads));

    // Naive double loop (outer-parallel if OpenMP is available)
    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        const int tid =
        #ifdef _OPENMP
            omp_get_thread_num();
        #else
            0;
        #endif
        COO& coo = tl_coo[static_cast<std::size_t>(tid)];

        const Determinant& Di = Dlist[static_cast<std::size_t>(i)];
        for (std::size_t j = 0; j < N; ++j) {
            const Determinant& Dj = Dlist[j];
            cx val = KL(Di, Dj, H, V);          // exact Slater–Condon dispatcher
            if (std::abs(val) >= tol) {
                coo.push(static_cast<int32_t>(i), static_cast<int32_t>(j), val, tol);
            }
        }
    }

    // Concatenate thread-local COO and pack to CSR
    COO all;
    std::size_t total = 0;
    for (const auto& c : tl_coo) total += c.val.size();
    all.row.reserve(total); all.col.reserve(total); all.val.reserve(total);
    for (auto& c : tl_coo) {
        all.row.insert(all.row.end(), c.row.begin(), c.row.end());
        all.col.insert(all.col.end(), c.col.begin(), c.col.end());
        all.val.insert(all.val.end(), c.val.begin(), c.val.end());
    }

    return coo_to_csr(N, N, std::move(all));
}

// 
namespace {
template<class Vec>
inline void dedup_sorted(Vec& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}
} // namespace

FixedBasisMatvec::FixedBasisMatvec(const std::vector<SlaterDeterminant>& basis,
                                   const H1View& H, const ERIView& V,
                                   bool enable_magnetic, double tol)
: H_(H), V_(V)
{
    N_ = basis.size();
    if (N_ == 0) return;
    M_ = basis[0].num_spatial_orbitals();

    // build connection structures once
    auto [Ia, Ib] = unique_alpha_beta(basis);
    Connections Ca = Caa(Ia, enable_magnetic);
    Connections Cb = Caa(Ib, enable_magnetic);
    BasisIndex BI  = make_basis_index(basis);

    D_.resize(N_);
    Hii_.resize(N_);
    cols_.resize(N_);

    // cache combined determinants and diagonals
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)N_; ++i) {
        D_[i] = combined_from(basis[(std::size_t)i]);
        std::vector<int> occK;
        occ_list_alpha_beta(basis[(std::size_t)i], occK);
        Hii_[i] = OO(H_, V_, occK);
    }

    // helper to push neighbors produced by swapping alpha or beta part
    auto push_from_parts = [&](const SpinDeterminant& A,
                               const SpinDeterminant& B,
                               const std::vector<SpinDeterminant>& setA,
                               std::vector<int32_t>& nbrs)
    {
        for (const auto& dA : setA) {
            Determinant ND(2 * M_);
            for (auto it = dA.begin_occ(); it != dA.end_occ(); ++it) ND.set((std::size_t)(*it));
            for (auto it = B.begin_occ();  it != B.end_occ();  ++it) ND.set((std::size_t)(M_ + *it));
            auto jt = BI.map.find(ND);
            if (jt != BI.map.end()) nbrs.push_back((int32_t)jt->second);
        }
    };

    // build per-row neighbor lists, deduped, without diagonal
    #pragma omp parallel for schedule(dynamic)
    for (ptrdiff_t irow = 0; irow < (ptrdiff_t)N_; ++irow) {
        const SlaterDeterminant& S = basis[(std::size_t)irow];
        const SpinDeterminant& Sa = S.alpha();
        const SpinDeterminant& Sb = S.beta();

        std::vector<int32_t> nbrs;
        nbrs.reserve(64);

        if (auto it = Ca.Cs.find(Sa); it != Ca.Cs.end()) push_from_parts(Sa, Sb, it->second, nbrs);
        if (auto it = Ca.Cd.find(Sa); it != Ca.Cd.end()) push_from_parts(Sa, Sb, it->second, nbrs);
        if (auto it = Cb.Cs.find(Sb); it != Cb.Cs.end()) push_from_parts(Sb, Sa, it->second, nbrs);
        if (auto it = Cb.Cd.find(Sb); it != Cb.Cd.end()) push_from_parts(Sb, Sa, it->second, nbrs);

        // double = single alpha + single beta
        if (auto ita = Ca.Cs.find(Sa); ita != Ca.Cs.end()) {
            if (auto itb = Cb.Cs.find(Sb); itb != Cb.Cs.end()) {
                for (const auto& dA : ita->second) {
                    for (const auto& dB : itb->second) {
                        Determinant ND(2 * M_);
                        for (auto it = dA.begin_occ(); it != dA.end_occ(); ++it) ND.set((std::size_t)(*it));
                        for (auto it = dB.begin_occ(); it != dB.end_occ(); ++it) ND.set((std::size_t)(M_ + *it));
                        auto jt = BI.map.find(ND);
                        if (jt != BI.map.end()) nbrs.push_back((int32_t)jt->second);
                    }
                }
            }
        }

        // magnetic interactions if any
        if (enable_magnetic) {
            if (auto itMa = Ca.Cmag.find(Sa); itMa != Ca.Cmag.end()) {
                if (auto itMb = Cb.Cmag.find(Sb); itMb != Cb.Cmag.end()) {
                    for (const auto& dA : itMa->second) {
                        for (const auto& dB : itMb->second) {
                            Determinant ND(2 * M_);
                            for (auto it = dA.begin_occ(); it != dA.end_occ(); ++it) ND.set((std::size_t)(*it));
                            for (auto it = dB.begin_occ(); it != dB.end_occ(); ++it) ND.set((std::size_t)(M_ + *it));
                            auto jt = BI.map.find(ND);
                            if (jt != BI.map.end()) {
                                // optional Hamming distance guard is already ensured by builder,
                                // add here if your Cmag can produce singles
                                nbrs.push_back((int32_t)jt->second);
                            }
                        }
                    }
                }
            }
        }

        // drop diagonal, then dedup
        nbrs.push_back((int32_t)irow);
        dedup_sorted(nbrs);
        if (!nbrs.empty() && nbrs.front() == (int32_t)irow) nbrs.erase(nbrs.begin());

        cols_[(std::size_t)irow] = std::move(nbrs);
    }
}

void FixedBasisMatvec::apply(const cx* x, cx* y) const
{
    if (N_ == 0) return;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)N_; ++i) y[(std::size_t)i] = cx(0.0, 0.0);

    const auto& H = H_;
    const auto& V = V_;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t irow = 0; irow < (ptrdiff_t)N_; ++irow) {
        cx acc = Hii_[(std::size_t)irow] * x[(std::size_t)irow];

        const Determinant& Di = D_[(std::size_t)irow];
        const auto& nbrs = cols_[(std::size_t)irow];

        for (int32_t jj : nbrs) {
            // Dj is just the combined det of the neighbor basis det, which we cached
            const Determinant& Dj = D_[(std::size_t)jj];
            cx hij = KL(Di, Dj, H, V);
            if (std::abs(hij) != 0.0) acc += hij * x[(std::size_t)jj];
        }
        y[(std::size_t)irow] = acc;
    }
}

} // namespace ci