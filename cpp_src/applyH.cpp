// cpp_src/applyH.cpp
#include "applyH.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace ci {

// This function was previously local to hamiltonian.cpp. We need it here as well.
static inline Determinant combined_from(const SlaterDeterminant& sd) {
    const size_t M = sd.num_spatial_orbitals();
    Determinant D(2 * M);
    for (auto it = sd.alpha().begin_occ(); it != sd.alpha().end_occ(); ++it)
        D.set(static_cast<size_t>(*it));
    for (auto it = sd.beta().begin_occ(); it != sd.beta().end_occ(); ++it)
        D.set(static_cast<size_t>(M + *it));
    return D;
}


// Direct C++ translation of your Python `build_*` functions
ScreenedHamiltonian build_screened_hamiltonian(
    size_t K, const H1View& H, const ERIView& V, double tol)
{
    ScreenedHamiltonian sh;
    sh.n_spin_orbitals = K;

    // --- TableSh0 (build_Sh0) ---
    for (uint32_t r = 0; r < K; ++r) {
        std::vector<uint32_t> targets;
        for (uint32_t p = 0; p < K; ++p) {
            if (p == r) continue;
            if (std::abs(H(p, r)) >= tol) {
                targets.push_back(p);
            }
        }
        if (!targets.empty()) {
            sh.sh0[r] = std::move(targets);
        }
    }

    // --- TableSU (build_SU detailed=True) ---
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = 0; j < K; ++j) {
            if (i == j) continue;
            std::vector<uint32_t> spectators;
            for (uint32_t p = 0; p < K; ++p) {
                // Exact translation of the 4 terms checked in your Python `build_SU`
                if (std::abs(V(j, p, p, i)) > tol ||
                    std::abs(V(j, p, i, p)) > tol ||
                    std::abs(V(p, j, p, i)) > tol ||
                    std::abs(V(p, j, i, p)) > tol) {
                    spectators.push_back(p);
                }
            }
            if (!spectators.empty()) {
                sh.su[i][j] = std::move(spectators);
            }
        }
    }

    // --- TableD (build_D) ---
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = i + 1; j < K; ++j) {
            uint64_t key_ij = i | (static_cast<uint64_t>(j) << 32);
            std::vector<std::pair<uint32_t, uint32_t>> targets;
            for (uint32_t k = 0; k < K; ++k) {
                for (uint32_t l = 0; l < K; ++l) {
                    if (k == l || k == i || k == j || l == i || l == j) continue;
                    // Exact translation of the 4 terms checked in your Python `build_D`
                    if (std::abs(V(k, l, i, j)) > tol ||
                        std::abs(V(l, k, i, j)) > tol ||
                        std::abs(V(k, l, j, i)) > tol ||
                        std::abs(V(l, k, j, i)) > tol) {
                        targets.emplace_back(k, l);
                    }
                }
            }
            if (!targets.empty()) {
                sh.d[key_ij] = std::move(targets);
            }
        }
    }
    return sh;
}


Wavefunction apply_hamiltonian(
    const Wavefunction& psi,
    const ScreenedHamiltonian& sh,
    const H1View& H,
    const ERIView& V,
    double tol_element)
{
    using Coeff = Wavefunction::Coeff;
    using Map = std::unordered_map<SlaterDeterminant, Coeff>;

    const size_t M = psi.num_spatial_orbitals();
    const size_t K = sh.n_spin_orbitals;

    std::vector<std::pair<SlaterDeterminant, Coeff>> items;
    items.reserve(psi.data().size());
    for (const auto& kv : psi.data()) {
        items.emplace_back(kv.first, kv.second);
    }

    int T = 1;
    #ifdef _OPENMP
    T = omp_get_max_threads();
    #endif
    std::vector<Map> local_maps(T);

    #pragma omp parallel for schedule(dynamic)
    for (size_t item_idx = 0; item_idx < items.size(); ++item_idx) {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& acc = local_maps[tid];
        const auto& [det_in, coeff] = items[item_idx];

        std::unordered_set<SlaterDeterminant> seen;
        
        std::vector<int> occ_so_vec;
        occ_so_vec.reserve(K);
        for (auto it = det_in.alpha().begin_occ(); it != det_in.alpha().end_occ(); ++it) occ_so_vec.push_back(*it);
        for (auto it = det_in.beta().begin_occ(); it != det_in.beta().end_occ(); ++it) occ_so_vec.push_back(*it + static_cast<int>(M));
        std::sort(occ_so_vec.begin(), occ_so_vec.end());

        std::unordered_set<int> occ_so_set(occ_so_vec.begin(), occ_so_vec.end());
        
        Determinant d_in_comb(K, occ_so_vec);
        SlaterDeterminant det_out_scratch(M);
        int8_t sign; // FIX 2: Declare a proper sign variable

        // 1) Diagonal
        cx val_diag = KL(d_in_comb, d_in_comb, H, V);
        if (std::abs(val_diag) > 0) acc[det_in] += coeff * val_diag;
        seen.insert(det_in);

        // 2) Singles from Sh0
        for (int r : occ_so_vec) {
            auto it = sh.sh0.find(r);
            if (it == sh.sh0.end()) continue;
            for (uint32_t p : it->second) {
                if (occ_so_set.count(p)) continue; 

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) { // FIX 2: Pass the sign variable

                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch); // FIX 1: Now works
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                }
            }
        }

        // 3) Singles from SU
        for (int r : occ_so_vec) {
            auto it_r = sh.su.find(r);
            if (it_r == sh.su.end()) continue;
            for (const auto& [p, spectators] : it_r->second) {
                if (occ_so_set.count(p)) continue; 

                bool spectator_ok = false;
                for (uint32_t s : spectators) {
                    if (occ_so_set.count(s)) { spectator_ok = true; break; }
                }
                if (!spectator_ok) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) { // FIX 2: Pass the sign variable

                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch); // FIX 1: Now works
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                }
            }
        }

        // 4) Doubles from D
        for (size_t i = 0; i < occ_so_vec.size(); ++i) {
            for (size_t j = i + 1; j < occ_so_vec.size(); ++j) {
                uint32_t r = occ_so_vec[i];
                uint32_t s = occ_so_vec[j];
                uint64_t key_rs = r | (static_cast<uint64_t>(s) << 32);

                auto it = sh.d.find(key_rs);
                if (it == sh.d.end()) continue;

                for (const auto& [p, q] : it->second) {
                    if (occ_so_set.count(p) || occ_so_set.count(q)) continue;

                    if (SlaterDeterminant::apply_excitation_double_fast(
                            det_in, p % M, q % M, r % M, s % M,
                            p < M ? Spin::Alpha : Spin::Beta, q < M ? Spin::Alpha : Spin::Beta,
                            r < M ? Spin::Alpha : Spin::Beta, s < M ? Spin::Alpha : Spin::Beta,
                            det_out_scratch, sign)) { // FIX 2: Pass the sign variable
                        
                        if (seen.count(det_out_scratch)) continue;
                        seen.insert(det_out_scratch);

                        Determinant d_out_comb = combined_from(det_out_scratch); // FIX 1: Now works
                        cx val = KL(d_in_comb, d_out_comb, H, V);
                        if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                    }
                }
            }
        }
    }

    // --- Merge results ---
    Map final_acc;
    size_t hint = 0;
    for(const auto& m : local_maps) hint += m.size();
    final_acc.reserve(hint);
    for (const auto& m : local_maps) {
        for (const auto& kv : m) {
            final_acc[kv.first] += kv.second;
        }
    }

    Wavefunction out(M);
    for (const auto& kv : final_acc) {
        out.add_term(kv.first, kv.second, tol_element);
    }
    return out;
}

// ============================
// New fixed basis support
// ============================

struct AllowedSets {
    std::unordered_set<uint32_t> spatial;
    std::unordered_set<uint32_t> spinorbital;
};

static AllowedSets collect_allowed_from_basis(const std::vector<SlaterDeterminant>& basis, size_t M) {
    AllowedSets A;
    A.spatial.reserve(M);
    for (const auto& sd : basis) {
        for (auto it = sd.alpha().begin_occ(); it != sd.alpha().end_occ(); ++it) A.spatial.insert(static_cast<uint32_t>(*it));
        for (auto it = sd.beta().begin_occ();  it != sd.beta().end_occ();  ++it) A.spatial.insert(static_cast<uint32_t>(*it));
    }
    A.spinorbital.reserve(2 * A.spatial.size());
    for (uint32_t p : A.spatial) {
        A.spinorbital.insert(p);
        A.spinorbital.insert(p + static_cast<uint32_t>(M));
    }
    return A;
}

ScreenedHamiltonian build_fixed_basis_tables(
    const ScreenedHamiltonian& sh_full,
    const std::vector<SlaterDeterminant>& basis,
    size_t M)
{
    const auto A = collect_allowed_from_basis(basis, M);
    const uint32_t K = static_cast<uint32_t>(2 * M);

    ScreenedHamiltonian sh_fb;
    sh_fb.n_spin_orbitals = sh_full.n_spin_orbitals;

    // sh0
    for (uint32_t r = 0; r < K; ++r) {
        if (!A.spinorbital.count(r)) continue;
        auto it = sh_full.sh0.find(r);
        if (it == sh_full.sh0.end()) continue;

        std::vector<uint32_t> filtered;
        filtered.reserve(it->second.size());
        for (uint32_t p : it->second) if (A.spinorbital.count(p)) filtered.push_back(p);
        if (!filtered.empty()) sh_fb.sh0[r] = std::move(filtered);
    }

    // su
    for (const auto& rij : sh_full.su) {
        uint32_t i = rij.first;
        if (!A.spinorbital.count(i)) continue;
        for (const auto& pj : rij.second) {
            uint32_t j = pj.first;
            if (!A.spinorbital.count(j)) continue;

            std::vector<uint32_t> spectators_fb;
            spectators_fb.reserve(pj.second.size());
            for (uint32_t s : pj.second) if (A.spinorbital.count(s)) spectators_fb.push_back(s);
            if (!spectators_fb.empty()) sh_fb.su[i][j] = std::move(spectators_fb);
        }
    }

    // d
    for (const auto& kv : sh_full.d) {
        const uint64_t key = kv.first;
        const uint32_t r = static_cast<uint32_t>(key & 0xffffffffULL);
        const uint32_t s = static_cast<uint32_t>((key >> 32) & 0xffffffffULL);
        if (!A.spinorbital.count(r) || !A.spinorbital.count(s)) continue;

        std::vector<std::pair<uint32_t,uint32_t>> filtered;
        filtered.reserve(kv.second.size());
        for (auto [p,q] : kv.second) if (A.spinorbital.count(p) && A.spinorbital.count(q)) filtered.emplace_back(p,q);
        if (!filtered.empty()) sh_fb.d[r | (static_cast<uint64_t>(s) << 32)] = std::move(filtered);
    }

    return sh_fb;
}

Wavefunction apply_hamiltonian_fixed_basis(
    const Wavefunction& psi,
    const ScreenedHamiltonian& sh_fixed_basis,
    const std::vector<SlaterDeterminant>& basis,
    const H1View& H,
    const ERIView& V,
    double tol_element)
{
    using Coeff = Wavefunction::Coeff;
    using Map   = std::unordered_map<SlaterDeterminant, Coeff>;

    const size_t M = psi.num_spatial_orbitals();
    const size_t K = sh_fixed_basis.n_spin_orbitals;

    std::unordered_set<SlaterDeterminant> basis_set;
    basis_set.reserve(basis.size() * 2);
    for (const auto& d : basis) basis_set.insert(d);

    std::vector<std::pair<SlaterDeterminant, Coeff>> items;
    items.reserve(psi.data().size());
    for (const auto& kv : psi.data()) if (basis_set.count(kv.first)) items.emplace_back(kv.first, kv.second);

    int T = 1;
    #ifdef _OPENMP
    T = omp_get_max_threads();
    #endif
    std::vector<Map> local_maps(T);

    #pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < items.size(); ++idx) {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& acc = local_maps[tid];

        const auto& det_in  = items[idx].first;
        const auto& coeff   = items[idx].second;

        std::unordered_set<SlaterDeterminant> seen;
        seen.reserve(64);

        std::vector<int> occ_so_vec;
        occ_so_vec.reserve(K);
        for (auto it = det_in.alpha().begin_occ(); it != det_in.alpha().end_occ(); ++it) occ_so_vec.push_back(*it);
        for (auto it = det_in.beta().begin_occ(); it != det_in.beta().end_occ();  ++it) occ_so_vec.push_back(*it + static_cast<int>(M));
        std::sort(occ_so_vec.begin(), occ_so_vec.end());
        std::unordered_set<int> occ_so_set(occ_so_vec.begin(), occ_so_vec.end());

        Determinant d_in_comb(K, occ_so_vec);
        SlaterDeterminant det_out_scratch(M);
        int8_t sign = 1;

        // diagonal
        cx val_diag = KL(d_in_comb, d_in_comb, H, V);
        if (std::abs(val_diag) > 0) acc[det_in] += coeff * val_diag;
        seen.insert(det_in);

        // singles from sh0
        for (int r : occ_so_vec) {
            auto it = sh_fixed_basis.sh0.find(static_cast<uint32_t>(r));
            if (it == sh_fixed_basis.sh0.end()) continue;

            for (uint32_t p : it->second) {
                if (occ_so_set.count(static_cast<int>(p))) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) {

                    if (!basis_set.count(det_out_scratch)) continue;
                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch);
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                }
            }
        }

        // singles from su
        for (int r : occ_so_vec) {
            auto it_r = sh_fixed_basis.su.find(static_cast<uint32_t>(r));
            if (it_r == sh_fixed_basis.su.end()) continue;

            for (const auto& [p, spectators] : it_r->second) {
                if (occ_so_set.count(static_cast<int>(p))) continue;

                bool ok = false;
                for (uint32_t s : spectators) {
                    if (occ_so_set.count(static_cast<int>(s))) { ok = true; break; }
                }
                if (!ok) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) {

                    if (!basis_set.count(det_out_scratch)) continue;
                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch);
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                }
            }
        }

        // doubles from d
        for (size_t a = 0; a < occ_so_vec.size(); ++a) {
            for (size_t b = a + 1; b < occ_so_vec.size(); ++b) {
                uint32_t r = static_cast<uint32_t>(occ_so_vec[a]);
                uint32_t s = static_cast<uint32_t>(occ_so_vec[b]);
                uint64_t key_rs = r | (static_cast<uint64_t>(s) << 32);

                auto it = sh_fixed_basis.d.find(key_rs);
                if (it == sh_fixed_basis.d.end()) continue;

                for (const auto& [p, q] : it->second) {
                    if (occ_so_set.count(static_cast<int>(p)) || occ_so_set.count(static_cast<int>(q))) continue;

                    if (SlaterDeterminant::apply_excitation_double_fast(
                            det_in, p % M, q % M, r % M, s % M,
                            p < M ? Spin::Alpha : Spin::Beta, q < M ? Spin::Alpha : Spin::Beta,
                            r < M ? Spin::Alpha : Spin::Beta, s < M ? Spin::Alpha : Spin::Beta,
                            det_out_scratch, sign)) {

                        if (!basis_set.count(det_out_scratch)) continue;
                        if (seen.count(det_out_scratch)) continue;
                        seen.insert(det_out_scratch);

                        Determinant d_out_comb = combined_from(det_out_scratch);
                        cx val = KL(d_in_comb, d_out_comb, H, V);
                        if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                    }
                }
            }
        }
    }

    // merge
    std::unordered_map<SlaterDeterminant, Coeff> final_acc;
    size_t hint = 0;
    for (const auto& m : local_maps) hint += m.size();
    final_acc.reserve(hint);
    for (const auto& m : local_maps)
        for (const auto& kv : m) final_acc[kv.first] += kv.second;

    Wavefunction out(M);
    for (const auto& kv : final_acc) out.add_term(kv.first, kv.second, tol_element);
    return out;
}


// =============================
// Direct sparse Ham
// =============================


// local helper
static inline uint64_t pack_rs(uint32_t r, uint32_t s) {
    if (r < s) return r | (uint64_t(s) << 32);
    return s | (uint64_t(r) << 32);
}

FixedBasisCSR build_fixed_basis_csr(
    const ScreenedHamiltonian& sh_fb,
    const std::vector<SlaterDeterminant>& basis,
    const H1View& H,
    const ERIView& V)
{
    const size_t N = basis.size();
    if (N == 0) return FixedBasisCSR{0, {0}, {}, {}};

    const size_t M = basis.front().num_spatial_orbitals();
    const size_t K = 2 * M;

    // map det -> index
    std::unordered_map<SlaterDeterminant, int> idx;
    idx.reserve(N * 2);
    for (int i = 0; i < (int)N; ++i) idx.emplace(basis[i], i);

    // precompute combined and occupied spin orbitals
    std::vector<Determinant> Dcomb(N, Determinant(K));
    std::vector<std::vector<int>> occ_so(N);
    for (size_t i = 0; i < N; ++i) {
        Dcomb[i] = combined_from(basis[i]);
        auto& v = occ_so[i];
        v.reserve(K);
        for (auto it = basis[i].alpha().begin_occ(); it != basis[i].alpha().end_occ(); ++it) v.push_back(*it);
        for (auto it = basis[i].beta().begin_occ();  it != basis[i].beta().end_occ();  ++it) v.push_back(*it + int(M));
        std::sort(v.begin(), v.end());
    }

    int T = 1;
    #ifdef _OPENMP
    T = omp_get_max_threads();
    #endif
    std::vector<std::vector<int64_t>> t_rows(T), t_cols(T);
    std::vector<std::vector<cx>>      t_vals(T);

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < (int)N; ++ii) {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& rows = t_rows[tid];
        auto& cols = t_cols[tid];
        auto& vals = t_vals[tid];

        const auto& Di = Dcomb[ii];
        const auto& oi = occ_so[ii];
        std::unordered_set<int> oi_set(oi.begin(), oi.end());

        // NEW: prevent double counting the same neighbor |Dj> from different tables
        std::unordered_set<int> seen_cols;
        seen_cols.reserve(64);
        seen_cols.insert(ii); // diagonal

        // diagonal
        {
            cx hii = KL(Di, Di, H, V);
            if (std::abs(hii) > 0) {
                rows.push_back(ii); cols.push_back(ii); vals.push_back(hii);
            }
        }

        SlaterDeterminant tmp(M);
        int8_t sign = 1;

        // singles via Sh0
        for (int r : oi) {
            auto it = sh_fb.sh0.find((uint32_t)r);
            if (it == sh_fb.sh0.end()) continue;
            for (uint32_t p : it->second) {
                if (oi_set.count((int)p)) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        basis[ii], p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        tmp, sign))
                {
                    auto jt = idx.find(tmp);
                    if (jt == idx.end()) continue;
                    int jj = jt->second;
                    if (jj < ii) continue;
                    if (seen_cols.count(jj)) continue;            // <-- guard
                    seen_cols.insert(jj);

                    Determinant Dj = Di;
                    Dj.reset((size_t)r);
                    Dj.set((size_t)p);

                    cx hij = KL(Di, Dj, H, V);
                    if (std::abs(hij) > 0) {
                        rows.push_back(ii); cols.push_back(jj); vals.push_back(hij);
                        if (jj != ii) {
                            rows.push_back(jj); cols.push_back(ii); vals.push_back(std::conj(hij));
                        }
                    }
                }
            }
        }

        // singles via SU
        for (int r : oi) {
            auto it_r = sh_fb.su.find((uint32_t)r);
            if (it_r == sh_fb.su.end()) continue;

            for (const auto& [p, spectators] : it_r->second) {
                if (oi_set.count((int)p)) continue;

                bool ok = false;
                for (uint32_t s : spectators) { if (oi_set.count((int)s)) { ok = true; break; } }
                if (!ok) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        basis[ii], p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        tmp, sign))
                {
                    auto jt = idx.find(tmp);
                    if (jt == idx.end()) continue;
                    int jj = jt->second;
                    if (jj < ii) continue;
                    if (seen_cols.count(jj)) continue;            // <-- guard
                    seen_cols.insert(jj);

                    Determinant Dj = Di;
                    Dj.reset((size_t)r);
                    Dj.set((size_t)p);

                    cx hij = KL(Di, Dj, H, V);
                    if (std::abs(hij) > 0) {
                        rows.push_back(ii); cols.push_back(jj); vals.push_back(hij);
                        if (jj != ii) {
                            rows.push_back(jj); cols.push_back(ii); vals.push_back(std::conj(hij));
                        }
                    }
                }
            }
        }

        // doubles via D
        for (size_t a = 0; a < oi.size(); ++a) {
            for (size_t b = a + 1; b < oi.size(); ++b) {
                uint32_t r = (uint32_t)oi[a], s = (uint32_t)oi[b];
                uint64_t key = (r < s) ? (r | (uint64_t(s) << 32)) : (s | (uint64_t(r) << 32));
                auto it = sh_fb.d.find(key);
                if (it == sh_fb.d.end()) continue;

                for (const auto& [p, q] : it->second) {
                    if (oi_set.count((int)p) || oi_set.count((int)q)) continue;

                    if (SlaterDeterminant::apply_excitation_double_fast(
                            basis[ii], p % M, q % M, r % M, s % M,
                            p < M ? Spin::Alpha : Spin::Beta, q < M ? Spin::Alpha : Spin::Beta,
                            r < M ? Spin::Alpha : Spin::Beta, s < M ? Spin::Alpha : Spin::Beta,
                            tmp, sign))
                    {
                        auto jt = idx.find(tmp);
                        if (jt == idx.end()) continue;
                        int jj = jt->second;
                        if (jj < ii) continue;
                        if (seen_cols.count(jj)) continue;        // <-- guard (paranoid but safe)
                        seen_cols.insert(jj);

                        Determinant Dj = Di;
                        Dj.reset((size_t)r); Dj.reset((size_t)s);
                        Dj.set((size_t)p);   Dj.set((size_t)q);

                        cx hij = KL(Di, Dj, H, V);
                        if (std::abs(hij) > 0) {
                            rows.push_back(ii); cols.push_back(jj); vals.push_back(hij);
                            if (jj != ii) {
                                rows.push_back(jj); cols.push_back(ii); vals.push_back(std::conj(hij));
                            }
                        }
                    }
                }
            }
        }
    }


    // merge thread-local COO and convert to CSR
    size_t nnz = 0;
    for (int t = 0; t < T; ++t) nnz += t_vals[t].size();

    std::vector<int64_t> rows; rows.reserve(nnz);
    std::vector<int32_t> cols; cols.reserve(nnz);
    std::vector<cx>      vals; vals.reserve(nnz);
    for (int t = 0; t < T; ++t) {
        rows.insert(rows.end(), t_rows[t].begin(), t_rows[t].end());
        cols.insert(cols.end(), t_cols[t].begin(), t_cols[t].end());
        vals.insert(vals.end(), t_vals[t].begin(), t_vals[t].end());
    }

    // sort by (row, col)
    std::vector<size_t> order(rows.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b){
        if (rows[a] != rows[b]) return rows[a] < rows[b];
        return cols[a] < cols[b];
    });

    FixedBasisCSR A;
    A.N = N;
    A.indptr.assign(N + 1, 0);

    int64_t cur_row = -1;
    int32_t cur_col = -1;
    cx      acc     = cx(0.0, 0.0);

    auto flush = [&](int64_t r, int32_t c, cx v) {
        if (std::abs(v) == 0) return;
        A.indices.push_back(c);
        A.data.push_back(v);
        ++A.indptr[r + 1];
    };

    for (size_t k = 0; k < order.size(); ++k) {
        size_t i = order[k];
        int64_t r = rows[i];
        int32_t c = cols[i];
        cx v = vals[i];

        if (r != cur_row || c != cur_col) {
            if (cur_row >= 0) flush(cur_row, cur_col, acc);
            cur_row = r; cur_col = c; acc = v;
        } else {
            acc += v;
        }
    }
    if (cur_row >= 0) flush(cur_row, cur_col, acc);

    // prefix sum
    for (size_t r = 0; r < N; ++r) A.indptr[r + 1] += A.indptr[r];

    return A;
}

void csr_matvec(const FixedBasisCSR& A, const cx* x, cx* y) {
    const size_t N = A.N;
    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)N; ++i) {
        cx sum = cx(0.0, 0.0);
        const auto start = A.indptr[i];
        const auto end   = A.indptr[i + 1];
        for (auto k = start; k < end; ++k) sum += A.data[k] * x[A.indices[k]];
        y[i] = sum;
    }
}



FixedBasisCSR build_fixed_basis_csr_full(
    const std::vector<SlaterDeterminant>& basis,
    size_t M,
    const H1View& H,
    const ERIView& V,
    double tol_tables,
    double drop_tol)
{
    const size_t K = 2 * M;

    // 1) Full screened tables (complete adjacency within tol_tables)
    ScreenedHamiltonian sh_full = build_screened_hamiltonian(K, H, V, tol_tables);

    // 2) Restrict to fixed-basis orbitals
    ScreenedHamiltonian sh_fb = build_fixed_basis_tables(sh_full, basis, M);

    // 3) Build CSR
    FixedBasisCSR A = build_fixed_basis_csr(sh_fb, basis, H, V);

    // 4) Optional pruning of tiny entries (keeps Hermiticity because we stored symmetrically)
    if (drop_tol > 0.0) {
        const size_t N = A.N;
        std::vector<int64_t> new_indptr(N + 1, 0);
        std::vector<int32_t> new_indices;
        std::vector<cx>      new_data;
        new_indices.reserve(A.indices.size());
        new_data.reserve(A.data.size());

        for (size_t i = 0; i < N; ++i) {
            const auto start = A.indptr[i];
            const auto end   = A.indptr[i + 1];
            for (auto k = start; k < end; ++k) {
                const cx v = A.data[k];
                if (std::abs(v) > drop_tol) {
                    new_indices.push_back(A.indices[k]);
                    new_data.push_back(v);
                    ++new_indptr[i + 1];
                }
            }
        }
        for (size_t i = 0; i < N; ++i) new_indptr[i + 1] += new_indptr[i];

        A.indices.swap(new_indices);
        A.data.swap(new_data);
        A.indptr.swap(new_indptr);
    }

    return A;
}


} // namespace ci