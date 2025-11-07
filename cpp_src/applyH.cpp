// cpp_src/applyH.cpp
#include "applyH.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <unordered_set>
#include <vector>

namespace ci {

// --- Helper function to solve the 'combined_from' error ---
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

} // namespace ci