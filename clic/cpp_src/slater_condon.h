#pragma once
// slater_condon.hpp
// -----------------------------------------------------------------------------
// Slater–Condon rules and phase utilities on ci::Determinant (combined K=2M).
// - 0-based indices
// - AlphaFirst spin-orbital ordering (α: 0..M-1, β: M..2M-1)
// - Complex-valued one- and two-electron integrals
// - Exact port of your Julia OO/OS/OD kernels and phase logic
// -----------------------------------------------------------------------------

#include "ci_core.h"
#include <complex>
#include <vector>
#include <cstddef>

namespace ci {

using cx = std::complex<double>;

// Lightweight views over dense row-major arrays (no ownership)
struct H1View {
    const cx* ptr{nullptr};
    std::size_t K{0};            // number of spin-orbitals
    inline const cx& operator()(std::size_t i, std::size_t j) const noexcept {
        return ptr[i*K + j];
    }
};

struct ERIView {
    const cx* ptr{nullptr};
    std::size_t K{0};            // number of spin-orbitals
    inline const cx& operator()(std::size_t i, std::size_t j,
                                std::size_t k, std::size_t l) const noexcept {
        return ptr[(((i*K + j)*K + k)*K + l)];
    }
};

// --------------------------- Phase (parity) utilities ------------------------

// (-1)^{ #occ strictly between min(a,r) and max(a,r) in combined determinant }
int parity_single(const Determinant& I, int a, int r) noexcept;

// Double excitation parity, applying (a->r) first then (b->s) on the intermediate.
// Matches Julia's get_p1 + excite_psi_a_r + get_p1.
int parity_double(const Determinant& I, int a, int r, int b, int s) noexcept;

// -------------------------- Slater–Condon kernels ----------------------------

// Diagonal <D|H|D>  (OO)
cx OO(const H1View& H, const ERIView& V, const std::vector<int>& occK) noexcept;

// Single excitation <D|H|D(a->r)>  (OS)
cx OS(const H1View& H, const ERIView& V, int a, int r,
      const std::vector<int>& occK) noexcept;

// Double excitation <D|H|D({m,n}->{p,q})>  (OD)
cx OD(const ERIView& V, int m, int n, int p, int q) noexcept;

// Dispatcher: returns <D1|H|D2>. Returns 0 if Hamming distance ∉ {0,2,4}.
cx KL(const Determinant& D1, const Determinant& D2,
      const H1View& H, const ERIView& V) noexcept;

} // namespace ci