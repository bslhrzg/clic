// slater_condon.cpp
#include "slater_condon.h"
#include <bit>      // std::popcount
#include <algorithm>

namespace ci {

// ----- helpers -----

static inline std::size_t popcount_between_exclusive(const BitsetVec& b,
                                                     int x, int y) noexcept {
    // count set bits in (min, max) exclusive => [min+1, max)
    int lo = std::min(x, y);
    int hi = std::max(x, y);
    if (hi - lo <= 1) return 0;
    // popcount in [0, hi) minus popcount in [0, lo+1)
    return b.popcount_below(static_cast<std::size_t>(hi))
         - b.popcount_below(static_cast<std::size_t>(lo + 1));
}

int parity_single(const Determinant& I, int a, int r) noexcept {
    const auto& bits = I.bits();
    const std::size_t c = popcount_between_exclusive(bits, a, r);
    return static_cast<int>(c & 1u);
}

int parity_double(const Determinant& I, int a, int r, int b, int s) noexcept {
    const auto& bits = I.bits();

    // First move (a -> r)
    int p1 = parity_single(I, a, r);

    // For the second move (b -> s), the intermediate state differs at a,r only.
    // Count occ strictly between (b,s) on I, then correct for (a,r) if they lie inside.
    int lo = std::min(b, s);
    int hi = std::max(b, s);

    std::size_t base = popcount_between_exclusive(bits, b, s);

    // a: 1 -> 0 in the intermediate
    if (hi - lo > 1 && a > lo && a < hi) {
        // Only subtract if a was occupied in I (it is, for a->r to be valid)
        base = (base > 0) ? (base - 1) : 0;
    }
    // r: 0 -> 1 in the intermediate
    if (hi - lo > 1 && r > lo && r < hi) {
        base += 1;
    }

    int p2 = static_cast<int>(base & 1u);
    return (p1 + p2) & 1;
}

// --------------------------- Slater–Condon kernels ---------------------------

cx OO(const H1View& H, const ERIView& V, const std::vector<int>& occK) noexcept {
    // e1 = sum H[i,i]
    cx e1 = cx{0.0, 0.0};
    for (int i : occK) e1 += H(static_cast<std::size_t>(i), static_cast<std::size_t>(i));

    // e2 = sum over i,j ( (ij|ij) - (ij|ji) )
    cx e2 = cx{0.0, 0.0};
    const std::size_t n = occK.size();
    for (std::size_t a = 0; a < n; ++a) {
        const int i = occK[a];
        for (std::size_t b = 0; b < n; ++b) {
            const int j = occK[b];
            e2 += V(i,j,i,j) - V(i,j,j,i);
        }
    }
    return e1 + e2 * 0.5; // divide by 2 as in Julia OO
}

cx OS(const H1View& H, const ERIView& V, int a, int r,
      const std::vector<int>& occK) noexcept
{
    // e1 = H[a,r]
    cx e1 = H(static_cast<std::size_t>(a), static_cast<std::size_t>(r));

    // e2 = sum_i ( (a i | r i) - (a i | i r) )
    cx e2 = cx{0.0, 0.0};
    for (int i : occK) {
        e2 += V(a,i,r,i) - V(a,i,i,r);
    }
    return e1 + e2;
}

cx OD(const ERIView& V, int m, int n, int p, int q) noexcept {
    // (mn|pq) - (mn|qp)
    return V(m,n,p,q) - V(m,n,q,p);
}

// ------------------------------ KL dispatcher --------------------------------

static inline int hamming(const Determinant& A, const Determinant& B) noexcept {
    const auto& wa = A.bits().words();
    const auto& wb = B.bits().words();
    const std::size_t nw = wa.size();
    std::size_t c = 0;
    for (std::size_t w = 0; w < nw; ++w) c += std::popcount(wa[w] ^ wb[w]);
    return static_cast<int>(c);
}

static inline void differing_positions(const Determinant& D1, const Determinant& D2,
                                       std::vector<int>& from_D1, std::vector<int>& from_D2)
{
    from_D1.clear(); from_D2.clear();
    const auto& wa = D1.bits().words();
    const auto& wb = D2.bits().words();
    const std::size_t nw = wa.size();
    for (std::size_t w = 0; w < nw; ++w) {
        std::uint64_t x = wa[w] ^ wb[w];
        while (x) {
            unsigned tz = std::countr_zero(x);
            int bit = static_cast<int>(w*64 + tz);
            bool in1 = ((wa[w] >> tz) & 1ull) != 0ull;
            if (in1) from_D1.push_back(bit);
            else     from_D2.push_back(bit);
            x &= (x - 1); // clear lowest set bit
        }
    }
    // Preserve low-to-high discovery order to match your Julia diffbit behavior
    std::sort(from_D1.begin(), from_D1.end());
    std::sort(from_D2.begin(), from_D2.end());
}

static inline void occ_list(const Determinant& D, std::vector<int>& occK_out) {
    occK_out.clear();
    for (auto it = D.begin_occ(); it != D.end_occ(); ++it)
        occK_out.push_back(*it);
}

cx KL(const Determinant& D1, const Determinant& D2,
      const H1View& H, const ERIView& V) noexcept
{
    const int ham = hamming(D1, D2);
    if (ham == 0) {
        std::vector<int> occK;
        occ_list(D1, occK);
        return OO(H, V, occK);
    }

    std::vector<int> from1, from2;
    differing_positions(D1, D2, from1, from2);

    if (ham == 2) {
        // Single excitation: from1 = {a}, from2 = {r}
        if (from1.size() != 1 || from2.size() != 1) return cx{0.0, 0.0};
        int a = from1[0], r = from2[0];
        const int p = parity_single(D1, a, r);
        std::vector<int> occK;
        occ_list(D1, occK);
        return ((p & 1) ? cx{-1.0, 0.0} : cx{1.0, 0.0}) * OS(H, V, a, r, occK);
    }

    if (ham == 4) {
        // Double excitation: from1 = {m,n}, from2 = {p,q}
        if (from1.size() != 2 || from2.size() != 2) return cx{0.0, 0.0};
        int m = from1[0], n = from1[1];
        int p = from2[0], q = from2[1];
        const int s = parity_double(D1, m, p, n, q); // matches Julia's get_p1p2 order
        return ((s & 1) ? cx{-1.0, 0.0} : cx{1.0, 0.0}) * OD(V, m, n, p, q);
    }

    return cx{0.0, 0.0};
}

} // namespace ci