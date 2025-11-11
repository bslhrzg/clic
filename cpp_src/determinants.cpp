// ci_core.cpp
#include "ci_core.h"

#include <unordered_set>
#include <bit>          // std::popcount, std::countr_zero
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <bitset>

#include <omp.h>

namespace ci {

// ============================== BitsetVec ====================================

BitsetVec::BitsetVec() = default;

BitsetVec::BitsetVec(std::size_t n_bits)
    : n_bits_(n_bits), words_((n_bits + 63) / 64, u64{0}) {}

std::size_t BitsetVec::size()   const noexcept { return n_bits_; }
std::size_t BitsetVec::nwords() const noexcept { return words_.size(); }

bool BitsetVec::test(std::size_t bit) const noexcept {
    assert(bit < n_bits_);
    const auto w = bit / 64, b = bit % 64;
    return (words_[w] >> b) & u64{1};
}

void BitsetVec::set(std::size_t bit) noexcept {
    assert(bit < n_bits_);
    const auto w = bit / 64, b = bit % 64;
    words_[w] |= (u64{1} << b);
}

void BitsetVec::reset(std::size_t bit) noexcept {
    assert(bit < n_bits_);
    const auto w = bit / 64, b = bit % 64;
    words_[w] &= ~(u64{1} << b);
}

void BitsetVec::clear() noexcept {
    std::fill(words_.begin(), words_.end(), u64{0});
}

std::size_t BitsetVec::popcount_all() const noexcept {
    std::size_t c = 0;
    for (u64 w : words_) c += std::popcount(w);
    return c;
}

// Count set bits in [0, bit_exclusive)
std::size_t BitsetVec::popcount_below(std::size_t bit_exclusive) const noexcept {
    if (bit_exclusive == 0) return 0;
    auto idx = bit_exclusive - 1;
    const auto wlim = idx / 64;
    const auto blim = idx % 64;
    std::size_t c = 0;
    for (std::size_t w = 0; w < wlim; ++w) c += std::popcount(words_[w]);
    const u64 mask = (blim == 63) ? ~u64{0} : ((u64{1} << (blim + 1)) - 1);
    c += std::popcount(words_[wlim] & mask);
    return c;
}

void BitsetVec::mask_tail() noexcept {
    if (words_.empty()) return;
    const std::size_t valid = n_bits_ % 64;
    if (valid == 0) return;
    const u64 mask = (u64{1} << valid) - 1;
    words_.back() &= mask;
}

std::string BitsetVec::to_binary() const {
    std::stringstream ss;
    for (auto it = words_.rbegin(); it != words_.rend(); ++it)
        ss << std::bitset<64>(*it).to_string();
    auto s = ss.str();
    if (n_bits_ == 0) return s;
    return s.substr(s.size() - n_bits_);
}

BitsetVec::OccIter::OccIter() = default;

BitsetVec::OccIter::OccIter(const BitsetVec* bs, std::size_t start)
    : bs_(bs)
{
    if (!bs_ || start >= bs_->n_bits_) { done_ = true; return; }
    idx_ = start; w_ = idx_ / 64; b_ = idx_ % 64;
    advance_to_next();
}

void BitsetVec::OccIter::advance_to_next() {
    done_ = true;
    while (w_ < bs_->nwords()) {
        u64 word = bs_->words_[w_];
        if (b_) word &= (~u64{0} << b_);           // clear lower bits
        if (word) {
            unsigned tz = std::countr_zero(word);
            std::size_t bit = w_ * 64 + tz;
            if (bit < bs_->n_bits_) {
                idx_ = bit + 1;
                w_   = idx_ / 64;
                b_   = idx_ % 64;
                current0_ = static_cast<int>(bit);
                done_ = false;
                return;
            }
        }
        ++w_; b_ = 0;
    }
}

int BitsetVec::OccIter::operator*() const noexcept { return current0_; }

BitsetVec::OccIter& BitsetVec::OccIter::operator++() { advance_to_next(); return *this; }

bool BitsetVec::OccIter::operator!=(const OccIter& other) const noexcept {
    // end() is represented by done_==true
    return done_ != other.done_;
}

BitsetVec::OccIter BitsetVec::begin_occ() const { return OccIter(this, 0); }
BitsetVec::OccIter BitsetVec::end_occ()   const { return OccIter(); }

bool operator==(const BitsetVec& a, const BitsetVec& b) noexcept {
    return a.n_bits_ == b.n_bits_ && a.words_ == b.words_;
}
bool operator<(const BitsetVec& a, const BitsetVec& b) noexcept {
    if (a.n_bits_ != b.n_bits_) return a.n_bits_ < b.n_bits_;
    return std::lexicographical_compare(a.words_.begin(), a.words_.end(),
                                        b.words_.begin(), b.words_.end());
}

// =============================== Determinant =================================

Determinant::Determinant() = default;

Determinant::Determinant(std::size_t total_orbitals)
    : bits_(total_orbitals) {}

Determinant::Determinant(std::size_t total_orbitals, const std::vector<int>& occupied0)
    : bits_(total_orbitals)
{
    for (int k0 : occupied0) {
        if (k0 < 0 || static_cast<std::size_t>(k0) >= total_orbitals)
            throw std::out_of_range("Determinant: orbital index out of range (0-based).");
        bits_.set(static_cast<std::size_t>(k0));
    }
    bits_.mask_tail();
}

std::size_t Determinant::num_orbitals() const noexcept { return bits_.size(); }

bool Determinant::occupied(std::size_t orb0) const noexcept { return bits_.test(orb0); }
void Determinant::set     (std::size_t orb0) noexcept { bits_.set(orb0); }
void Determinant::reset   (std::size_t orb0) noexcept { bits_.reset(orb0); }

std::size_t Determinant::count_electrons() const noexcept { return bits_.popcount_all(); }

BitsetVec::OccIter Determinant::begin_occ() const { return bits_.begin_occ(); }
BitsetVec::OccIter Determinant::end_occ()   const { return bits_.end_occ();   }

std::string Determinant::to_string_binary() const { return bits_.to_binary(); }

bool operator==(const Determinant& a, const Determinant& b) noexcept { return a.bits_ == b.bits_; }
bool operator<( const Determinant& a, const Determinant& b) noexcept { return a.bits_ <  b.bits_; }

const BitsetVec& Determinant::bits() const noexcept { return bits_; }

// ============================== SpinDeterminant ==============================

SpinDeterminant::SpinDeterminant() = default;

SpinDeterminant::SpinDeterminant(std::size_t n_spatial)
    : bits_(n_spatial) {}

SpinDeterminant::SpinDeterminant(std::size_t n_spatial, const std::vector<int>& occupied0)
    : bits_(n_spatial)
{
    for (int i0 : occupied0) {
        if (i0 < 0 || static_cast<std::size_t>(i0) >= n_spatial)
            throw std::out_of_range("SpinDeterminant: spatial index out of range (0-based).");
        bits_.set(static_cast<std::size_t>(i0));
    }
    bits_.mask_tail();
}

std::size_t SpinDeterminant::num_orbitals() const noexcept { return bits_.size(); }
bool        SpinDeterminant::occupied(std::size_t i0) const noexcept { return bits_.test(i0); }
void        SpinDeterminant::set     (std::size_t i0) noexcept { bits_.set(i0); }
void        SpinDeterminant::reset   (std::size_t i0) noexcept { bits_.reset(i0); }
std::size_t SpinDeterminant::count_electrons() const noexcept { return bits_.popcount_all(); }

BitsetVec::OccIter SpinDeterminant::begin_occ() const { return bits_.begin_occ(); }
BitsetVec::OccIter SpinDeterminant::end_occ()   const { return bits_.end_occ();   }

const BitsetVec& SpinDeterminant::raw() const noexcept { return bits_; }

bool operator==(const SpinDeterminant& a, const SpinDeterminant& b) noexcept { return a.bits_ == b.bits_; }
bool operator<( const SpinDeterminant& a, const SpinDeterminant& b) noexcept { return a.bits_ <  b.bits_; }

// sign = (-1)^{N(<i)} with N counting occupied bits strictly below i.
std::optional<SpinDeterminant::OpResult>
SpinDeterminant::create(const SpinDeterminant& d, std::size_t i0) noexcept {
    if (i0 >= d.num_orbitals()) return std::nullopt;
    if (d.occupied(i0))         return std::nullopt; // Pauli
    SpinDeterminant out = d;
    const auto N_below = d.raw().popcount_below(i0);
    const int8_t sgn   = (N_below % 2 == 0) ? +1 : -1;
    out.set(i0);
    return OpResult{ std::move(out), sgn };
}

std::optional<SpinDeterminant::OpResult>
SpinDeterminant::annihilate(const SpinDeterminant& d, std::size_t i0) noexcept {
    if (i0 >= d.num_orbitals()) return std::nullopt;
    if (!d.occupied(i0))        return std::nullopt; // Pauli
    SpinDeterminant out = d;
    const auto N_below = d.raw().popcount_below(i0);
    const int8_t sgn   = (N_below % 2 == 0) ? +1 : -1;
    out.reset(i0);
    return OpResult{ std::move(out), sgn };
}

// ============================== SlaterDeterminant ============================

SlaterDeterminant::SlaterDeterminant() = default;

SlaterDeterminant::SlaterDeterminant(std::size_t n_spatial)
    : n_spatial_(n_spatial), alpha_(n_spatial), beta_(n_spatial) {}

SlaterDeterminant::SlaterDeterminant(std::size_t n_spatial,
                                     const std::vector<int>& occ_alpha0,
                                     const std::vector<int>& occ_beta0)
    : n_spatial_(n_spatial), alpha_(n_spatial, occ_alpha0), beta_(n_spatial, occ_beta0) {}

std::size_t            SlaterDeterminant::num_spatial_orbitals() const noexcept { return n_spatial_; }
const SpinDeterminant& SlaterDeterminant::alpha() const noexcept { return alpha_; }
const SpinDeterminant& SlaterDeterminant::beta () const noexcept { return beta_;  }
SpinDeterminant& SlaterDeterminant::alpha() noexcept { return alpha_; }
SpinDeterminant& SlaterDeterminant::beta()  noexcept { return beta_;  }

std::size_t SlaterDeterminant::count_electrons() const noexcept {
    return alpha_.count_electrons() + beta_.count_electrons();
}
double SlaterDeterminant::Sz() const noexcept {
    return 0.5 * (static_cast<double>(alpha_.count_electrons()) -
                  static_cast<double>(beta_.count_electrons()));
}

bool operator==(const SlaterDeterminant& a, const SlaterDeterminant& b) noexcept {
    return a.n_spatial_ == b.n_spatial_ && a.alpha_ == b.alpha_ && a.beta_ == b.beta_;
}
bool operator<(const SlaterDeterminant& a, const SlaterDeterminant& b) noexcept {
    if (a.n_spatial_ != b.n_spatial_) return a.n_spatial_ < b.n_spatial_;
    if (a.alpha_     != b.alpha_)     return a.alpha_     < b.alpha_;
    return a.beta_ < b.beta_;
}

static inline int8_t apply_beta_order_sign(int8_t sgn, const SlaterDeterminant& s,
                                           std::size_t i0, SpinOrbitalOrder order)
{
    if (order == SpinOrbitalOrder::AlphaFirst) {
        // β sector is placed after all α orbitals; crossing Nα electrons.
        return (s.alpha().count_electrons() % 2) ? int8_t(-sgn) : sgn;
    } else {
        // Interleaved: (α0,β0,α1,β1,...). For β_i, you cross α_j for j<=i.
        const auto Nalpha_le_i = s.alpha().raw().popcount_below(i0 + 1);
        return (Nalpha_le_i % 2) ? int8_t(-sgn) : sgn;
    }
}

std::optional<SlaterDeterminant::OpResult>
SlaterDeterminant::create(const SlaterDeterminant& s, std::size_t i0, Spin spin,
                          SpinOrbitalOrder order) noexcept
{
    if (i0 >= s.n_spatial_) return std::nullopt;

    if (spin == Spin::Alpha) {
        auto r = SpinDeterminant::create(s.alpha_, i0);
        if (!r) return std::nullopt;
        SlaterDeterminant out = s;
        out.alpha_ = std::move(r->det);
        return OpResult{ std::move(out), r->sign };
    } else {
        auto r = SpinDeterminant::create(s.beta_, i0);
        if (!r) return std::nullopt;
        int8_t sign = apply_beta_order_sign(r->sign, s, i0, order);
        SlaterDeterminant out = s;
        out.beta_ = std::move(r->det);
        return OpResult{ std::move(out), sign };
    }
}

std::optional<SlaterDeterminant::OpResult>
SlaterDeterminant::annihilate(const SlaterDeterminant& s, std::size_t i0, Spin spin,
                              SpinOrbitalOrder order) noexcept
{
    if (i0 >= s.n_spatial_) return std::nullopt;

    if (spin == Spin::Alpha) {
        auto r = SpinDeterminant::annihilate(s.alpha_, i0);
        if (!r) return std::nullopt;
        SlaterDeterminant out = s;
        out.alpha_ = std::move(r->det);
        return OpResult{ std::move(out), r->sign };
    } else {
        auto r = SpinDeterminant::annihilate(s.beta_, i0);
        if (!r) return std::nullopt;
        int8_t sign = apply_beta_order_sign(r->sign, s, i0, order);
        SlaterDeterminant out = s;
        out.beta_ = std::move(r->det);
        return OpResult{ std::move(out), sign };
    }
}

// ========================== Interleave / Deinterleave ========================

Determinant interleave(const SlaterDeterminant& s, SpinOrbitalOrder order)
{
    const auto M = s.num_spatial_orbitals();
    Determinant d(2 * M);

    if (order == SpinOrbitalOrder::AlphaFirst) {
        // α at [0..M-1], β at [M..2M-1]
        for (auto it = s.alpha().begin_occ(); it != s.alpha().end_occ(); ++it) {
            const std::size_t i0 = static_cast<std::size_t>(*it);
            d.set(i0);
        }
        for (auto it = s.beta().begin_occ(); it != s.beta().end_occ(); ++it) {
            const std::size_t i0 = static_cast<std::size_t>(*it);
            d.set(M + i0);
        }
    } else {
        // Interleaved: α_i -> 2*i, β_i -> 2*i+1
        for (auto it = s.alpha().begin_occ(); it != s.alpha().end_occ(); ++it) {
            const std::size_t i0 = static_cast<std::size_t>(*it);
            d.set(2 * i0);
        }
        for (auto it = s.beta().begin_occ(); it != s.beta().end_occ(); ++it) {
            const std::size_t i0 = static_cast<std::size_t>(*it);
            d.set(2 * i0 + 1);
        }
    }
    return d;
}

SlaterDeterminant deinterleave(const Determinant& d, SpinOrbitalOrder order)
{
    const auto K = d.num_orbitals();
    if (K % 2 != 0) throw std::invalid_argument("deinterleave: number of spin-orbitals must be even.");
    const auto M = K / 2;

    std::vector<int> occ_a, occ_b;
    occ_a.reserve(M); occ_b.reserve(M);

    if (order == SpinOrbitalOrder::AlphaFirst) {
        for (auto it = d.begin_occ(); it != d.end_occ(); ++it) {
            const std::size_t p0 = static_cast<std::size_t>(*it);
            if (p0 < M) occ_a.push_back(static_cast<int>(p0));
            else        occ_b.push_back(static_cast<int>(p0 - M));
        }
    } else {
        for (auto it = d.begin_occ(); it != d.end_occ(); ++it) {
            const std::size_t p0 = static_cast<std::size_t>(*it);
            const std::size_t i0 = p0 / 2;
            if ((p0 % 2) == 0) occ_a.push_back(static_cast<int>(i0));
            else               occ_b.push_back(static_cast<int>(i0));
        }
    }

    return SlaterDeterminant(M, occ_a, occ_b);
}

} // namespace ci

// ============================== Hashing ======================================
namespace {
inline std::size_t hash_mix_u64(std::size_t seed, ci::u64 v) noexcept {
    v ^= v >> 30; v *= 0xbf58476d1ce4e5b9ULL;
    v ^= v >> 27; v *= 0x94d049bb133111ebULL;
    v ^= v >> 31;
    return seed ^ (static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ULL + (seed<<6) + (seed>>2));
}
} // anonymous namespace

std::size_t std::hash<ci::BitsetVec>::operator()(const ci::BitsetVec& b) const noexcept {
    std::size_t seed = std::hash<std::size_t>{}(b.size());
    for (auto w : b.words()) seed = hash_mix_u64(seed, w);
    return seed;
}
std::size_t std::hash<ci::Determinant>::operator()(const ci::Determinant& d) const noexcept {
    std::size_t seed = d.num_orbitals();
    for (auto w : d.bits().words()) seed = hash_mix_u64(seed, w);
    return seed;
}
std::size_t std::hash<ci::SpinDeterminant>::operator()(const ci::SpinDeterminant& sd) const noexcept {
    return std::hash<ci::BitsetVec>{}(sd.raw());
}
std::size_t std::hash<ci::SlaterDeterminant>::operator()(const ci::SlaterDeterminant& s) const noexcept {
    std::size_t h_alpha = std::hash<ci::SpinDeterminant>{}(s.alpha());
    std::size_t h_beta  = std::hash<ci::SpinDeterminant>{}(s.beta());
    return h_alpha ^ (h_beta + 0x9e3779b97f4a7c15ULL + (h_alpha << 6) + (h_alpha >> 2));
}