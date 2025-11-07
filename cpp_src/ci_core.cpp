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

// ======================= SlaterDeterminant Fast Operators =======================
// This section implements suggestion (B)

// Helper to calculate sign for a single creation or annihilation on a SpinDeterminant
// Returns false if Pauli-blocked.
inline bool op_sign(const SpinDeterminant& d, size_t i0, bool create, int8_t& sign) {
    if (create) {
        if (d.occupied(i0)) return false; // Pauli block
        const auto N_below = d.raw().popcount_below(i0);
        sign = (N_below % 2 == 0) ? +1 : -1;
    } else {
        if (!d.occupied(i0)) return false; // Pauli block
        const auto N_below = d.raw().popcount_below(i0);
        sign = (N_below % 2 == 0) ? +1 : -1;
    }
    return true;
}

// Same-spin excitation: sign is (-1)^C where C is the number of electrons
// strictly between i and j.
inline int8_t same_spin_excitation_sign(const SpinDeterminant& d, size_t i0, size_t j0) {
    size_t p_min = std::min(i0, j0);
    size_t p_max = std::max(i0, j0);
    // popcount_below(p_max) counts set bits in [0, p_max-1]
    // popcount_below(p_min + 1) counts set bits in [0, p_min]
    // The difference is the number of set bits in [p_min+1, p_max-1]
    const auto crossings = d.raw().popcount_below(p_max) - d.raw().popcount_below(p_min + 1);
    return (crossings % 2 == 0) ? +1 : -1;
}

bool SlaterDeterminant::apply_excitation_single_fast(
    const SlaterDeterminant& s, size_t i0, size_t j0,
    Spin spin_i, Spin spin_j, SlaterDeterminant& out, int8_t& sign,
    SpinOrbitalOrder order) noexcept
{
    if (i0 >= s.n_spatial_ || j0 >= s.n_spatial_) return false;

    // Case 1: Same spin sectors (e.g., c_iα† c_jα)
    if (spin_i == spin_j) {
        if (i0 == j0) { // Number operator: c_i† c_i
            const auto& d = (spin_i == Spin::Alpha) ? s.alpha_ : s.beta_;
            if (!d.occupied(i0)) return false;
            out = s;
            sign = 1;
            return true;
        }
        const auto& d_in = (spin_i == Spin::Alpha) ? s.alpha_ : s.beta_;
        if (!d_in.occupied(j0) || d_in.occupied(i0)) return false; // Pauli
        
        sign = same_spin_excitation_sign(d_in, i0, j0);
        out = s;
        auto& d_out = (spin_i == Spin::Alpha) ? out.alpha_ : out.beta_;
        d_out.reset(j0);
        d_out.set(i0);
        return true;
    }
    // Case 2: Different spin sectors (e.g., c_iα† c_jβ)
    else {
        auto r_ann = annihilate(s, j0, spin_j, order);
        if (!r_ann) return false;
        auto r_cre = create(r_ann->det, i0, spin_i, order);
        if (!r_cre) return false;

        out = std::move(r_cre->det);
        sign = r_cre->sign * r_ann->sign;
        return true;
    }
}

bool SlaterDeterminant::apply_excitation_double_fast(
    const SlaterDeterminant& s, size_t i0, size_t j0, size_t k0, size_t l0,
    Spin spin_i, Spin spin_j, Spin spin_k, Spin spin_l,
    SlaterDeterminant& out, int8_t& sign,
    SpinOrbitalOrder order) noexcept
{
    // Apply operators right to left: k, l, j, i
    // This implementation re-uses the std::optional path for simplicity.
    // A fully optimized version would pass temporary SpinDeterminants by reference.
    auto r1 = annihilate(s, k0, spin_k, order);
    if (!r1) return false;

    auto r2 = annihilate(r1->det, l0, spin_l, order);
    if (!r2) return false;

    auto r3 = create(r2->det, j0, spin_j, order);
    if (!r3) return false;

    auto r4 = create(r3->det, i0, spin_i, order);
    if (!r4) return false;

    out = std::move(r4->det);
    sign = r4->sign * r3->sign * r2->sign * r1->sign;
    return true;
}

// ================================ Wavefunction ===============================

Wavefunction::Wavefunction(std::size_t n_spatial) : n_spatial_(n_spatial) {}
Wavefunction::Wavefunction(std::size_t n_spatial, const Data& init)
    : n_spatial_(n_spatial), data_(init) {}

std::size_t Wavefunction::num_spatial_orbitals() const noexcept { return n_spatial_; }
const Wavefunction::Data& Wavefunction::data() const noexcept { return data_; }

void Wavefunction::add_term(const SlaterDeterminant& det, Coeff c, double tol) {
    if (std::abs(c) == 0.0) return;
    auto& v = data_[det];
    v += c;
    if (std::abs(v) < tol) data_.erase(det);
}

Wavefunction::Wavefunction(std::size_t n_spatial, 
    const std::vector<SlaterDeterminant>& basis, 
    const std::vector<Coeff>& coeffs,
    bool keep_zeros) // New parameter
    : n_spatial_(n_spatial)
{
    if (basis.size() != coeffs.size()) {
        throw std::invalid_argument("Basis and coefficients vectors must have the same size.");
    }
    data_.reserve(basis.size());

    if (keep_zeros) {
        // Direct insertion: Keep all terms, including zeros.
        for (size_t i = 0; i < basis.size(); ++i) {
            data_[basis[i]] = coeffs[i];
        }
    } else {
        // Default behavior: Prune zero-coefficient terms on creation.
        for (size_t i = 0; i < basis.size(); ++i) {
            add_term(basis[i], coeffs[i], 0);
        }
    }
}

void Wavefunction::normalize(double tol) {
    long double norm2 = 0.0L;
    for (const auto& kv : data_) norm2 += std::norm(kv.second);
    if (norm2 == 0.0L) return;
    const double norm = std::sqrt(static_cast<double>(norm2));
    for (auto& kv : data_) kv.second /= norm;
    prune(tol);
}

void Wavefunction::prune(double threshold) {
    const double t2 = threshold * threshold;
    for (auto it = data_.begin(); it != data_.end(); ) {
        if (std::norm(it->second) < t2) it = data_.erase(it);
        else ++it;
    }
}

std::vector<SlaterDeterminant> Wavefunction::basis_sorted() const {
    std::vector<SlaterDeterminant> b; b.reserve(data_.size());
    for (const auto& kv : data_) b.push_back(kv.first);
    std::sort(b.begin(), b.end());
    return b;
}

std::vector<Wavefunction::Coeff>
Wavefunction::coeffs_sorted(const std::vector<SlaterDeterminant>& basis) const {
    std::vector<Coeff> c; c.reserve(basis.size());
    for (const auto& d : basis) c.push_back(data_.at(d));
    return c;
}

Wavefunction apply_creation(const Wavefunction& wf, std::size_t i0, Spin spin,
                            SpinOrbitalOrder order)
{
    Wavefunction out(wf.num_spatial_orbitals());
    for (const auto& [det, coeff] : wf.data()) {
        auto r = SlaterDeterminant::create(det, i0, spin, order);
        if (!r) continue;
        out.add_term(r->det, coeff * static_cast<double>(r->sign));
    }
    return out;
}

Wavefunction apply_annihilation(const Wavefunction& wf, std::size_t i0, Spin spin,
                                SpinOrbitalOrder order)
{
    Wavefunction out(wf.num_spatial_orbitals());
    for (const auto& [det, coeff] : wf.data()) {
        auto r = SlaterDeterminant::annihilate(det, i0, spin, order);
        if (!r) continue;
        out.add_term(r->det, coeff * static_cast<double>(r->sign));
    }
    return out;
}


// Apply c†_i c_j
std::optional<SlaterDeterminant::OpResult>
SlaterDeterminant::apply_excitation_single(const SlaterDeterminant& s, size_t i0, size_t j0,
                                           Spin spin_i, Spin spin_j,
                                           SpinOrbitalOrder order) noexcept
{
    // Annihilate j first
    auto r_ann = annihilate(s, j0, spin_j, order);
    if (!r_ann) return std::nullopt;

    // Create i on the intermediate determinant
    auto r_cre = create(r_ann->det, i0, spin_i, order);
    if (!r_cre) return std::nullopt;
    
    // Combine signs
    r_cre->sign *= r_ann->sign;
    return r_cre;
}

// Apply c†_i c†_j c_l c_k
std::optional<SlaterDeterminant::OpResult>
SlaterDeterminant::apply_excitation_double(const SlaterDeterminant& s,
                                           size_t i0, size_t j0, size_t k0, size_t l0,
                                           Spin spin_i, Spin spin_j, Spin spin_k, Spin spin_l,
                                           SpinOrbitalOrder order) noexcept
{
    // Apply operators from right to left: c_k, c_l, c†_j, c†_i
    auto r1 = annihilate(s, k0, spin_k, order);
    if (!r1) return std::nullopt;

    auto r2 = annihilate(r1->det, l0, spin_l, order);
    if (!r2) return std::nullopt;

    auto r3 = create(r2->det, j0, spin_j, order);
    if (!r3) return std::nullopt;

    auto r4 = create(r3->det, i0, spin_i, order);
    if (!r4) return std::nullopt;

    // Combine all signs
    r4->sign = r4->sign * r3->sign * r2->sign * r1->sign;
    return r4;
}

Wavefunction::Coeff Wavefunction::dot(const Wavefunction& other) const {
    if (n_spatial_ != other.n_spatial_) {
        throw std::invalid_argument("Wavefunctions must have the same number of spatial orbitals for dot product.");
    }

    Coeff result = {0.0, 0.0};
    // Iterate over the smaller of the two maps for efficiency
    const auto& map_a = (data_.size() < other.data_.size()) ? data_ : other.data_;
    const auto& map_b = (data_.size() < other.data_.size()) ? other.data_ : data_;

    for (const auto& [det, coeff_a] : map_a) {
        auto it = map_b.find(det);
        if (it != map_b.end()) {
            const auto& coeff_b = it->second;
            result += std::conj(coeff_a) * coeff_b;
        }
    }
    return result;
}

void Wavefunction::add_wavefunction(const Wavefunction& other, Coeff scale) {
    if (n_spatial_ != other.n_spatial_) {
        throw std::invalid_argument("Wavefunctions must have the same number of spatial orbitals for addition.");
    }
    for (const auto& [det, coeff] : other.data_) {
        add_term(det, coeff * scale);
    }
}


// ====================== Wavefunction Operator Application =======================

// --- Helper for bucketing ---
// Key for one-body term: packs (j, spin_j)
inline uint64_t make_key1(size_t j, Spin spin_j) {
    return (static_cast<uint64_t>(j) << 1) | static_cast<uint64_t>(spin_j);
}

// Key for two-body term: packs (k, l, spin_k, spin_l)
inline uint64_t make_key2(size_t k, size_t l, Spin spin_k, Spin spin_l) {
    // Ensure canonical order (k, sk) <= (l, sl) to halve bucket search space for some cases
    size_t p1 = k; Spin s1 = spin_k;
    size_t p2 = l; Spin s2 = spin_l;
    if (p1 > p2 || (p1 == p2 && static_cast<uint8_t>(s1) > static_cast<uint8_t>(s2))) {
        std::swap(p1, p2);
        std::swap(s1, s2);
    }
    return (static_cast<uint64_t>(p1) << 32) |
           (static_cast<uint64_t>(p2) << 2)  |
           (static_cast<uint64_t>(s1) << 1)  |
           static_cast<uint64_t>(s2);
}


Wavefunction apply_one_body_operator_(const Wavefunction& wf,
                                     const std::vector<OneBodyTerm>& terms)
{
    Wavefunction out(wf.num_spatial_orbitals());
    if (wf.data().empty()) return out;

    // 1. Build buckets of terms based on annihilated orbital (j, spin_j)
    std::unordered_map<uint64_t, std::vector<const OneBodyTerm*>> buckets;
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) { // Simple screening
            buckets[make_key1(term.j, term.spin_j)].push_back(&term);
        }
    }

    // 2. Iterate through determinants in the wavefunction
    SlaterDeterminant excited_det(wf.num_spatial_orbitals());
    int8_t sign;

    for (const auto& [det, coeff] : wf.data()) {
        // 3. For each determinant, check its occupied alpha orbitals
        for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
            const size_t j = static_cast<size_t>(*it);
            auto bucket_it = buckets.find(make_key1(j, Spin::Alpha));
            if (bucket_it != buckets.end()) {
                // 4. If a bucket exists, apply all excitations from it
                for (const auto* term : bucket_it->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, term->i, term->j, term->spin_i, term->spin_j, excited_det, sign)) {
                        out.add_term(excited_det, coeff * term->value * static_cast<double>(sign));
                    }
                }
            }
        }
        // 5. Repeat for beta orbitals
        for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
            const size_t j = static_cast<size_t>(*it);
            auto bucket_it = buckets.find(make_key1(j, Spin::Beta));
            if (bucket_it != buckets.end()) {
                for (const auto* term : bucket_it->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, term->i, term->j, term->spin_i, term->spin_j, excited_det, sign)) {
                        out.add_term(excited_det, coeff * term->value * static_cast<double>(sign));
                    }
                }
            }
        }
    }
    return out;
}

Wavefunction apply_two_body_operator_(const Wavefunction& wf,
                                     const std::vector<TwoBodyTerm>& terms)
{
    Wavefunction out(wf.num_spatial_orbitals());
    if (wf.data().empty()) return out;

    // 1. Build buckets for two-body terms
    std::unordered_map<uint64_t, std::vector<const TwoBodyTerm*>> buckets;
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key2(term.k, term.l, term.spin_k, term.spin_l)].push_back(&term);
        }
    }

    // 2. Iterate through determinants
    SlaterDeterminant excited_det(wf.num_spatial_orbitals());
    int8_t sign;
    std::vector<std::pair<size_t, Spin>> occupied_orbs;

    for (const auto& [det, coeff] : wf.data()) {
        // Collect all occupied spin-orbitals for easier iteration over pairs
        occupied_orbs.clear();
        for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Alpha);
        for (auto it = det.beta().begin_occ();  it != det.beta().end_occ();  ++it) occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Beta);

        // 3. Iterate over all PAIRS of occupied orbitals (k, sk) and (l, sl)
        for (size_t idx1 = 0; idx1 < occupied_orbs.size(); ++idx1) {
            for (size_t idx2 = idx1 + 1; idx2 < occupied_orbs.size(); ++idx2) {
                const auto& [k, sk] = occupied_orbs[idx1];
                const auto& [l, sl] = occupied_orbs[idx2];
                
                auto bucket_it = buckets.find(make_key2(k, l, sk, sl));
                if (bucket_it != buckets.end()) {
                    for (const auto* term : bucket_it->second) {
                        // The key is canonical, but the term may not be. Check both permutations.
                        if ((term->k == k && term->l == l && term->spin_k == sk && term->spin_l == sl) ||
                            (term->k == l && term->l == k && term->spin_k == sl && term->spin_l == sk)) {
                            if (SlaterDeterminant::apply_excitation_double_fast(det, term->i, term->j, term->k, term->l, term->spin_i, term->spin_j, term->spin_k, term->spin_l, excited_det, sign)) {
                                out.add_term(excited_det, coeff * term->value * static_cast<double>(sign));
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}


// Hash & Eq already defined for SlaterDeterminant in your code
using Coeff = Wavefunction::Coeff;
using Map   = std::unordered_map<SlaterDeterminant, Coeff>;

static inline std::size_t shard_of(const SlaterDeterminant& d, std::size_t S) {
    std::size_t h = std::hash<SlaterDeterminant>{}(d);
    return h & (S - 1); // S must be power of two
}

Wavefunction apply_one_body_operator(const Wavefunction& wf,
                                     const std::vector<OneBodyTerm>& terms)
{
    Wavefunction out(wf.num_spatial_orbitals());
    if (wf.data().empty()) return out;
    const std::size_t M = wf.num_spatial_orbitals();

    // Buckets (read-only)
    std::unordered_map<uint64_t, std::vector<const OneBodyTerm*>> buckets;
    buckets.reserve(terms.size());
    for (const auto& t : terms) if (std::abs(t.value) > 0.0)
        buckets[make_key1(t.j, t.spin_j)].push_back(&t);

    // Flatten wf for parallel iteration
    std::vector<std::pair<SlaterDeterminant, Coeff>> items;
    items.reserve(wf.data().size());
    for (const auto& kv : wf.data()) items.emplace_back(kv.first, kv.second);

    // Local per-thread maps (sums duplicates early)
    int T = 1;
#ifdef _OPENMP
    T = std::max(1, omp_get_max_threads());
#endif
    std::vector<Map> local_maps(T);

    // Generate + accumulate locally
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& acc = local_maps[tid];
        acc.reserve(std::max<std::size_t>(512, items.size()/T));

        SlaterDeterminant excited(M);
        int8_t sign;

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t idx = 0; idx < (std::ptrdiff_t)items.size(); ++idx) {
            const auto& det   = items[(std::size_t)idx].first;
            const auto& coeff = items[(std::size_t)idx].second;

            // α
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
                const size_t j = (size_t)*it;
                auto bi = buckets.find(make_key1(j, Spin::Alpha));
                if (bi == buckets.end()) continue;
                for (const auto* t : bi->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, t->i, t->j, t->spin_i, t->spin_j,
                                                                        excited, sign)) {
                        acc[excited] += coeff * t->value * double(sign);
                    }
                }
            }
            // β
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
                const size_t j = (size_t)*it;
                auto bi = buckets.find(make_key1(j, Spin::Beta));
                if (bi == buckets.end()) continue;
                for (const auto* t : bi->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, t->i, t->j, t->spin_i, t->spin_j,
                                                                        excited, sign)) {
                        acc[excited] += coeff * t->value * double(sign);
                    }
                }
            }
        }
    }

    // Sharded global reduce to avoid a single hot hash map
    // Choose power-of-two shard count (e.g. next power of two >= 2*T)
    std::size_t S = 1;
    while (S < (std::size_t)(2 * T)) S <<= 1;
    std::vector<Map> shards(S);

#ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
#endif
    for (int t = 0; t < T; ++t) {
        for (const auto& kv : local_maps[t]) {
            const auto& det = kv.first;
            auto s = shard_of(det, S);
            // To avoid locks, move per-thread into a thread-local buffer first,
            // then we do a second parallel pass to combine shard-by-shard:
#ifdef _OPENMP
            #pragma omp critical  // cheap: S is large; contention low
#endif
            shards[s][det] += kv.second;
        }
    }

    // Finalize Wavefunction
    for (auto& sh : shards)
        for (auto& kv : sh)
            out.add_term(kv.first, kv.second);

    return out;
}

// In the .cpp that defines this, make sure you have:
//   #include <omp.h>
// and you link OpenMP in CMake (you already do).

Wavefunction apply_two_body_operator(const Wavefunction& wf,
                                     const std::vector<TwoBodyTerm>& terms)
{
    using Coeff = Wavefunction::Coeff;

    Wavefunction out(wf.num_spatial_orbitals());
    if (wf.data().empty() || terms.empty()) return out;
    const std::size_t M = wf.num_spatial_orbitals();

    // 1) Read-only buckets keyed by canonical annihilated pair (k,sk),(l,sl)
    std::unordered_map<uint64_t, std::vector<const TwoBodyTerm*>> buckets;
    buckets.reserve(terms.size());
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key2(term.k, term.l, term.spin_k, term.spin_l)].push_back(&term);
        }
    }

    // 2) Flatten wf for parallel iteration
    std::vector<std::pair<SlaterDeterminant, Coeff>> items;
    items.reserve(wf.data().size());
    for (const auto& kv : wf.data()) items.emplace_back(kv.first, kv.second);

    // 3) Per-thread accumulation maps: SlaterDeterminant -> summed Coeff
    using Map = std::unordered_map<SlaterDeterminant, Coeff>;
    int T = 1;
#ifdef _OPENMP
    T = std::max(1, omp_get_max_threads());
#endif
    std::vector<Map> local_maps((std::size_t)T);

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& acc = local_maps[(std::size_t)tid];
        acc.reserve(std::max<std::size_t>(512, items.size() / std::max(1, T)));

        SlaterDeterminant excited_det(M);
        int8_t sign;

        std::vector<std::pair<size_t, Spin>> occ;
        occ.reserve(2 * M);

#ifdef _OPENMP
        #pragma omp for schedule(guided)
#endif
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(items.size()); ++idx) {
            const auto& det   = items[(std::size_t)idx].first;
            const auto& coeff = items[(std::size_t)idx].second;

            // Collect occupied spin-orbitals
            occ.clear();
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it)
                occ.emplace_back(static_cast<size_t>(*it), Spin::Alpha);
            for (auto it = det.beta().begin_occ();  it != det.beta().end_occ();  ++it)
                occ.emplace_back(static_cast<size_t>(*it), Spin::Beta);

            const std::size_t L = occ.size();
            for (std::size_t a = 0; a + 1 < L; ++a) {
                const auto [k, sk] = occ[a];
                for (std::size_t b = a + 1; b < L; ++b) {
                    const auto [l, sl] = occ[b];

                    auto itB = buckets.find(make_key2(k, l, sk, sl));
                    if (itB == buckets.end()) continue;

                    // Apply all matching two-body terms from the bucket
                    for (const auto* term : itB->second) {
                        const bool match_kl = (term->k == k && term->l == l &&
                                               term->spin_k == sk && term->spin_l == sl);
                        const bool match_lk = (term->k == l && term->l == k &&
                                               term->spin_k == sl && term->spin_l == sk);
                        if (!(match_kl || match_lk)) continue;

                        if (SlaterDeterminant::apply_excitation_double_fast(
                                det, term->i, term->j, term->k, term->l,
                                term->spin_i, term->spin_j, term->spin_k, term->spin_l,
                                excited_det, sign))
                        {
                            acc[excited_det] += coeff * term->value * static_cast<double>(sign);
                        }
                    }
                }
            }
        } // omp for
    } // omp parallel

    // 4) Lock-free sharded merge of the local maps into S shards
    auto shard_of = [](const SlaterDeterminant& d, std::size_t S)->std::size_t {
        std::size_t h = std::hash<SlaterDeterminant>{}(d);
        return h & (S - 1); // S must be a power of two
    };

    std::size_t S = 1;
    while (S < (std::size_t)(2 * T)) S <<= 1; // at least 2*T shards
    std::vector<Map> shards(S);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < (int)S; ++s) {
        auto& dst = shards[(std::size_t)s];
        // Rough capacity hint: distribute keys evenly
        std::size_t hint = 0;
        for (const auto& lm : local_maps) hint += lm.size();
        hint = hint / S + 8;
        dst.reserve(hint);

        for (const auto& lm : local_maps) {
            for (const auto& kv : lm) {
                if ((int)shard_of(kv.first, S) == s) {
                    dst[kv.first] += kv.second;
                }
            }
        }
    }

    // 5) Emit to output wavefunction
    for (auto& sh : shards)
        for (auto& kv : sh)
            out.add_term(kv.first, kv.second);

    return out;
}

// ============================ Connectivity Generation ===========================

std::vector<SlaterDeterminant>
get_connections_one_body_(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<OneBodyTerm>& terms)
{
    std::unordered_set<SlaterDeterminant> unique_dets;
    if (basis.empty()) return {};

    const size_t n_spatial = basis[0].num_spatial_orbitals();
    std::unordered_map<uint64_t, std::vector<const OneBodyTerm*>> buckets;
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key1(term.j, term.spin_j)].push_back(&term);
        }
    }

    SlaterDeterminant excited_det(n_spatial);
    int8_t sign;

    for (const auto& det : basis) {
        // Alpha
        for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
            const size_t j = static_cast<size_t>(*it);
            auto bucket_it = buckets.find(make_key1(j, Spin::Alpha));
            if (bucket_it != buckets.end()) {
                for (const auto* term : bucket_it->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, term->i, term->j, term->spin_i, term->spin_j, excited_det, sign)) {
                        unique_dets.insert(excited_det);
                    }
                }
            }
        }
        // Beta
        for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
            const size_t j = static_cast<size_t>(*it);
            auto bucket_it = buckets.find(make_key1(j, Spin::Beta));
            if (bucket_it != buckets.end()) {
                for (const auto* term : bucket_it->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, term->i, term->j, term->spin_i, term->spin_j, excited_det, sign)) {
                        unique_dets.insert(excited_det);
                    }
                }
            }
        }
    }

    std::vector<SlaterDeterminant> connected_dets(unique_dets.begin(), unique_dets.end());
    std::sort(connected_dets.begin(), connected_dets.end());
    return connected_dets;
}
                         
std::vector<SlaterDeterminant>
get_connections_two_body_(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<TwoBodyTerm>& terms)
{
    // 1. Handle edge cases
    if (basis.empty() || terms.empty()) {
        return {};
    }

    // 2. Use a hash set for efficient collection of unique determinants
    std::unordered_set<SlaterDeterminant> unique_dets;
    const size_t n_spatial = basis[0].num_spatial_orbitals();

    // 3. Build buckets of terms based on the CANONICAL annihilated pair (k, sk), (l, sl)
    //    This drastically prunes the search space.
    std::unordered_map<uint64_t, std::vector<const TwoBodyTerm*>> buckets;
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key2(term.k, term.l, term.spin_k, term.spin_l)].push_back(&term);
        }
    }

    // 4. Pre-allocate objects to avoid re-allocation inside the tight loops
    SlaterDeterminant excited_det(n_spatial);
    int8_t sign;
    std::vector<std::pair<size_t, Spin>> occupied_orbs;
    occupied_orbs.reserve(n_spatial * 2);

    // 5. Iterate through each determinant in the starting basis
    for (const auto& det : basis) {
        // Create a flat list of occupied spin-orbitals for easy iteration over pairs
        occupied_orbs.clear();
        for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
            occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Alpha);
        }
        for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
            occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Beta);
        }

        // 6. Iterate over all unique PAIRS of occupied orbitals { (k, sk), (l, sl) }
        for (size_t idx1 = 0; idx1 < occupied_orbs.size(); ++idx1) {
            for (size_t idx2 = idx1 + 1; idx2 < occupied_orbs.size(); ++idx2) {
                const auto& [k, sk] = occupied_orbs[idx1];
                const auto& [l, sl] = occupied_orbs[idx2];

                // 7. Find the corresponding bucket for this annihilated pair
                auto bucket_it = buckets.find(make_key2(k, l, sk, sl));
                if (bucket_it != buckets.end()) {
                    
                    // 8. Apply all operator terms from the bucket
                    for (const auto* term : bucket_it->second) {
                        // The bucket key is canonical, but the term itself may be stored as v_ijkl or v_ijlk.
                        // We must ensure the term we apply matches the orbitals we're annihilating.
                        bool term_matches_kl = (term->k == k && term->l == l && term->spin_k == sk && term->spin_l == sl);
                        bool term_matches_lk = (term->k == l && term->l == k && term->spin_k == sl && term->spin_l == sk);
                        
                        if (term_matches_kl || term_matches_lk) {
                            // Apply the excitation operator c†_i c†_j c_l c_k
                            if (SlaterDeterminant::apply_excitation_double_fast(det,
                                                             term->i, term->j, term->k, term->l,
                                                             term->spin_i, term->spin_j, term->spin_k, term->spin_l,
                                                             excited_det, sign))
                            {
                                // If successful, add the new determinant to our set
                                unique_dets.insert(excited_det);
                            }
                        }
                    }
                }
            }
        }
    }

    // 9. Convert the set to a sorted vector and return
    std::vector<SlaterDeterminant> connected_dets(unique_dets.begin(), unique_dets.end());
    std::sort(connected_dets.begin(), connected_dets.end());
    return connected_dets;
}

std::vector<SlaterDeterminant>
get_connections_one_body(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<OneBodyTerm>& terms)
{
    if (basis.empty()) return {};

    const size_t n_spatial = basis[0].num_spatial_orbitals();

    // Build read-only buckets once
    std::unordered_map<uint64_t, std::vector<const OneBodyTerm*>> buckets;
    buckets.reserve(terms.size());
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key1(term.j, term.spin_j)].push_back(&term);
        }
    }

    // Thread-local accumulation of unique dets; gathered after parallel region
    std::vector<std::unordered_set<SlaterDeterminant>> partials;
    partials.reserve(omp_get_max_threads());

    #pragma omp parallel
    {
        std::unordered_set<SlaterDeterminant> local;
        // Rough reserve to reduce rehashing
        local.reserve(std::max<size_t>(16, basis.size() / std::max(1, omp_get_num_threads())));

        SlaterDeterminant excited_det(n_spatial);
        int8_t sign;

        #pragma omp for schedule(guided)
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(basis.size()); ++idx) {
            const auto& det = basis[static_cast<size_t>(idx)];

            // Alpha occupied orbitals
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
                const size_t j = static_cast<size_t>(*it);
                auto key = make_key1(j, Spin::Alpha);
                auto b = buckets.find(key);
                if (b != buckets.end()) {
                    for (const auto* term : b->second) {
                        if (SlaterDeterminant::apply_excitation_single_fast(
                                det, term->i, term->j, term->spin_i, term->spin_j,
                                excited_det, sign))
                        {
                            local.insert(excited_det);
                        }
                    }
                }
            }
            // Beta occupied orbitals
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
                const size_t j = static_cast<size_t>(*it);
                auto key = make_key1(j, Spin::Beta);
                auto b = buckets.find(key);
                if (b != buckets.end()) {
                    for (const auto* term : b->second) {
                        if (SlaterDeterminant::apply_excitation_single_fast(
                                det, term->i, term->j, term->spin_i, term->spin_j,
                                excited_det, sign))
                        {
                            local.insert(excited_det);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            partials.push_back(std::move(local));
        }
    }

    // Merge thread-local sets
    std::unordered_set<SlaterDeterminant> unique_dets;
    size_t hint = 0; for (auto& s : partials) hint += s.size();
    unique_dets.reserve(hint);
    for (auto& s : partials) {
        unique_dets.insert(s.begin(), s.end());
    }

    std::vector<SlaterDeterminant> connected_dets(unique_dets.begin(), unique_dets.end());
    std::sort(connected_dets.begin(), connected_dets.end());
    return connected_dets;
}

// Requires: #include <omp.h>

std::vector<SlaterDeterminant>
get_connections_two_body(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<TwoBodyTerm>& terms)
{
    if (basis.empty() || terms.empty()) return {};

    const size_t n_spatial = basis[0].num_spatial_orbitals();

    // Read-only buckets keyed by canonical annihilated pair (k,sk),(l,sl)
    std::unordered_map<uint64_t, std::vector<const TwoBodyTerm*>> buckets;
    buckets.reserve(terms.size());
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key2(term.k, term.l, term.spin_k, term.spin_l)].push_back(&term);
        }
    }

    // Thread-local accumulation
    std::vector<std::unordered_set<SlaterDeterminant>> partials;
    partials.reserve(omp_get_max_threads());

    #pragma omp parallel
    {
        std::unordered_set<SlaterDeterminant> local;
        local.reserve(std::max<size_t>(16, basis.size() / std::max(1, omp_get_num_threads())));

        SlaterDeterminant excited_det(n_spatial);
        int8_t sign;

        // Per-thread scratch
        std::vector<std::pair<size_t, Spin>> occupied_orbs;
        occupied_orbs.reserve(n_spatial * 2);

        #pragma omp for schedule(guided)
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(basis.size()); ++idx) {
            const auto& det = basis[static_cast<size_t>(idx)];

            occupied_orbs.clear();
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it)
                occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Alpha);
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it)
                occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Beta);

            const size_t L = occupied_orbs.size();
            for (size_t a = 0; a + 1 < L; ++a) {
                const auto [k, sk] = occupied_orbs[a];
                for (size_t b = a + 1; b < L; ++b) {
                    const auto [l, sl] = occupied_orbs[b];

                    auto key = make_key2(k, l, sk, sl);
                    auto bkt = buckets.find(key);
                    if (bkt == buckets.end()) continue;

                    for (const auto* term : bkt->second) {
                        // ensure the term orientation matches the occupied pair
                        const bool match_kl = (term->k == k && term->l == l &&
                                               term->spin_k == sk && term->spin_l == sl);
                        const bool match_lk = (term->k == l && term->l == k &&
                                               term->spin_k == sl && term->spin_l == sk);
                        if (!(match_kl || match_lk)) continue;

                        if (SlaterDeterminant::apply_excitation_double_fast(
                                det, term->i, term->j, term->k, term->l,
                                term->spin_i, term->spin_j, term->spin_k, term->spin_l,
                                excited_det, sign))
                        {
                            local.insert(excited_det);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            partials.push_back(std::move(local));
        }
    }

    // Merge thread-local sets
    std::unordered_set<SlaterDeterminant> unique_dets;
    size_t hint = 0; for (auto& s : partials) hint += s.size();
    unique_dets.reserve(hint);
    for (auto& s : partials) {
        unique_dets.insert(s.begin(), s.end());
    }

    std::vector<SlaterDeterminant> connected_dets(unique_dets.begin(), unique_dets.end());
    std::sort(connected_dets.begin(), connected_dets.end());
    return connected_dets;
}



} // namespace ci

// ============================== Hashing ======================================

namespace {
inline std::size_t hash_mix_u64(std::size_t seed, ci::u64 v) noexcept {
    // splitmix64-ish mix, then combine
    v ^= v >> 30; v *= 0xbf58476d1ce4e5b9ULL;
    v ^= v >> 27; v *= 0x94d049bb133111ebULL;
    v ^= v >> 31;
    return seed ^ (static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ULL + (seed<<6) + (seed>>2));
}
}

std::size_t std::hash<ci::BitsetVec>::operator()(const ci::BitsetVec& b) const noexcept {
    // Start with a hash of the size, then mix in the words.
    std::size_t seed = std::hash<std::size_t>{}(b.size());
    for (auto w : b.words()) {
        seed = hash_mix_u64(seed, w);
    }
    return seed;
}

std::size_t std::hash<ci::Determinant>::operator()(const ci::Determinant& d) const noexcept {
    std::size_t seed = d.num_orbitals();
    for (auto w : d.bits().words()) seed = hash_mix_u64(seed, w);
    return seed;
}


std::size_t std::hash<ci::SpinDeterminant>::operator()(const ci::SpinDeterminant& sd) const noexcept {
    // This now correctly depends on ALL data within the bitset.
    return std::hash<ci::BitsetVec>{}(sd.raw());
}

std::size_t std::hash<ci::SlaterDeterminant>::operator()(const ci::SlaterDeterminant& s) const noexcept {
    std::size_t h_alpha = std::hash<ci::SpinDeterminant>{}(s.alpha());
    std::size_t h_beta  = std::hash<ci::SpinDeterminant>{}(s.beta());
    // Combine them. The way you do it is a standard technique.
    return h_alpha ^ (h_beta + 0x9e3779b97f4a7c15ULL + (h_alpha << 6) + (h_alpha >> 2));
}


// ---------------------------------------------------------