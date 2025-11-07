#pragma once
// ci_core.h
// -----------------------------------------------------------------------------
// Core building blocks for a CI code (C++20).
// - 0-based indexing throughout the public API
// - Bit-packed occupancy with tail masking for correctness
// - Determinant (combined spin-orbitals) and SpinDeterminant (one spin)
// - SlaterDeterminant (alpha+beta) with creation/annihilation
// - Interleave/deinterleave between spin-separated and combined representations
// - Wavefunction over Slater determinants with operator application
// - Configurable spin-orbital ordering to control fermionic signs
//
// This header contains class declarations and hash specializations.
// Implementations are in ci_core.cpp.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <vector>
#include <optional>
#include <compare>
#include <string>
#include <complex>
#include <unordered_map>

namespace ci {

// -----------------------------------------------------------------------------
// Spin-orbital linearization choices (controls β signs).
//  - AlphaFirst:    (α0, α1, ..., αM-1, β0, β1, ..., βM-1)
//  - Interleaved:   (α0, β0, α1, β1, ..., αM-1, βM-1)
// -----------------------------------------------------------------------------
enum class SpinOrbitalOrder { AlphaFirst, Interleaved };

enum class Spin : uint8_t { Alpha=0, Beta=1 };

using u64 = std::uint64_t;

// -----------------------------------------------------------------------------
// Internal bit storage (value type). Exposed only via public classes.
// -----------------------------------------------------------------------------
class BitsetVec {
public:
    BitsetVec();
    explicit BitsetVec(std::size_t n_bits);

    std::size_t size()   const noexcept;
    std::size_t nwords() const noexcept;

    bool test (std::size_t bit) const noexcept;
    void set  (std::size_t bit) noexcept;
    void reset(std::size_t bit) noexcept;
    void clear() noexcept;

    std::size_t popcount_all()   const noexcept;
    std::size_t popcount_below(std::size_t bit_exclusive) const noexcept;
    void        mask_tail() noexcept;

    // Binary string, MSB-first, trimmed to size(); for debugging/logging.
    std::string to_binary() const;

    // Iterator over occupied indices (0-based) without allocating.
    class OccIter {
    public:
        OccIter();
        int            operator*() const noexcept; // returns 0-based index
        OccIter&       operator++();               // pre-increment
        bool           operator!=(const OccIter& other) const noexcept;

    private:
        friend class BitsetVec;
        explicit OccIter(const BitsetVec* bs, std::size_t start);
        void advance_to_next();
        const BitsetVec* bs_{};
        std::size_t idx_{}, w_{}, b_{};
        int current0_{-1};
        bool done_{true};
    };

    OccIter begin_occ() const;
    OccIter end_occ()   const;

    // Access for hashing
    const std::vector<u64>& words() const noexcept { return words_; }

    friend bool operator==(const BitsetVec& a, const BitsetVec& b) noexcept;
    friend bool operator<( const BitsetVec& a, const BitsetVec& b) noexcept;

private:
    std::size_t       n_bits_{0};
    std::vector<u64>  words_;
};

// -----------------------------------------------------------------------------
// Determinant: combined spin-orbital occupancy (length K = 2*M or arbitrary)
// 0-based indices.
// -----------------------------------------------------------------------------
class Determinant {
public:
    Determinant();
    explicit Determinant(std::size_t total_orbitals);
    Determinant(std::size_t total_orbitals, const std::vector<int>& occupied0);

    std::size_t num_orbitals() const noexcept;

    bool occupied(std::size_t orb0) const noexcept;
    void set     (std::size_t orb0) noexcept;
    void reset   (std::size_t orb0) noexcept;

    std::size_t count_electrons() const noexcept;

    BitsetVec::OccIter begin_occ() const;
    BitsetVec::OccIter end_occ()   const;

    std::string to_string_binary() const;

    friend bool operator==(const Determinant& a, const Determinant& b) noexcept;
    friend bool operator<( const Determinant& a, const Determinant& b) noexcept;

    const BitsetVec& bits() const noexcept;

private:
    BitsetVec bits_;
};

// -----------------------------------------------------------------------------
// SpinDeterminant: occupancy for ONE spin over M spatial orbitals.
// 0-based indices.
// -----------------------------------------------------------------------------
class SpinDeterminant {
public:
    SpinDeterminant();
    explicit SpinDeterminant(std::size_t n_spatial);
    SpinDeterminant(std::size_t n_spatial, const std::vector<int>& occupied0);

    std::size_t num_orbitals()   const noexcept;
    bool        occupied(std::size_t i0) const noexcept;
    void        set     (std::size_t i0) noexcept;
    void        reset   (std::size_t i0) noexcept;
    std::size_t count_electrons() const noexcept;

    BitsetVec::OccIter begin_occ() const;
    BitsetVec::OccIter end_occ()   const;

    const BitsetVec& raw() const noexcept;

    friend bool operator==(const SpinDeterminant& a, const SpinDeterminant& b) noexcept;
    friend bool operator<( const SpinDeterminant& a, const SpinDeterminant& b) noexcept;

    struct OpResult;

    // Returns nullopt if Pauli-forbidden; otherwise new determinant + sign.
    static std::optional<OpResult> create     (const SpinDeterminant& d, std::size_t i0) noexcept;
    static std::optional<OpResult> annihilate (const SpinDeterminant& d, std::size_t i0) noexcept;

private:
    BitsetVec bits_;
};

struct SpinDeterminant::OpResult { SpinDeterminant det; int8_t sign; };


// -----------------------------------------------------------------------------
// SlaterDeterminant: spin-separated determinant (α + β) over M spatial orbitals.
// 0-based indices.
// -----------------------------------------------------------------------------
class SlaterDeterminant {
public:
    SlaterDeterminant();
    explicit SlaterDeterminant(std::size_t n_spatial);
    SlaterDeterminant(std::size_t n_spatial,
                      const std::vector<int>& occ_alpha0,
                      const std::vector<int>& occ_beta0);

    std::size_t           num_spatial_orbitals() const noexcept;
    const SpinDeterminant& alpha() const noexcept;
    const SpinDeterminant& beta () const noexcept;

    // NON-CONST OVERLOADS FOR MUTABLE ACCESS
    SpinDeterminant& alpha() noexcept;
    SpinDeterminant& beta() noexcept;

    std::size_t count_electrons() const noexcept;
    double      Sz()              const noexcept;

    friend bool operator==(const SlaterDeterminant& a, const SlaterDeterminant& b) noexcept;
    friend bool operator<( const SlaterDeterminant& a, const SlaterDeterminant& b) noexcept;

    struct OpResult ;

    // Apply c†_{i,σ} or c_{i,σ} directly on basis determinants.
    static std::optional<OpResult>
    create    (const SlaterDeterminant& s, std::size_t i0, Spin spin,
               SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;

    static std::optional<OpResult>
    annihilate(const SlaterDeterminant& s, std::size_t i0, Spin spin,
               SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;

    // --- Single and Double Excitation Operators ---
    // Apply c†_i c_j
    static std::optional<OpResult>
    apply_excitation_single(const SlaterDeterminant& s, size_t i0, size_t j0,
                            Spin spin_i, Spin spin_j,
                            SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;

    // Apply c†_i c†_j c_l c_k
    static std::optional<OpResult>
    apply_excitation_double(const SlaterDeterminant& s,
                            size_t i0, size_t j0, size_t k0, size_t l0,
                            Spin spin_i, Spin spin_j, Spin spin_k, Spin spin_l,
                            SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;

    static bool
    apply_excitation_single_fast(const SlaterDeterminant& s, size_t i0, size_t j0,
                                 Spin spin_i, Spin spin_j, SlaterDeterminant& out, int8_t& sign,
                                 SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;

    static bool
    apply_excitation_double_fast(const SlaterDeterminant& s,
                                 size_t i0, size_t j0, size_t k0, size_t l0,
                                 Spin spin_i, Spin spin_j, Spin spin_k, Spin spin_l,
                                 SlaterDeterminant& out, int8_t& sign,
                                 SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;

private:
    std::size_t     n_spatial_{0};
    SpinDeterminant alpha_, beta_;
};

struct SlaterDeterminant::OpResult { SlaterDeterminant det; int8_t sign; };


// -----------------------------------------------------------------------------
// Interleave/deinterleave between spin-separated and combined determinants
// -----------------------------------------------------------------------------
Determinant        interleave  (const SlaterDeterminant& s,
                                SpinOrbitalOrder order = SpinOrbitalOrder::Interleaved);

SlaterDeterminant  deinterleave(const Determinant& d,
                                SpinOrbitalOrder order = SpinOrbitalOrder::Interleaved);


} // namespace ci

// -----------------------------------------------------------------------------
// std::hash specializations (must live in a header so all TUs see them)
// -----------------------------------------------------------------------------
#include <functional>

template<> struct std::hash<ci::Determinant> {
    std::size_t operator()(const ci::Determinant& d) const noexcept;
};

template<> struct std::hash<ci::SpinDeterminant> {
    std::size_t operator()(const ci::SpinDeterminant& sd) const noexcept;
};

template<> struct std::hash<ci::SlaterDeterminant> {
    std::size_t operator()(const ci::SlaterDeterminant& s) const noexcept;
};

template<> struct std::hash<ci::BitsetVec> {
    std::size_t operator()(const ci::BitsetVec& b) const noexcept;
};




namespace ci {
// --- NEW: Compact term structs for operator representation (Suggestion D) ---
using Coeff = std::complex<double>;

struct OneBodyTerm {
    size_t i, j;
    Spin spin_i, spin_j;
    Coeff value;
};

struct TwoBodyTerm {
    size_t i, j, k, l;
    Spin spin_i, spin_j, spin_k, spin_l;
    Coeff value;
};

// -----------------------------------------------------------------------------
// Wavefunction over Slater determinants
// -----------------------------------------------------------------------------
class Wavefunction {
public:
    using Coeff = std::complex<double>;
    using Data  = std::unordered_map<SlaterDeterminant, Coeff>;

    explicit Wavefunction(std::size_t n_spatial);

    Wavefunction(std::size_t n_spatial, const Data& init);


    Wavefunction(std::size_t n_spatial, 
        const std::vector<SlaterDeterminant>& basis, 
        const std::vector<Coeff>& coeffs,
        bool keep_zeros = true);

    std::size_t  num_spatial_orbitals() const noexcept;
    const Data&  data() const noexcept;

    void add_term(const SlaterDeterminant& det, Coeff c, double tol=0);
    void normalize(double tol=1e-15);
    void prune(double threshold);

    Coeff dot(const Wavefunction& other) const;

    void add_wavefunction(const Wavefunction& other, Coeff scale = {1.0, 0.0});


    std::vector<SlaterDeterminant> basis_sorted() const;
    std::vector<Coeff>             coeffs_sorted(const std::vector<SlaterDeterminant>& basis) const;
    

private:
    std::size_t n_spatial_{0};
    Data        data_;
};

// Wavefunction operator application (on Slater basis)
Wavefunction apply_creation   (const Wavefunction& wf, std::size_t i0, Spin spin,
                               SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst);

Wavefunction apply_annihilation(const Wavefunction& wf, std::size_t i0, Spin spin,
                               SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst);

Wavefunction apply_one_body_operator(const Wavefunction& wf,
                                     const std::vector<OneBodyTerm>& terms);

Wavefunction apply_two_body_operator(const Wavefunction& wf,
                                     const std::vector<TwoBodyTerm>& terms);

std::vector<SlaterDeterminant>
get_connections_one_body(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<OneBodyTerm>& terms);
                         
std::vector<SlaterDeterminant>
get_connections_two_body(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<TwoBodyTerm>& terms);

} // namespace ci

