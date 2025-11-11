#pragma once
// determinants.h
// -----------------------------------------------------------------------------
// Core determinant data structures for quantum chemistry.
// - BitsetVec: A dynamic, 64-bit word-packed bitset.
// - Determinant: A generic representation of spin-orbital occupancies.
// - SpinDeterminant: Occupancy for a single spin channel (alpha or beta).
// - SlaterDeterminant: A pair of SpinDeterminants (alpha and beta).
//
// Also includes hash specializations for use in unordered containers.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <vector>
#include <optional>
#include <string>
#include <functional> // For std::hash

namespace ci {

// Type aliases and enums
using u64 = std::uint64_t;
enum class Spin : uint8_t { Alpha=0, Beta=1 };
enum class SpinOrbitalOrder { AlphaFirst, Interleaved };

// ============================= BitsetVec =====================================
class BitsetVec {
public:
    BitsetVec();
    explicit BitsetVec(std::size_t n_bits);

    std::size_t size()   const noexcept;
    std::size_t nwords() const noexcept;
    const std::vector<u64>& words() const noexcept { return words_; }

    bool test (std::size_t bit) const noexcept;
    void set  (std::size_t bit) noexcept;
    void reset(std::size_t bit) noexcept;
    void clear() noexcept;

    std::size_t popcount_all()   const noexcept;
    std::size_t popcount_below(std::size_t bit_exclusive) const noexcept;
    void        mask_tail() noexcept;
    std::string to_binary() const;

    // Iterator over occupied indices (0-based)
    class OccIter {
    public:
        OccIter();
        int operator*() const noexcept;
        OccIter& operator++();
        bool operator!=(const OccIter& other) const noexcept;
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

    friend bool operator==(const BitsetVec& a, const BitsetVec& b) noexcept;
    friend bool operator<( const BitsetVec& a, const BitsetVec& b) noexcept;

private:
    std::size_t       n_bits_{0};
    std::vector<u64>  words_;
};

// ============================= Determinant ===================================
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
    const BitsetVec& bits() const noexcept;

    friend bool operator==(const Determinant& a, const Determinant& b) noexcept;
    friend bool operator<( const Determinant& a, const Determinant& b) noexcept;

private:
    BitsetVec bits_;
};

// =========================== SpinDeterminant =================================
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

    struct OpResult { SpinDeterminant det; int8_t sign; };
    static std::optional<OpResult> create     (const SpinDeterminant& d, std::size_t i0) noexcept;
    static std::optional<OpResult> annihilate (const SpinDeterminant& d, std::size_t i0) noexcept;

private:
    BitsetVec bits_;
};

// ========================== SlaterDeterminant ================================
class SlaterDeterminant {
public:
    SlaterDeterminant();
    explicit SlaterDeterminant(std::size_t n_spatial);
    SlaterDeterminant(std::size_t n_spatial,
                      const std::vector<int>& occ_alpha0,
                      const std::vector<int>& occ_beta0);

    std::size_t            num_spatial_orbitals() const noexcept;
    const SpinDeterminant& alpha() const noexcept;
    const SpinDeterminant& beta () const noexcept;
    SpinDeterminant&       alpha() noexcept;
    SpinDeterminant&       beta()  noexcept;

    std::size_t count_electrons() const noexcept;
    double      Sz()              const noexcept;

    friend bool operator==(const SlaterDeterminant& a, const SlaterDeterminant& b) noexcept;
    friend bool operator<( const SlaterDeterminant& a, const SlaterDeterminant& b) noexcept;

    struct OpResult { SlaterDeterminant det; int8_t sign; };
    static std::optional<OpResult> create    (const SlaterDeterminant& s, std::size_t i0, Spin spin, SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;
    static std::optional<OpResult> annihilate(const SlaterDeterminant& s, std::size_t i0, Spin spin, SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;

    static bool apply_excitation_single_fast(const SlaterDeterminant& s, size_t i0, size_t j0, Spin spin_i, Spin spin_j, SlaterDeterminant& out, int8_t& sign, SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;
    static bool apply_excitation_double_fast(const SlaterDeterminant& s, size_t i0, size_t j0, size_t k0, size_t l0, Spin spin_i, Spin spin_j, Spin spin_k, Spin spin_l, SlaterDeterminant& out, int8_t& sign, SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst) noexcept;

private:
    std::size_t     n_spatial_{0};
    SpinDeterminant alpha_, beta_;
};

// ======================== Interleave/Deinterleave ============================
Determinant       interleave  (const SlaterDeterminant& s, SpinOrbitalOrder order);
SlaterDeterminant deinterleave(const Determinant& d,       SpinOrbitalOrder order);

} // namespace ci

// ========================== Hash Specializations =============================
namespace std {
template<> struct hash<ci::BitsetVec> {
    size_t operator()(const ci::BitsetVec& b) const noexcept;
};
template<> struct hash<ci::Determinant> {
    size_t operator()(const ci::Determinant& d) const noexcept;
};
template<> struct hash<ci::SpinDeterminant> {
    size_t operator()(const ci::SpinDeterminant& sd) const noexcept;
};
template<> struct hash<ci::SlaterDeterminant> {
    size_t operator()(const ci::SlaterDeterminant& s) const noexcept;
};
} // namespace std