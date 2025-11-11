#pragma once
// wavefunction.h
// -----------------------------------------------------------------------------
// A representation of a quantum wavefunction as a linear combination of
// Slater determinants.
// -----------------------------------------------------------------------------

#include "determinants.h" // Depends on SlaterDeterminant
#include <vector>
#include <complex>
#include <unordered_map>

namespace ci {

class Wavefunction {
public:
    using Coeff = std::complex<double>;
    using Data  = std::unordered_map<SlaterDeterminant, Coeff>;

    explicit Wavefunction(std::size_t n_spatial);
    Wavefunction(std::size_t n_spatial, const Data& init);
    Wavefunction(std::size_t n_spatial, 
                 const std::vector<SlaterDeterminant>& basis, 
                 const std::vector<Coeff>& coeffs,
                 bool keep_zeros = false);

    std::size_t  num_spatial_orbitals() const noexcept;
    const Data&  data() const noexcept;

    void add_term(const SlaterDeterminant& det, Coeff c, double tol = 1e-16);
    void normalize(double tol = 1e-15);
    void prune(double threshold);

    Coeff dot(const Wavefunction& other) const;
    void add_wavefunction(const Wavefunction& other, Coeff scale = {1.0, 0.0});

    std::vector<SlaterDeterminant> basis_sorted() const;
    std::vector<Coeff>             coeffs_sorted(const std::vector<SlaterDeterminant>& basis) const;

private:
    std::size_t n_spatial_{0};
    Data        data_;
};

} // namespace ci