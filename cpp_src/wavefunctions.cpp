// wavefunction.cpp
#include "wavefunction.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ci {

// ================================ Wavefunction ===============================
Wavefunction::Wavefunction(std::size_t n_spatial) : n_spatial_(n_spatial) {}
Wavefunction::Wavefunction(std::size_t n_spatial, const Data& init)
    : n_spatial_(n_spatial), data_(init) {}

Wavefunction::Wavefunction(std::size_t n_spatial, 
    const std::vector<SlaterDeterminant>& basis, 
    const std::vector<Coeff>& coeffs,
    bool keep_zeros)
    : n_spatial_(n_spatial)
{
    if (basis.size() != coeffs.size()) {
        throw std::invalid_argument("Basis and coefficients vectors must have the same size.");
    }
    data_.reserve(basis.size());

    if (keep_zeros) {
        for (size_t i = 0; i < basis.size(); ++i) {
            data_[basis[i]] = coeffs[i];
        }
    } else {
        for (size_t i = 0; i < basis.size(); ++i) {
            add_term(basis[i], coeffs[i], 0.0);
        }
    }
}

std::size_t Wavefunction::num_spatial_orbitals() const noexcept { return n_spatial_; }
const Wavefunction::Data& Wavefunction::data() const noexcept { return data_; }

void Wavefunction::add_term(const SlaterDeterminant& det, Coeff c, double tol) {
    if (std::abs(c) == 0.0) return;
    auto& v = data_[det];
    v += c;
    if (std::abs(v) < tol) data_.erase(det);
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

Wavefunction::Coeff Wavefunction::dot(const Wavefunction& other) const {
    if (n_spatial_ != other.n_spatial_) {
        throw std::invalid_argument("Wavefunctions must have the same number of spatial orbitals for dot product.");
    }

    Coeff result = {0.0, 0.0};
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

} // namespace ci