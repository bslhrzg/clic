// cpp_src/applyH.h
#pragma once

#include "ci_core.h"
#include "slater_condon.h"
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace ci {

// Corresponds to Python's Sh0: map from annihilated orbital `r` to created orbitals `p`.
using TableSh0 = std::unordered_map<uint32_t, std::vector<uint32_t>>;

// Corresponds to Python's SU: map `r` -> `p` -> {spectators `s`}.
using TableSU = std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<uint32_t>>>;

// Corresponds to Python's D: map {`r`,`s`} -> {{`p`,`q`}}.
// Key: r | (s << 32) where r < s.
// Value: p | (q << 32).
using TableD = std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>>;

struct ScreenedHamiltonian {
    size_t n_spin_orbitals = 0;
    TableSh0 sh0;
    TableSU su;
    TableD d;
};

// Build the screening tables from integral views.
ScreenedHamiltonian build_screened_hamiltonian(
    size_t K, const H1View& H, const ERIView& V, double tol
);

// Apply the screened Hamiltonian to a wavefunction, exactly mirroring the Python logic.
Wavefunction apply_hamiltonian(
    const Wavefunction& psi,
    const ScreenedHamiltonian& screened_H,
    const H1View& H,
    const ERIView& V,
    double tol_element
);

} // namespace ci