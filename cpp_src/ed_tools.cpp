#include "ed_tools.h"
#include <stdexcept>

#ifdef _MSC_VER
#include <intrin.h>
#define popcount __popcnt64
#else
#define popcount __builtin_popcountll
#endif

// Helper to calculate the fermionic sign.
// Counts set bits in a state *before* a given index.
static int phase(uint64_t state, int index_0based) {
    if (index_0based == 0) return 1;
    uint64_t mask = (1ULL << index_0based) - 1;
    return (popcount(state & mask) % 2 == 0) ? 1 : -1;
}

std::vector<Triplet> get_annihilation_operator_sparse(int num_spin_orbitals, int orbital_index_0based) {
    if (num_spin_orbitals > 64) {
        throw std::invalid_argument("ED tools support a maximum of 64 spin-orbitals.");
    }
    if (orbital_index_0based < 0 || orbital_index_0based >= num_spin_orbitals) {
        throw std::out_of_range("Orbital index out of range.");
    }

    std::vector<Triplet> triplets;
    uint64_t n_states = 1ULL << num_spin_orbitals;
    uint64_t orbital_mask = 1ULL << orbital_index_0based;

    for (uint64_t state = 0; state < n_states; ++state) {
        // Check if the orbital is occupied
        if ((state & orbital_mask) != 0) {
            uint64_t final_state = state & ~orbital_mask; // Annihilate the particle
            double sign = static_cast<double>(phase(state, orbital_index_0based));
            
            // Note: matrix indices are 0-based for easy conversion to Python/SciPy
            triplets.push_back({static_cast<int>(final_state), static_cast<int>(state), {sign, 0.0}});
        }
    }
    return triplets;
}

std::vector<Triplet> get_creation_operator_sparse(int num_spin_orbitals, int orbital_index_0based) {
    if (num_spin_orbitals > 64) {
        throw std::invalid_argument("ED tools support a maximum of 64 spin-orbitals.");
    }
    if (orbital_index_0based < 0 || orbital_index_0based >= num_spin_orbitals) {
        throw std::out_of_range("Orbital index out of range.");
    }

    std::vector<Triplet> triplets;
    uint64_t n_states = 1ULL << num_spin_orbitals;
    uint64_t orbital_mask = 1ULL << orbital_index_0based;

    for (uint64_t state = 0; state < n_states; ++state) {
        // Check if the orbital is unoccupied (Pauli exclusion)
        if ((state & orbital_mask) == 0) {
            uint64_t final_state = state | orbital_mask; // Create the particle
            double sign = static_cast<double>(phase(state, orbital_index_0based));
            
            triplets.push_back({static_cast<int>(final_state), static_cast<int>(state), {sign, 0.0}});
        }
    }
    return triplets;
}