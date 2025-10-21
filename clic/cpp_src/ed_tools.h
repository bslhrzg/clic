#ifndef ED_TOOLS_H
#define ED_TOOLS_H

#include <cstdint>
#include <vector>
#include <complex>

// We need to pass back a sparse matrix. Using triplets is the standard way.
struct Triplet {
    int row;
    int col;
    std::complex<double> value;
};

// Creates the annihilation operator c_i as a sparse matrix.
// This is the C++ equivalent of your `destruct` function.
std::vector<Triplet> get_annihilation_operator_sparse(int num_spin_orbitals, int orbital_index_0based);

// Creates the creation operator c_i^† as a sparse matrix.
std::vector<Triplet> get_creation_operator_sparse(int num_spin_orbitals, int orbital_index_0based);

#endif // ED_TOOLS_H