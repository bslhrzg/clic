# CI Solver Library (`cisolverlib`)

A high-performance C++ backend with Python bindings for Configuration Interaction (CI) calculations.

## Overview

`cisolverlib` provides a powerful and efficient toolset for quantum chemistry calculations. It includes:
- Core C++ data structures for `SlaterDeterminant` and `Wavefunction` objects.
- An efficient implementation of the Slater-Condon rules for calculating Hamiltonian matrix elements.
- A fast, OpenMP-parallelized routine to build the full CI Hamiltonian matrix in a given determinant basis.
- Functions for dynamic operator application (`H|Ψ⟩`) and basis connectivity analysis.

## Features

- **Object-Oriented API**: Manipulate `SlaterDeterminant` and `Wavefunction` objects directly from Python.
- **High Performance**: Hamiltonian construction and operator application are implemented in C++20 for maximum speed.
- **Parallelism**: Utilizes OpenMP to parallelize expensive loops.
- **Interoperability**: Easily integrates with NumPy for defining integrals and SciPy for sparse matrix diagonalization.

## Installation and Build

1.  **Requirements**:
    - A C++20 compliant compiler (e.g., modern GCC, Clang).
    - CMake (version 3.18+).
    - Python (3.8+).
    - `pybind11`, `numpy`, `scipy`, `h5py` Python packages.
    - For macOS with OpenMP, `llvm` is required (`brew install llvm`).

2.  **Build Steps**:
    From the project root, run the following commands:

    ```bash
    # (Activate your virtual environment first)
    pip install pybind11 numpy scipy h5py

    # After editing ANY file in clic/ or clic/cpp_src/
    CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ pip install -e .
    ```

## Quick Start: The Hubbard Dimer

This example builds and diagonalizes the Hamiltonian for the 2-electron Hubbard dimer.

```python
import numpy as np
from scipy.sparse.linalg import eigsh
import qmcisolver_core as qc

# --- System Definition ---
M = 2  # Spatial orbitals
Nelec = 2
t = 1.0
U = 4.0

# --- 1. Generate the FCI Basis ---
basis = [
    qc.SlaterDeterminant(M, [], [0,1]),      # |β₀, β₁⟩
    qc.SlaterDeterminant(M, [0], [0]),       # |α₀, β₀⟩
    qc.SlaterDeterminant(M, [0], [1]),       # |α₀, β₁⟩
    qc.SlaterDeterminant(M, [1], [0]),       # |α₁, β₀⟩
    qc.SlaterDeterminant(M, [1], [1]),       # |α₁, β₁⟩
    qc.SlaterDeterminant(M, [0,1], [])       # |α₀, α₁⟩
]
basis.sort()

# --- 2. Define Integrals (AlphaFirst basis) ---
K = 2 * M
h0 = np.zeros((K, K), dtype=np.complex128)
h0[0, 1] = h0[1, 0] = -t  # α hopping
h0[2, 3] = h0[3, 2] = -t  # β hopping

# Note: The C++ code expects V[p,q,r,s] = <pq|V|rs>
U_mat = np.zeros((K, K, K, K), dtype=np.complex128)
U_mat[0, 2, 0, 2] = U  # <α₀,β₀|V|α₀,β₀>
U_mat[1, 3, 1, 3] = U  # <α₁,β₁|V|α₁,β₁>

# --- 3. Build and Diagonalize Hamiltonian ---
H_mat = qc.build_hamiltonian_openmp(basis, h0, U_mat)
eigvals, _ = eigsh(H_mat, k=1, which='SA')

print(f"Hubbard Dimer Ground State Energy: {eigvals[0]:.8f}")

