import numpy as np
import h5py
from itertools import combinations
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

import clic_clib as qc

def load_integrals_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        h0 = f['h0'][:]
        U = f['U'][:]
        Ne = int(f.attrs['Ne'])
        Enuc = float(f.attrs['Enuc'])
        K = h0.shape[0]
        M = K // 2
    return h0, U, Ne, Enuc, M, K

def calculate_hf_energy_python(h0, U, Ne, Enuc):
    """Calculates HF energy from raw interleaved integrals, proven to be correct."""
    occupied_indices = range(Ne)
    e1 = sum(h0[i, i] for i in occupied_indices)
    e2 = sum(
        (U[i, j, i, j] - U[i, j, j, i])
        for i in occupied_indices
        for j in occupied_indices
    )
    return e1 + 0.5 * e2 + Enuc

def get_fci_basis(num_spatial, num_electrons):
    """Generates a sorted list of SlaterDeterminant objects for the FCI space."""
    num_spin_orbitals = 2 * num_spatial
    basis_dets = []
    for occupied_indices in combinations(range(num_spin_orbitals), num_electrons):
        occ_a = [i for i in occupied_indices if i < num_spatial]
        occ_b = [i - num_spatial for i in occupied_indices if i >= num_spatial]
        basis_dets.append(qc.SlaterDeterminant(num_spatial, occ_a, occ_b))
    return sorted(basis_dets)

def transform_integrals_interleaved_to_alphafirst(h0_int, U_int, M):
    """
    Transforms integrals from spin-interleaved to AlphaFirst ordering.
    The underlying notation (i.e., meaning of V[p,q,r,s]) is preserved.
    """
    K = 2 * M
    
    # af_map[i_af] gives the corresponding index in the original interleaved basis
    af_map = np.zeros(K, dtype=int)
    for i in range(M):
        af_map[i] = 2 * i
        af_map[i + M] = 2 * i + 1

    h0_af = np.zeros_like(h0_int)
    for p_af in range(K):
        for q_af in range(K):
            h0_af[p_af, q_af] = h0_int[af_map[p_af], af_map[q_af]]
            
    U_af = np.zeros_like(U_int)
    for p_af in range(K):
        for q_af in range(K):
            for r_af in range(K):
                for s_af in range(K):
                    p_int, q_int = af_map[p_af], af_map[q_af]
                    r_int, s_int = af_map[r_af], af_map[s_af]
                    U_af[p_af, q_af, r_af, s_af] = U_int[p_int, q_int, r_int, s_int]
                    
    return h0_af, U_af

def test_h2o_fci_energy():
    print("--- Testing FCI Ground State Energy of H2O/STO-3G ---")
    
    # 1. Load data
    h0_interleaved, U_interleaved, Ne, Enuc, M, K = load_integrals_from_h5("h2o_h0U.h5")
    
    # 2. Transform integrals to AlphaFirst basis
    print("\nTransforming integrals to AlphaFirst basis...")
    h0_alphafirst_raw, U_alphafirst_raw = transform_integrals_interleaved_to_alphafirst(
        h0_interleaved, U_interleaved, M
    )
    
    # 3. CRITICAL: Ensure arrays are C-contiguous and have the correct dtype
    h0_alphafirst = np.ascontiguousarray(h0_alphafirst_raw, dtype=np.complex128)
    U_alphafirst = np.ascontiguousarray(U_alphafirst_raw, dtype=np.complex128)

    # 4. Generate the FCI basis
    print("\nGenerating FCI basis...")
    fci_basis = get_fci_basis(M, Ne)
    print(f"  FCI basis size for ({M}o, {Ne}e) = {len(fci_basis)} determinants.")
    
    # 5. Build the full FCI Hamiltonian matrix
    print("\nBuilding FCI Hamiltonian matrix with C++ kernel...")
    H_sparse = qc.build_hamiltonian_openmp(fci_basis, h0_alphafirst, U_alphafirst)
    print(f"  Hamiltonian construction complete. Matrix shape: {H_sparse.shape}")
    
    # 6. Diagonalize to find the lowest eigenvalue
    print("\nDiagonalizing FCI Hamiltonian...")
    electronic_eigenvalues, _ = eigsh(H_sparse, k=4, which='SA')
    #electronic_eigenvalues, _ = eigh(H_sparse.toarray())
    electronic_gs_energy = electronic_eigenvalues[0]
    
    # 7. Calculate total energy and validate
    total_gs_energy = electronic_gs_energy + Enuc
    print(f"\n  Lowest electronic energy = {electronic_gs_energy:.8f} Hartree")
    print(f"  Total ground state energy (Electronic + Nuclear) = {total_gs_energy:.8f} Hartree")
    
    # Reference value for FCI/STO-3G water with these integrals
    reference_fci_energy = -75.0232909847239
    print(f"  Reference FCI energy = {reference_fci_energy:.8f} Hartree")
    
    np.testing.assert_allclose(total_gs_energy, reference_fci_energy, atol=1e-6)
    print("\n✅✅✅ FINAL SUCCESS: Calculated FCI energy matches the reference value.")

if __name__ == "__main__":
    # You can keep the old test function if you want, but this is the main one
    test_h2o_fci_energy()

