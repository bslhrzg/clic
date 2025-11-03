import numpy as np
import h5py
from itertools import combinations
from scipy.sparse.linalg import eigsh
#import clic_clib as qc
from clic import *
import time

# --- Integral Transformation Functions ---

def load_spatial_integrals(filename):
    """Loads spatial integrals and metadata from the HDF5 file."""
    with h5py.File(filename, "r") as f:
        hcore = f["h0"][:]
        ee = f["U"][:]
    Enuc = 9.003584105404158

    M = hcore.shape[0]
    print(f"Loaded data from {filename}:")
    print(f"  Spatial Orbitals (M) = {M}, Nuclear Repulsion (Enuc) = {Enuc:.8f}")
    return hcore, ee, Enuc, M

def save_integrals_to_h5(filename, h0, U, Ne, Enuc):
    """Save one- and two-body integrals and metadata to an HDF5 file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('h0', data=np.asarray(h0))
        f.create_dataset('U', data=np.asarray(U))
        f.attrs['Ne'] = int(Ne)
        f.attrs['Enuc'] = float(Enuc)
    print(f"Saved integrals to {filename}")

# --- Helper to create sparse operator terms from dense Numpy arrays ---
def get_one_body_terms(h1_matrix, M):
    terms = []
    for i in range(2*M):
        for j in range(2*M):
            if abs(h1_matrix[i, j]) > 1e-12:
                spin_i = Spin.Alpha if i < M else Spin.Beta
                spin_j = Spin.Alpha if j < M else Spin.Beta
                orb_i = i if i < M else i - M
                orb_j = j if j < M else j - M
                terms.append((orb_i, orb_j, spin_i, spin_j, complex(h1_matrix[i, j])))
    return terms

def get_two_body_terms(v2_tensor, M):
    terms = []
    for i in range(2*M):
        for j in range(2*M):
            for k in range(2*M):
                for l in range(2*M):
                    if abs(v2_tensor[i, j, k, l]) > 1e-12:
                        spins = [Spin.Alpha if idx < M else Spin.Beta for idx in [i, j, k, l]]
                        orbs = [idx if idx < M else idx - M for idx in [i, j, k, l]]
                        terms.append((orbs[0], orbs[1], orbs[2], orbs[3],
                                      spins[0], spins[1], spins[2], spins[3],
                                      complex(v2_tensor[i, j, k, l])))
    return terms

# --- Main Test Function ---

def test_h2o_iterative_ci():
    print("--- Testing Iterative CI for H2O/6-31G ---")
    
    # --- 1. Load and Transform Integrals ---
    h_core, ee_mo, Enuc, M = load_spatial_integrals("my_data.hdf5")
    
    print("\nConverting spatial integrals to spin-orbital form (AlphaFirst)...")
    h0 = double_h(h_core, M)
    # Since C++ code and Julia code use same formula, and Julia uses physicist notation,
    # we just need to spin-adapt the physicist's notation integrals.
    U_phys = umo2so(ee_mo, M)
    
    h0_clean = np.ascontiguousarray(h0, dtype=np.complex128)
    U_clean = np.ascontiguousarray(U_phys, dtype=np.complex128)


    
    # We still need operator terms for the dynamic part.
    # U_phys is <ij|V|kl>. The C++ expects V[i,j,k,l] to be passed to get_connections
    # for the operator c_i† c_j† c_l c_k. This matches.
    one_body_terms = get_one_body_terms(h0_clean, M)
    two_body_terms = get_two_body_terms(U_clean, M)
    
    # --- 2. Define HF and run calculations at each CI level ---
    Ne = 10
    hf_det = SlaterDeterminant(M, list(range(Ne//2)), list(range(Ne//2)))
    save_integrals_to_h5("h2o_631g_alphafirst.h5", h0_clean, U_clean, Ne, Enuc)

    ref_energies = {
        "HF": -75.98394138177851,
        "CISD": -76.11534669560683,
        "CISDT": -76.122,
        "CISDTQ": -76.12236,
    }

    # --- Iteration Loop ---
    current_basis = sorted([hf_det])
    
    for level in ["HF", "CISD", "CISDT", "CISDTQ"]:
        print(f"\n--- Calculating {level} Energy ---")
        
        if level != "HF":
            print(f"Expanding basis from {len(current_basis)} determinants...")
            t_start = time.time()
            connected_by_H1 = get_connections_one_body(current_basis, one_body_terms)
            connected_by_H2 = get_connections_two_body(current_basis, two_body_terms)
            
            new_basis_set = set(current_basis) | set(connected_by_H1) | set(connected_by_H2)
            current_basis = sorted(list(new_basis_set))
            t_end = time.time()
            print(f"  New basis size = {len(current_basis)} (generated in {t_end - t_start:.2f}s)")

        print(f"Building {level} Hamiltonian ({len(current_basis)}x{len(current_basis)})...")
        t_start = time.time()
        H_sparse = build_hamiltonian_openmp(current_basis, h0_clean, U_clean)
        t_end = time.time()
        print(f"  Hamiltonian built in {t_end - t_start:.2f}s")
        
        print("Diagonalizing...")
        t_start = time.time()
        # CORRECTED EIGENSOLVER HANDLING
        if len(current_basis) == 1:
            electronic_gs_energy = H_sparse[0, 0]
        else:
            eigvals, _ = eigsh(H_sparse, k=1, which='SA')
            electronic_gs_energy = eigvals[0]
        total_gs_energy = electronic_gs_energy + Enuc
        t_end = time.time()
        print(f"  Diagonalized in {t_end - t_start:.2f}s")
        
        print(f"  {level} Total Energy = {np.real(total_gs_energy):.8f}")
        print(f"  Reference Energy   = {ref_energies[level]:.8f}")
        np.testing.assert_allclose(total_gs_energy, ref_energies[level], atol=1e-3)
        print(f"  ✅ {level} energy is correct.")

if __name__ == "__main__":
    test_h2o_iterative_ci()
