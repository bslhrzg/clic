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
    h0_alphafirst = 0.0* np.ascontiguousarray(h0_alphafirst_raw, dtype=np.complex128)
    U_alphafirst =  np.ascontiguousarray(U_alphafirst_raw, dtype=np.complex128)


    # 4. Generate the FCI basis
    print("\nGenerating FCI basis...")
    fci_basis = get_fci_basis(M, Ne)
    print(f"  FCI basis size for ({M}o, {Ne}e) = {len(fci_basis)} determinants.")
    
    # 5. Build the full FCI Hamiltonian matrix
    print("\nBuilding FCI Hamiltonian matrix with C++ kernel...")
    H_sparse = qc.build_hamiltonian_openmp(fci_basis, h0_alphafirst, U_alphafirst)
    print(f"  Hamiltonian construction complete. Matrix shape: {H_sparse.shape}")


    #============================================================================
    #============================================================================
    # The Test: Apply H to the Hartree-Fock state and compare results
    #============================================================================

    Hfci = H_sparse.toarray()

    # The HF state is | up, up, down, down > = | a0, a1, b0, b1 >
    hf_det = qc.SlaterDeterminant(M, list(range(Ne//2)), list(range(Ne//2)))
    wfhf = qc.Wavefunction(M, [hf_det], [1.0])

    # Apply your new function
    # Using tol=0 ensures we don't miss any connections
    Hhf_new = qc.apply_hamiltonian(wfhf, h0_alphafirst, U_alphafirst, 0, 0)
    # Find the index of the HF determinant in the FCI basis
    try:
        hf_index = fci_basis.index(hf_det)
    except ValueError:
        print("Error: HF determinant not found in FCI basis.")
        exit()

    print(f"len gen = {len(Hhf_new.get_basis())}")
    print(f"<hf|H|hf> = {Hhf_new.amplitude(hf_det)}")

    # The reference vector is the column of H corresponding to the HF state
    hf_column_ref = Hfci[:, hf_index]

    print("\n--- Comparing H|HF> results ---")
    print(f"{'Determinant':<45s} {'Reference':>15s} {'Calculated':>15s} {'abs. diff':>15s} {'Status'}")
    print("-" * 85)

    n_mismatch = 0
    tol = 1e-9

    # Iterate through the entire FCI space to check all possible connections
    for i, det_i in enumerate(fci_basis):
        cref = hf_column_ref[i]
        if np.abs(cref) > 1e-12:
            # Get the calculated amplitude using your bound function
            amp_new = Hhf_new.amplitude(det_i)

            # Check for discrepancies
            is_zero_ref = abs(cref) < tol
            is_zero_new = abs(amp_new) < tol
            
            status = ""
            # Only print entries that should be non-zero or where there's a mismatch
            if not is_zero_ref or not is_zero_new:
                if is_zero_ref and not is_zero_new:
                    status = "!! SPURIOUS !!" # New function created a connection that shouldn't exist
                    n_mismatch += 1
                elif not is_zero_ref and is_zero_new:
                    status = "!! MISSING !!"  # New function missed a connection
                    n_mismatch += 1
                elif abs(cref - amp_new) > tol:
                    status = "!! WRONG VALUE !!" # Value/sign mismatch
                    n_mismatch += 1
                else:
                    status = "OK"

                print(f"{str(det_i):<45s} {cref:15.8f} {amp_new:15.8f} {np.abs(cref-amp_new)} {status}")

    print("-" * 85)
    if n_mismatch == 0:
        print("SUCCESS: All calculated matrix elements match the reference.")
    else:
        print(f"FAILURE: Found {n_mismatch} mismatched matrix elements.")

    # Additionally, let's check for any determinants generated that are NOT in the FCI space (should be none)
    fci_basis_set = set(fci_basis)
    spurious_dets_outside_fci = []
    for det_new in Hhf_new.get_basis():
        if det_new not in fci_basis_set:
            spurious_dets_outside_fci.append(det_new)

    if spurious_dets_outside_fci:
        print("\n!! CRITICAL ERROR: Generated determinants outside the FCI space:")
        for d in spurious_dets_outside_fci:
            print(f"  - {d}")
    
    #============================================================================

    #============================================================================
    #============================================================================
    # Now using existing routines
    #============================================================================
    def get_one_body_terms(h1_matrix):
        """Converts a dense (K,K) numpy array to a sparse list of tuples for C++."""
        terms = []
        M = h1_matrix.shape[0] // 2
        for i in range(2 * M):
            for j in range(2 * M):
                if abs(h1_matrix[i, j]) > 0:
                    spin_i = qc.Spin.Alpha if i < M else qc.Spin.Beta
                    spin_j = qc.Spin.Alpha if j < M else qc.Spin.Beta
                    orb_i = i if i < M else i - M
                    orb_j = j if j < M else j - M
                    terms.append((orb_i, orb_j, spin_i, spin_j, complex(h1_matrix[i, j])))
        return terms

    def get_two_body_terms(v2_tensor):
        """Converts a dense (K,K,K,K) numpy array to a sparse list of tuples for C++."""
        terms = []
        M = v2_tensor.shape[0] // 2
        for i in range(2 * M):
            for j in range(2 * M):
                for k in range(2 * M):
                    for l in range(2 * M):
                        if abs(v2_tensor[i, j, k, l]) > 0:
                            spins = [qc.Spin.Alpha if idx < M else qc.Spin.Beta for idx in [i, j, k, l]]
                            orbs = [idx if idx < M else idx - M for idx in [i, j, k, l]]
                            terms.append((orbs[0], orbs[1], orbs[2], orbs[3],
                spins[0], spins[1], spins[2], spins[3],
                complex(v2_tensor[i, j, k, l])))
        return terms
    
    one_body_terms = get_one_body_terms(h0_alphafirst)
    two_body_terms = get_two_body_terms(0.5 * U_alphafirst)


    # The HF state is | up, up, down, down > = | a0, a1, b0, b1 >
    hf_det = qc.SlaterDeterminant(M, list(range(Ne//2)), list(range(Ne//2)))
    wfhf = qc.Wavefunction(M, [hf_det], [1.0])

    # Apply your new function
    # Using tol=0 ensures we don't miss any connections
    Hhf_new = qc.apply_one_body_operator(wfhf,one_body_terms) + qc.apply_two_body_operator(wfhf,two_body_terms)
    print(f"len gen = {len(Hhf_new.get_basis())}")

    # Find the index of the HF determinant in the FCI basis
    try:
        hf_index = fci_basis.index(hf_det)
    except ValueError:
        print("Error: HF determinant not found in FCI basis.")
        exit()

    # The reference vector is the column of H corresponding to the HF state
    hf_column_ref = Hfci[:, hf_index]

    print("\n--- Comparing H|HF> results ---")
    print(f"{'Determinant':<45s} {'Reference':>15s} {'Calculated':>15s} {'abs. diff':>15s} {'Status'}")
    print("-" * 85)

    n_mismatch = 0
    tol = 1e-9

    # Iterate through the entire FCI space to check all possible connections
    for i, det_i in enumerate(fci_basis):
        cref = hf_column_ref[i]
        if np.abs(cref) > 1e-12:
        
            # Get the calculated amplitude using your bound function
            amp_new = Hhf_new.amplitude(det_i)

            # Check for discrepancies
            is_zero_ref = abs(cref) < tol
            is_zero_new = abs(amp_new) < tol
            
            status = ""
            # Only print entries that should be non-zero or where there's a mismatch
            if not is_zero_ref or not is_zero_new:
                if is_zero_ref and not is_zero_new:
                    status = "!! SPURIOUS !!" # New function created a connection that shouldn't exist
                    n_mismatch += 1
                elif not is_zero_ref and is_zero_new:
                    status = "!! MISSING !!"  # New function missed a connection
                    n_mismatch += 1
                elif abs(cref - amp_new) > tol:
                    status = "!! WRONG VALUE !!" # Value/sign mismatch
                    n_mismatch += 1
                else:
                    status = "OK"

                print(f"{str(det_i):<45s} {cref:15.8f} {amp_new:15.8f} {np.abs(cref-amp_new)} {status}")

    print("-" * 85)
    if n_mismatch == 0:
        print("SUCCESS: All calculated matrix elements match the reference.")
    else:
        print(f"FAILURE: Found {n_mismatch} mismatched matrix elements.")

    # Additionally, let's check for any determinants generated that are NOT in the FCI space (should be none)
    fci_basis_set = set(fci_basis)
    spurious_dets_outside_fci = []
    for det_new in Hhf_new.get_basis():
        if det_new not in fci_basis_set:
            spurious_dets_outside_fci.append(det_new)

    if spurious_dets_outside_fci:
        print("\n!! CRITICAL ERROR: Generated determinants outside the FCI space:")
        for d in spurious_dets_outside_fci:
            print(f"  - {d}")
    
    #============================================================================



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

