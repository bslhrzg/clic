import numpy as np
import clic_clib as qc

def test_large_non_interacting_system():
    print("--- Testing Large Non-Interacting System (1D Tight-Binding) ---")
    
    # --- 1. System Setup ---
    M = 50      # 50 spatial sites
    K = 2 * M   # 100 spin-orbitals
    Ne = 50     # 50 electrons (half-filling)
    t = 1.0     # Hopping parameter
    
    print(f"System: M={M}, K={K}, Ne={Ne}")

    # --- 2. Hamiltonian in Site Basis ---
    # 1D tight-binding model for spatial orbitals
    h_spatial_site = np.zeros((M, M))
    for i in range(M - 1):
        h_spatial_site[i, i+1] = h_spatial_site[i+1, i] = -t
    
    # Expand to spin-orbitals in AlphaFirst basis
    h0_site = np.zeros((K, K))
    h0_site[0:M, 0:M] = h_spatial_site
    h0_site[M:K, M:K] = h_spatial_site
    
    # No two-body interaction
    U_site = np.zeros((K, K, K, K))

    # --- 3. Find Molecular Orbital (MO) Basis ---
    # The MO basis diagonalizes the one-particle Hamiltonian
    eps, C_spatial = np.linalg.eigh(h_spatial_site)
    
    # Expand the MO transformation matrix C to the spin-orbital basis
    C = np.zeros((K, K))
    C[0:M, 0:M] = C_spatial
    C[M:K, M:K] = C_spatial

    # --- 4. Define Fermi Sea (HF state) in MO Basis ---
    # The ground state occupies the Ne/2 = 25 lowest-energy MOs
    n_occ_pairs = Ne // 2
    occupied_alpha_mo = list(range(n_occ_pairs)) # MO indices 0, 1, ..., 24
    occupied_beta_mo = list(range(n_occ_pairs))  # MO indices 0, 1, ..., 24
    
    fermi_sea = qc.SlaterDeterminant(M, occupied_alpha_mo, occupied_beta_mo)
    fermi_sea_occ_list = fermi_sea.get_occupied_spin_orbitals()
    print(f"\nFermi sea determinant defined by occupying the {n_occ_pairs} lowest MOs for each spin.")

    # --- 5. Transform Integrals to MO Basis ---
    # h_mo = C.T @ h_site @ C
    h0_mo = C.T @ h0_site @ C
    # U_mo is still zero
    U_mo = np.zeros_like(U_site)
    
    # Ensure arrays are clean before passing to C++
    h0_mo_clean = np.ascontiguousarray(h0_mo, dtype=np.complex128)
    U_mo_clean = np.ascontiguousarray(U_mo, dtype=np.complex128)

    # --- 6. Test 1: Diagonal Energy of the Fermi Sea ---
    print("\n--- Test 1: Ground State Energy ---")
    
    # Exact energy is the sum of the energies of occupied MOs
    exact_energy = 2 * np.sum(eps[:n_occ_pairs])
    print(f"Exact electronic energy (sum of eigenvalues) = {exact_energy:.8f}")
    
    # Calculate energy using the C++ OO kernel
    hf_energy_cpp = qc.KL(fermi_sea_occ_list, fermi_sea_occ_list, K, h0_mo_clean, U_mo_clean)
    print(f"C++ KL⟨HF|H|HF⟩                             = {np.real(hf_energy_cpp):.8f}")
    
    np.testing.assert_allclose(exact_energy, hf_energy_cpp, atol=1e-9)
    print("✅ SUCCESS: Diagonal energy matches the sum of eigenvalues.")

    # --- 7. Test 2: Single Excitation Matrix Element ---
    print("\n--- Test 2: Single Excitation Matrix Element ---")
    
    # Create a singly-excited determinant |S⟩ = c†_j c_i |HF⟩
    # Excite from the highest occupied (HOMO) to the lowest unoccupied (LUMO)
    homo_idx = n_occ_pairs - 1  # Occupied MO index 24
    lumo_idx = n_occ_pairs       # Unoccupied MO index 25
    
    # We'll excite an alpha electron
    s_alpha_occ = occupied_alpha_mo.copy()
    s_alpha_occ.remove(homo_idx)
    s_alpha_occ.append(lumo_idx)
    
    single_det = qc.SlaterDeterminant(M, s_alpha_occ, occupied_beta_mo)
    single_det_occ_list = single_det.get_occupied_spin_orbitals()
    
    print(f"Testing element ⟨S|H|HF⟩ for excitation MO {homo_idx}α -> {lumo_idx}α")
    
    # For a non-interacting H in the eigenbasis, this element must be zero.
    # h_mo is diagonal, so h_mo[i, j] = 0 for i != j
    expected_os_val = 0.0
    print(f"Expected value (since h_mo is diagonal) = {expected_os_val}")
    
    # Calculate with C++ OS kernel
    os_val_cpp = qc.KL(single_det_occ_list, fermi_sea_occ_list, K, h0_mo_clean, U_mo_clean)
    print(f"C++ KL⟨S|H|HF⟩                          = {np.real(os_val_cpp):.8f}")
    
    np.testing.assert_allclose(expected_os_val, os_val_cpp, atol=1e-9)
    print("✅ SUCCESS: Single excitation matrix element is zero, as expected.")

    # --- 8. Test 3: Double Excitation Matrix Element ---
    print("\n--- Test 3: Double Excitation Matrix Element ---")
    
    # Create a doubly-excited determinant |D⟩ = c†_k c†_l c_j c_i |HF⟩
    homo_minus_1_idx = n_occ_pairs - 2 # MO index 23
    lumo_plus_1_idx  = n_occ_pairs + 1 # MO index 26
    
    d_alpha_occ = occupied_alpha_mo.copy()
    d_alpha_occ.remove(homo_idx)
    d_alpha_occ.remove(homo_minus_1_idx)
    d_alpha_occ.append(lumo_idx)
    d_alpha_occ.append(lumo_plus_1_idx)
    
    double_det = qc.SlaterDeterminant(M, d_alpha_occ, occupied_beta_mo)
    double_det_occ_list = double_det.get_occupied_spin_orbitals()
    
    print(f"Testing element ⟨D|H|HF⟩ for a double excitation.")
    
    # For a one-body Hamiltonian, this must be zero by Slater-Condon rules.
    expected_od_val = 0.0
    print(f"Expected value (one-body H) = {expected_od_val}")
    
    # Calculate with C++ KL dispatcher
    od_val_cpp = qc.KL(double_det_occ_list, fermi_sea_occ_list, K, h0_mo_clean, U_mo_clean)
    print(f"C++ KL⟨D|H|HF⟩                  = {np.real(od_val_cpp):.8f}")
    
    np.testing.assert_allclose(expected_od_val, od_val_cpp, atol=1e-9)
    print("✅ SUCCESS: Double excitation matrix element is zero, as expected.")


if __name__ == "__main__":
    test_large_non_interacting_system()