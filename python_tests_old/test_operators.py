import numpy as np
import clic_clib as qc

def test_single_determinant_operators():
    print("--- Testing Operators on a Single SlaterDeterminant ---")
    
    M = 3  # 3 spatial orbitals (α₀, α₁, α₂, β₀, β₁, β₂)
    
    # 1. Define an initial determinant: |α₀, β₁⟩
    det_initial = qc.SlaterDeterminant(M, occ_alpha0=[0], occ_beta0=[1])
    print(f"\nInitial determinant: {det_initial}")
    
    # --- Annihilation ---
    
    # 2a. Annihilate β₁: c_{1,β} |α₀, β₁⟩
    # Expected result: sign * |α₀⟩.
    # Sign rule: (-1)^{N_α} * (-1)^{N_β(<1)} = (-1)^1 * (-1)^0 = -1.
    print(f"\nApplying annihilation c_1β to {det_initial}...")
    result_ann = qc.SlaterDeterminant.annihilate(det_initial, i0=1, spin=qc.Spin.Beta)
    det_expected_ann = qc.SlaterDeterminant(M, occ_alpha0=[0], occ_beta0=[])
    
    print(f"  Result: {result_ann}")
    print(f"  Result determinant: {result_ann.det}")
    print(f"  Expected determinant: {det_expected_ann}")
    print(f"  Result sign: {result_ann.sign}, Expected sign: -1")
    assert result_ann.det == det_expected_ann
    assert result_ann.sign == -1
    print("  ✅ Annihilation test PASSED.")
    
    # 2b. Attempt to annihilate an empty orbital: c_{2,α} |α₀, β₁⟩
    # Expected result: None (Pauli exclusion)
    print(f"\nApplying annihilation c_2α (Pauli forbidden)...")
    result_pauli = qc.SlaterDeterminant.annihilate(det_initial, i0=2, spin=qc.Spin.Alpha)
    print(f"  Result: {result_pauli}, Expected: None")
    assert result_pauli is None
    print("  ✅ Pauli forbidden annihilation test PASSED.")

    # --- Creation ---
    
    # 3a. Create α₂: c†_{2,α} |α₀, β₁⟩
    # Expected result: sign * |α₀, α₂, β₁⟩
    # Sign rule: (-1)^{N_α(<2)} = (-1)^1 = -1.
    print(f"\nApplying creation c†_2α to {det_initial}...")
    result_cre = qc.SlaterDeterminant.create(det_initial, i0=2, spin=qc.Spin.Alpha)
    det_expected_cre = qc.SlaterDeterminant(M, occ_alpha0=[0, 2], occ_beta0=[1])
    
    print(f"  Result: {result_cre}")
    print(f"  Result determinant: {result_cre.det}")
    print(f"  Expected determinant: {det_expected_cre}")
    print(f"  Result sign: {result_cre.sign}, Expected sign: -1")
    assert result_cre.det == det_expected_cre
    assert result_cre.sign == -1
    print("  ✅ Creation test PASSED.")
    
    # 3b. Attempt to create an occupied orbital: c†_{0,α} |α₀, β₁⟩
    # Expected result: None (Pauli exclusion)
    print(f"\nApplying creation c†_0α (Pauli forbidden)...")
    result_pauli_cre = qc.SlaterDeterminant.create(det_initial, i0=0, spin=qc.Spin.Alpha)
    print(f"  Result: {result_pauli_cre}, Expected: None")
    assert result_pauli_cre is None
    print("  ✅ Pauli forbidden creation test PASSED.")

def test_wavefunction_operators():
    print("\n\n--- Testing Operators on a Wavefunction ---")
    
    M = 3
    coeff = 1.0 / np.sqrt(2.0)
    
    # 1. Define an initial wavefunction: Ψ = (1/√2)|α₀⟩ + (1/√2)|β₁⟩
    det1 = qc.SlaterDeterminant(M, [0], [])
    det2 = qc.SlaterDeterminant(M, [], [1])
    wf_initial = qc.Wavefunction(M)
    wf_initial.add_term(det1, coeff)
    wf_initial.add_term(det2, coeff)
    print(f"\nInitial wavefunction: {wf_initial}")
    print(f"  Data: {wf_initial.data()}")
    
    # --- Creation ---
    
    # 2. Apply creation operator c†_{0,β} to Ψ
    # c†_{0,β} |α₀⟩ -> -|α₀,β₀⟩
    # c†_{0,β} |β₁⟩ -> +|β₀,β₁⟩
    # Expected Ψ_final = (-1/√2)|α₀,β₀⟩ + (1/√2)|β₀,β₁⟩
    print(f"\nApplying creation c†_0β to wavefunction...")
    wf_final_cre = qc.apply_creation(wf_initial, i0=0, spin=qc.Spin.Beta)
    
    det_expected1 = qc.SlaterDeterminant(M, [0], [0])
    det_expected2 = qc.SlaterDeterminant(M, [], [0, 1])
    
    print(f"  Final wavefunction: {wf_final_cre}")
    data_cre = wf_final_cre.data()
    print(f"  Final data: {data_cre}")
    assert len(data_cre) == 2
    assert np.isclose(data_cre[det_expected1], -coeff)
    assert np.isclose(data_cre[det_expected2], coeff)
    print("  ✅ Wavefunction creation test PASSED.")
    
    # --- Annihilation ---
    
    # 3. Apply annihilation operator c_{0,α} to the original Ψ
    # c_{0,α} |α₀⟩ -> +|vac⟩
    # c_{0,α} |β₁⟩ -> 0
    # Expected Ψ_final = (1/√2)|vac⟩
    print(f"\nApplying annihilation c_0α to wavefunction...")
    wf_final_ann = qc.apply_annihilation(wf_initial, i0=0, spin=qc.Spin.Alpha)
    
    det_vac = qc.SlaterDeterminant(M, [], [])
    
    print(f"  Final wavefunction: {wf_final_ann}")
    data_ann = wf_final_ann.data()
    print(f"  Final data: {data_ann}")
    assert len(data_ann) == 1
    assert np.isclose(data_ann[det_vac], coeff)
    print("  ✅ Wavefunction annihilation test PASSED.")

if __name__ == "__main__":
    test_single_determinant_operators()
    test_wavefunction_operators()
