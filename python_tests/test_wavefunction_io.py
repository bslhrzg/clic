import numpy as np
import clic_clib as qc

def test_slater_determinant_io():
    print("--- Testing SlaterDeterminant I/O ---")
    M = 4 # 4 spatial orbitals
    
    # |α₀, α₂, β₁, β₃⟩
    det = qc.SlaterDeterminant(M, occ_alpha0=[0, 2], occ_beta0=[1, 3])
    print(f"Initial determinant: {det}")

    # In AlphaFirst ordering: α₀,α₁,α₂,α₃, β₀,β₁,β₂,β₃
    # α₀ -> 0
    # α₂ -> 2
    # β₁ -> 1 + M = 5
    # β₃ -> 3 + M = 7
    expected_spin_orbitals = [0, 2, 5, 7]
    
    occupied_list = det.get_occupied_spin_orbitals()
    print(f"get_occupied_spin_orbitals() -> {occupied_list}")
    print(f"Expected list -> {expected_spin_orbitals}")
    
    assert occupied_list == expected_spin_orbitals
    print("✅ SlaterDeterminant.get_occupied_spin_orbitals() is correct.")


def test_wavefunction_constructor_and_getters():
    print("\n--- Testing Wavefunction Constructor and Getters ---")
    
    M = 3
    
    # 1. Define a basis and a set of coefficients
    det1 = qc.SlaterDeterminant(M, [0], [1])
    det2 = qc.SlaterDeterminant(M, [1], [2])
    det3 = qc.SlaterDeterminant(M, [0, 1], []) # Unsorted alpha
    
    # Note: the basis is intentionally unsorted here
    original_basis = [det2, det1, det3]
    original_amps = np.array([0.5, 0.2, 0.8], dtype=np.complex128)
    
    print("Original unsorted data:")
    for i in range(len(original_basis)):
        print(f"  {original_amps[i]} * {original_basis[i]}")

    # 2. Construct the Wavefunction object from the basis and amps
    wf = qc.Wavefunction(M, original_basis, original_amps)
    print(f"\nConstructed Wavefunction object: {wf}")
    assert len(wf.data()) == 3

    # 3. Use the new getter methods
    retrieved_basis = wf.get_basis()
    retrieved_amps = wf.get_amplitudes()
    
    print("\nData retrieved from Wavefunction object:")
    print("  get_basis():")
    for det in retrieved_basis:
        print(f"    {det}")
    print(f"  get_amplitudes(): {retrieved_amps}")

    # 4. Validate the results
    # The getters are guaranteed to return a sorted basis.
    # We must sort our original data to compare.
    
    # We can create a sorted list of (determinant, amplitude) pairs
    original_sorted_pairs = sorted(zip(original_basis, original_amps), key=lambda pair: pair[0])
    expected_sorted_basis = [pair[0] for pair in original_sorted_pairs]
    expected_sorted_amps = np.array([pair[1] for pair in original_sorted_pairs])
    
    print("\nExpected sorted data for comparison:")
    print("  Sorted basis:")
    for det in expected_sorted_basis:
        print(f"    {det}")
    print(f"  Sorted amplitudes: {expected_sorted_amps}")
    
    # Assert that the retrieved basis matches the expected sorted basis
    assert len(retrieved_basis) == len(expected_sorted_basis)
    for i in range(len(retrieved_basis)):
        assert retrieved_basis[i] == expected_sorted_basis[i]
    print("\n✅ Wavefunction.get_basis() is correct.")
        
    # Assert that the retrieved amplitudes match the expected sorted amplitudes
    np.testing.assert_allclose(retrieved_amps, expected_sorted_amps)
    print("✅ Wavefunction.get_amplitudes() is correct.")
    
    print("\nRe-constructing from retrieved data...")
    wf_reconstructed = qc.Wavefunction(M, retrieved_basis, retrieved_amps)
    print(f"Reconstructed wf: {wf_reconstructed}")
    assert wf.dot(wf_reconstructed) / np.sqrt(wf.dot(wf) * wf_reconstructed.dot(wf_reconstructed))
    print("✅ Reconstructed wavefunction is identical to the original.")


if __name__ == "__main__":
    test_slater_determinant_io()
    test_wavefunction_constructor_and_getters()