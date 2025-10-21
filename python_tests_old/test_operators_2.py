import numpy as np
from itertools import combinations
from scipy.linalg import eigh

import clic_clib as qc

# --- Helper to create sparse operator terms from dense Numpy arrays ---
def get_one_body_terms(h1_matrix):
    """Converts a dense (K,K) numpy array to a sparse list of tuples for C++."""
    terms = []
    M = h1_matrix.shape[0] // 2
    for i in range(2 * M):
        for j in range(2 * M):
            if abs(h1_matrix[i, j]) > 1e-12:
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
                    if abs(v2_tensor[i, j, k, l]) > 1e-12:
                        spins = [qc.Spin.Alpha if idx < M else qc.Spin.Beta for idx in [i, j, k, l]]
                        orbs = [idx if idx < M else idx - M for idx in [i, j, k, l]]
                        terms.append((orbs[0], orbs[1], orbs[2], orbs[3],
              spins[0], spins[1], spins[2], spins[3],
              complex(v2_tensor[i, j, k, l])))
    return terms

def test_basis_generation():
    print("--- Testing Basis Closure ---")
    
    # --- Setup ---
    M = 2
    t = 1.0
    U = 4.0
    
    K = 2 * M
    H1 = np.zeros((K, K), dtype=np.complex128)
    H1[0, 1] = H1[1, 0] = -t
    H1[2, 3] = H1[3, 2] = -t
    
    V = np.zeros((K, K, K, K), dtype=np.complex128)
    # V_pqrs = <pr|V|qs>
    V[0, 2, 0, 2] = 2.0 * U # <0α,0β|V|0α,0β>
    V[1, 3, 1, 3] = 2.0 * U # <1α,1β|V|1α,1β>
    
    one_body_terms = get_one_body_terms(H1)
    two_body_terms = get_two_body_terms(V)

    # --- 1. Define the complete Sz=0 basis for N=2, M=2 ---
    # This basis is closed under the Hubbard Hamiltonian.
    det1 = qc.SlaterDeterminant(M, [0], [0]) # |α₀, β₀>
    det2 = qc.SlaterDeterminant(M, [0], [1]) # |α₀, β₁>
    det3 = qc.SlaterDeterminant(M, [1], [0]) # |α₁, β₀>
    det4 = qc.SlaterDeterminant(M, [1], [1]) # |α₁, β₁>
    
    sz0_basis = sorted([det1, det2, det3, det4])
    sz0_basis_set = set(sz0_basis)
    
    print(f"Starting with the full Sz=0 basis (size = {len(sz0_basis)}):")
    for det in sz0_basis:
        print(f"  {det}")

    # --- 2. Apply H and check for closure ---
    # Apply H to the entire basis and find all unique connected determinants
    connected_by_H1 = qc.get_connections_one_body(sz0_basis, one_body_terms)
    connected_by_V2 = qc.get_connections_two_body(sz0_basis, two_body_terms)
    
    # The set of all generated determinants
    all_connected_set = set(connected_by_H1 + connected_by_V2)
    
    print(f"\nApplying H generated a basis of size {len(all_connected_set)}.")
    
    # Check if the generated set is a subset of (or equal to) the original set
    is_subset = all_connected_set.issubset(sz0_basis_set)
    
    print(f"Is the generated basis a subset of the original basis? {is_subset}")
    assert is_subset
    print("  ✅ The Sz=0 basis is closed under the Hamiltonian action, as expected.")

def test_density_matrix():
    print("\n--- Testing One-Particle Density Matrix (1-RDM) ---")
    
    # --- Setup ---
    M = 2
    K = 2 * M
    Nelec = 2
    t = 1.0
    U = 4.0
    
    # Get the ground state wavefunction by diagonalizing the full Hamiltonian
    fci_basis = []
    for occ_indices in combinations(range(K), Nelec):
        occ_a = [i for i in occ_indices if i < M]
        occ_b = [i - M for i in occ_indices if i >= M]
        fci_basis.append(qc.SlaterDeterminant(M, occ_a, occ_b))
    fci_basis = sorted(fci_basis)
    
    H1 = np.zeros((K, K), dtype=np.complex128)
    H1[0, 1] = H1[1, 0] = -t
    H1[2, 3] = H1[3, 2] = -t
    V = np.zeros((K, K, K, K), dtype=np.complex128)
    V[0, 2, 0, 2] = 2.0 * U
    V[1, 3, 1, 3] = 2.0 * U
    H_mat = qc.build_hamiltonian_openmp(fci_basis, H1, V).toarray()
    
    eigvals, eigvecs = eigh(H_mat)
    gs_coeffs = eigvecs[:, 0]
    
    gs_wf = qc.Wavefunction(M)
    for i, det in enumerate(fci_basis):
        gs_wf.add_term(det, gs_coeffs[i])
        
    print(f"Ground state energy: {eigvals[0]:.6f}")
    print(f"Ground state wavefunction has {len(gs_wf.data())} terms.")

    # --- 1. Compute 1-RDM using new dynamic operators ---
    print("\nComputing 1-RDM using dynamic apply_one_body_operator...")
    rdm_dynamic = np.zeros((K, K), dtype=np.complex128)
    for i in range(K):
        for j in range(K):
            # Create the operator term c†_i c_j
            spin_i = qc.Spin.Alpha if i < M else qc.Spin.Beta
            spin_j = qc.Spin.Alpha if j < M else qc.Spin.Beta
            orb_i = i if i < M else i - M
            orb_j = j if j < M else j - M
            
            # The operator term is a list containing a single tuple (h_ij = 1.0)
            op_term = [(orb_i, orb_j, spin_i, spin_j, 1.0)]
            
            # Apply the operator c†_i c_j to the ground state
            # This creates the state |Φ⟩ = c†_i c_j |Ψ⟩
            phi_wf = qc.apply_one_body_operator(gs_wf, op_term)
            
            # The RDM element is <Ψ|Φ>
            rdm_dynamic[i, j] = gs_wf.dot(phi_wf)

    print("1-RDM from dynamic operators:")
    print(np.real(rdm_dynamic))

    # --- 2. Compute 1-RDM using ED tools for validation ---
    print("\nComputing 1-RDM using ED tools for validation...")
    rdm_ed = np.zeros((K, K), dtype=np.complex128)
    
    gs_vec_fock = np.zeros(2**K, dtype=np.complex128)
    fock_map = {}
    for i, det in enumerate(fci_basis):
        alpha_bits = sum(1 << bit for bit in det.alpha_occupied_indices())
        beta_bits = sum(1 << (bit + M) for bit in det.beta_occupied_indices())
        fock_map[i] = alpha_bits | beta_bits

    for i, coeff in enumerate(gs_coeffs):
        gs_vec_fock[fock_map[i]] = coeff

    c_dag = [qc.get_creation_operator(K, k + 1) for k in range(K)]
    c = [qc.get_annihilation_operator(K, k + 1) for k in range(K)]
    
    for i in range(K):
        for j in range(K):
            op = c_dag[i] @ c[j]
            rdm_ed[i, j] = gs_vec_fock.conj().T @ op @ gs_vec_fock

    print("1-RDM from ED tools:")
    print(np.real(rdm_ed))

    # --- 3. Final Comparison ---
    np.testing.assert_allclose(rdm_dynamic, rdm_ed, atol=1e-12)
    print("\n✅ SUCCESS: 1-RDM from dynamic operators matches ED tools reference.")

if __name__ == "__main__":
    test_basis_generation()
    test_density_matrix()
