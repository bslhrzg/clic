import numpy as np
import scipy.sparse
from scipy.linalg import eigh
from itertools import combinations
import clic_clib as qc

def get_fci_basis(num_spatial, num_electrons):
    num_spin_orbitals = 2 * num_spatial
    basis_dets = []
    for occupied_indices in combinations(range(num_spin_orbitals), num_electrons):
        occ_a = [i for i in occupied_indices if i < num_spatial]
        occ_b = [i - num_spatial for i in occupied_indices if i >= num_spatial]
        det = qc.SlaterDeterminant(num_spatial, occ_a, occ_b)
        basis_dets.append(det)
    return sorted(basis_dets)

def create_hubbard_V(M, U_val):
    K = 2 * M
    V = np.zeros((K, K, K, K), dtype=np.complex128)
    for i in range(M):
        alpha_i = i
        beta_i  = i + M
        # To get a final energy of U from H = 1/2 sum <pq|V|rs>...,
        # the integral value <iα,iβ|V|iα,iβ> must be 2*U.
        V[alpha_i, beta_i, alpha_i, beta_i] = 2.0 * U_val
    return V

def get_hubbard_dimer_ed_ref(t, U, M):
    K = 2 * M
    c_dag = [qc.get_creation_operator(K, i + 1) for i in range(K)]
    c = [qc.get_annihilation_operator(K, i + 1) for i in range(K)]
    H = scipy.sparse.csr_matrix((2**K, 2**K), dtype=np.complex128)
    if M == 2:
        H += -t * (c_dag[0] @ c[1] + c_dag[1] @ c[0])
        H += -t * (c_dag[0+M] @ c[1+M] + c_dag[1+M] @ c[0+M])
    for i in range(M):
        n_up = c_dag[i] @ c[i]
        n_down = c_dag[i+M] @ c[i+M]
        H += U * (n_up @ n_down)
    return H

def test_hubbard_comparison():
    print("--- Final Test: Hubbard Dimer (N=2, full subspace) ---")
    
    # Model parameters
    M = 2
    K = 2 * M
    Nelec = 2
    t = 1.0
    U = 1.0

    # --- Part 1: ED Tools Reference ---
    H_ed_full = get_hubbard_dimer_ed_ref(t, U, M)
    states_2e_indices = [i for i in range(2**K) if bin(i).count('1') == Nelec]
    H_ed_2e = H_ed_full[np.ix_(states_2e_indices, states_2e_indices)]
    eigvals_ed, _ = eigh(H_ed_2e.toarray())
    print(f"\nED tools (N=2, U={U}) eigenvalues:\n{np.round(np.sort(eigvals_ed), 8)}")

    # --- Part 2: Slater-Condon Builder ---
    basis = get_fci_basis(M, Nelec)
    
    H1 = np.zeros((K, K), dtype=np.complex128)
    H1[0, 1] = H1[1, 0] = -t
    H1[2, 3] = H1[3, 2] = -t


    V = create_hubbard_V(M, U)

    print("\nBuilding Hamiltonian with Slater-Condon rules...")
    H_openmp = qc.build_hamiltonian_openmp(basis, H1, V)
    eigvals_openmp, _ = eigh(H_openmp.toarray())
    print(f"OpenMP builder (U={U}) eigenvalues:\n{np.round(np.sort(eigvals_openmp), 8)}")
    
    # Final check
    np.testing.assert_allclose(np.sort(eigvals_ed), np.sort(eigvals_openmp), atol=1e-12)
    print("\n✅ SUCCESS: OpenMP builder matches ED tools.")

if __name__ == "__main__":
    test_hubbard_comparison()
