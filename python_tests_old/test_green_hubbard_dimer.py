import numpy as np
import numpy.linalg as npl
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.linalg import eig,eigh
import clic_clib as qc
import time
import matplotlib.pyplot as plt
from itertools import combinations

def create_hubbard_V(M, U_val):
    K = 2 * M
    V = np.zeros((K, K, K, K), dtype=np.complex128)
    for i in range(M):
        alpha_i = i
        beta_i  = i + M
        V[alpha_i, beta_i, alpha_i, beta_i] = U_val
        V[beta_i, alpha_i, beta_i, alpha_i] = U_val
    return V

def get_hubbard_dimer_ed_ref(t, U, M):
    K = 2 * M
    c_dag = [qc.get_creation_operator(K, i + 1) for i in range(K)]
    c = [qc.get_annihilation_operator(K, i + 1) for i in range(K)]
    mu = U / 2
    H = csr_matrix((2**K, 2**K), dtype=np.complex128)
    if M == 2:
        H += -t * (c_dag[0] @ c[1] + c_dag[1] @ c[0])
        H += -t * (c_dag[0+M] @ c[1+M] + c_dag[1+M] @ c[0+M])
    for i in range(M):
        n_up = c_dag[i] @ c[i]
        n_down = c_dag[i+M] @ c[i+M]
        H += U * (n_up @ n_down)
        H += - mu * (n_up + n_down)
    return H


# --- Helpers for Operator Terms ---
def get_one_body_terms(h1, M):
    terms = []
    for i in range(2*M):
        for j in range(2*M):
            if abs(h1[i, j]) > 1e-14:
                si = qc.Spin.Alpha if i < M else qc.Spin.Beta
                sj = qc.Spin.Alpha if j < M else qc.Spin.Beta
                oi, oj = (i % M), (j % M)
                terms.append((oi, oj, si, sj, complex(h1[i, j])))
    return terms

def get_two_body_terms(U, M):
    terms = []
    for i,j,k,l in np.argwhere(np.abs(U) > 1e-14):
        spins = [qc.Spin.Alpha if idx < M else qc.Spin.Beta for idx in [i,j,k,l]]
        orbs  = [idx % M for idx in [i,j,k,l]]
        terms.append((orbs[0],orbs[1],orbs[2],orbs[3],
                      spins[0],spins[1],spins[2],spins[3], complex(0.5*U[i,j,k,l])))
    return terms
    
# --- The Matrix-Free Hamiltonian Operator ---
def make_H_on_psi(one_body_terms, two_body_terms):
    """Factory function to create the H|psi> operator."""
    def H_on_psi(psi_in):
        psi_out1 = qc.apply_one_body_operator(psi_in, one_body_terms)
        psi_out2 = qc.apply_two_body_operator(psi_in, two_body_terms)
        return psi_out1 + psi_out2
    return H_on_psi

# --- Gram-Schmidt for Wavefunction objects (Corrected) ---
def gram_schmidt_qr(block_V):
    """Performs QR decomposition on a block of Wavefunction vectors."""
    Q = []
    num_vecs = len(block_V)
    if num_vecs == 0:
        return Q, np.array([[]])
    R = np.zeros((num_vecs, num_vecs), dtype=np.complex128)
    
    for j in range(num_vecs):
        vj = block_V[j]
        # Subtract projections onto previous orthonormal vectors
        # CORRECTED LOOP: Iterate over the current length of Q
        for i in range(len(Q)):
            qi = Q[i]
            R[i, j] = qi.dot(vj)
            vj = vj - R[i, j] * qi
        
        # Normalize
        norm_vj = np.sqrt(abs(vj.dot(vj)))
        if norm_vj > 1e-12:
            R[len(Q), j] = norm_vj  # Use len(Q) as the row index
            Q.append((1.0 / norm_vj) * vj)
    
    rank = len(Q)
    return Q, R[:rank, :]

# --- Block Lanczos with Wavefunction objects (Corrected) ---
def get_block_lanczos_wf(H_op, S_block, L, reorth=True):
    if not S_block: raise ValueError("Seed block is empty.")
    Q0_list, R0 = gram_schmidt_qr(S_block)
    if not Q0_list: return [], [], [], R0
    Qs, As, Bs = [Q0_list], [], []
    HQ0 = [H_op(q) for q in Q0_list]
    A0 = np.array([[qi.dot(hqj) for hqj in HQ0] for qi in Q0_list])
    As.append(A0)
    Q_prev_list = []
    for k in range(L):
        Qk_list, Ak = Qs[-1], As[-1]
        zero_wf = qc.Wavefunction(Qk_list[0].n_spatial)
        W_list = [H_op(Qk_list[j]) - sum((Ak[i, j] * Qk_list[i] for i in range(len(Qk_list))), start=zero_wf) for j in range(len(Qk_list))]
        if k > 0 and Q_prev_list: # Check if Q_prev exists
            Bk_dagger = Bs[-1].conj().T
            W_list = [W_list[j] - sum((Bk_dagger[i, j] * Q_prev_list[i] for i in range(len(Q_prev_list))), start=zero_wf) for j in range(len(W_list))]
        if reorth:
            for Qj_list in Qs:
                # CORRECTED LOOP: Iterate over a copy of W_list
                W_list_new = []
                for w_vec in W_list:
                    for qj_vec in Qj_list:
                        proj = qj_vec.dot(w_vec)
                        w_vec = w_vec - proj * qj_vec
                    W_list_new.append(w_vec)
                W_list = W_list_new

        Q_next_list, B_next = gram_schmidt_qr(W_list)
        if not Q_next_list: break # This is the crucial breakdown check
        Bs.append(B_next)
        Qs.append(Q_next_list)
        HQ_next = [H_op(q) for q in Q_next_list]
        A_next = np.array([[qi.dot(hqj) for hqj in HQ_next] for qi in Q_next_list])
        As.append(A_next)
        Q_prev_list = Qk_list
        if len(Qs) >= L + 1: break
    return As, Bs, Qs, R0

def block_cf_top_left(As, Bs, z):
    if not As: return np.array([[]])
    m0 = As[0].shape[0]
    Id = np.eye(m0, dtype=complex)
    Sigma = np.zeros_like(As[-1], dtype=complex)
    for k in range(len(As)-2, -1, -1):
        Bkp1 = Bs[k]
        Ikp1 = np.eye(As[k+1].shape[0], dtype=complex)
        try:
            Sigma = Bkp1.conj().T @ npl.inv(z*Ikp1 - As[k+1] - Sigma) @ Bkp1
        except npl.LinAlgError:
            return np.full_like(Id, np.nan) # Return NaN on singular matrix
    return npl.inv(z*Id - As[0] - Sigma)

def green_function_block_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta):
    Norb = 2 * M
    S_add = [qc.apply_creation(psi0_wf, i % M, qc.Spin.Alpha if i < M else qc.Spin.Beta) for i in range(Norb)]
    S_rem = [qc.apply_annihilation(psi0_wf, i % M, qc.Spin.Alpha if i < M else qc.Spin.Beta) for i in range(Norb)]
    S_add = [wf for wf in S_add if wf.data()]
    S_rem = [wf for wf in S_rem if wf.data()]
    As_g, Bs_g, Qs_g, R0_g = get_block_lanczos_wf(H_op, S_add, L) if S_add else ([],[],[],np.array([]))
    As_l, Bs_l, Qs_l, R0_l = get_block_lanczos_wf(H_op, S_rem, L) if S_rem else ([],[],[],np.array([]))
    Nw = len(ws)
    G_all = np.zeros((Nw, Norb, Norb), dtype=np.complex128)
    have_g, have_l = (R0_g.size != 0 and len(As_g) > 0), (R0_l.size != 0 and len(As_l) > 0)
    for iw, w in enumerate(ws):
        z_g, z_l = (w + e0) + 1j*eta, (-w + e0) - 1j*eta
        Gg, Gl = np.zeros((Norb, Norb), dtype=complex), np.zeros((Norb, Norb), dtype=complex)
        if have_g:
            G00 = block_cf_top_left(As_g, Bs_g, z_g)
            Gg = R0_g.conj().T @ G00 @ R0_g
        if have_l:
            G00 = block_cf_top_left(As_l, Bs_l, z_l)
            Gl = R0_l.conj().T @ G00 @ R0_l
        G_all[iw, :, :] = Gg - Gl
    return G_all

def get_fci_basis(num_spatial, num_electrons):
    num_spin_orbitals = 2 * num_spatial
    basis_dets = []
    for occupied_indices in combinations(range(num_spin_orbitals), num_electrons):
        occ_a = [i for i in occupied_indices if i < num_spatial]
        occ_b = [i - num_spatial for i in occupied_indices if i >= num_spatial]
        det = qc.SlaterDeterminant(num_spatial, occ_a, occ_b)
        basis_dets.append(det)
    return sorted(basis_dets)

# --- Main Test Execution ---
if __name__ == "__main__":
 
    print("--- Final Test: Hubbard Dimer (N=2, full subspace) ---")
    
    # Model parameters
    M = 2
    K = 2 * M
    Nelec = 2
    t = 1.0
    U = 1.0
    mu = U/2.0

    # --- Part 1: ED Tools Reference ---
    H_ed_full = get_hubbard_dimer_ed_ref(t, U, M)
    states_2e_indices = [i for i in range(2**K) if bin(i).count('1') == Nelec]
    H_ed_2e = H_ed_full[np.ix_(states_2e_indices, states_2e_indices)]
    eigvals_ed, _ = eigh(H_ed_2e.toarray())
    print(f"\nED tools (N=2, U={U}) eigenvalues:\n{np.round(np.sort(eigvals_ed), 8)}")

    # --- Part 2: Slater-Condon Builder ---
    basis = get_fci_basis(M, Nelec)
    
    h0 = np.zeros((K, K), dtype=np.complex128)
    h0[0, 1] = h0[1, 0] = -t
    h0[2, 3] = h0[3, 2] = -t
    for i in range(K):
        h0[i,i] = -mu


    V = create_hubbard_V(M, U) # This gives 2U

    print("\nBuilding Hamiltonian with Slater-Condon rules...")
    H_openmp = qc.build_hamiltonian_openmp(basis, h0, V)
    eigvals, eigvecs = eigh(H_openmp.toarray())
    print(f"OpenMP builder (U={U}) eigenvalues:\n{np.round(np.sort(eigvals), 8)}")
    
    # Final check
    #np.testing.assert_allclose(np.sort(eigvals_ed), np.sort(eigvals), atol=1e-12)
    #print("\n✅ SUCCESS: OpenMP builder matches ED tools.")

    # --- Find Ground State ---
    e0 = eigvals[0]
    psi0_wf = qc.Wavefunction(M, basis, eigvecs[:, 0])
    
    print(f"Ground State Energy: {e0:.6f}")

    # --- Matrix-Free Lanczos Calculation ---
    one_body_terms = get_one_body_terms(h0, M)
    two_body_terms = get_two_body_terms(V, M)
    H_op = make_H_on_psi(one_body_terms, two_body_terms)
    
    ws = np.linspace(-6, 6, 1001)
    eta = 0.02
    L = 50
    
    print("\nRunning matrix-free block Lanczos for the impurity Green's function...")
    t_start = time.time()
    G_mat_lanc_wf = green_function_block_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta)
    t_end = time.time()
    print(f"  Calculation finished in {t_end-t_start:.2f}s")
    
    A_mat_lanc_wf = -(1/np.pi) * np.imag(G_mat_lanc_wf)
    
    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    for i in range(K):
        plt.plot(ws, A_mat_lanc_wf[:, i, i], label="A_"+str(i)+"(ω)")
    plt.title("Spectral Function for Hubbard Dimer (Matrix-Free Lanczos)")
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nScript finished. Check the plot for the spectral function.")