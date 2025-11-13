import numpy as np
import numpy.linalg as npl
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.linalg import eig,eigh
from clic import * 
from clic_clib import *

import time
import matplotlib.pyplot as plt



def get_random_model_and_ED(U, M):
    K = 2 * M
    c_dag = [get_creation_operator(K, i + 1) for i in range(K)]
    c = [get_annihilation_operator(K, i + 1) for i in range(K)]

    mu = U / 2

    H = csr_matrix((2**K, 2**K), dtype=np.complex128)


    h0 = get_linear_h0(M,1,True)
    h0 = double_h(h0,M) * (1 + 0.0 *1j)

    diag_scale = 1e-1
    offdiag_scale = 1e-1
    imag_scale = 1e-1

    h0 += diag_scale * (1+0.0j)* np.diag(np.random.random(K))
    h0 += offdiag_scale * np.random.random((K,K)) 
    h0 += offdiag_scale * imag_scale * 1j * np.random.random((K,K))
    h0 = h0 + h0.conj().T 

    for i in range(K):
        h0[i,i] -= mu

    for i in range(K):
        for j in range(K):
            H += h0[i,j] * c_dag[i] @ c[j]

    print("h0 = ")
    print(np.round(h0,4))
   
    for i in range(M):
        n_up = c_dag[i] @ c[i]
        n_down = c_dag[i+M] @ c[i+M]
        H += U * (n_up @ n_down)



    return H,h0,c,c_dag

def get_exact_green(H_, c, c_dag, ws, eta, beta=None):
    """
    Exact finite temperature Green function from Lehmann representation.

    Parameters
    ----------
    H_ : (N,N) array_like or sparse
        Many body Hamiltonian (Hermitian).
    c, c_dag : list of sparse/dense (N,N) operators
        Annihilation and creation operators for each orbital.
        len(c) = len(c_dag) = Norb
    ws : array_like
        Frequency grid (real).
    eta : float
        Positive broadening.
    beta : float or None
        Inverse temperature. If None or np.inf, use T=0 with the ground state only.

    Returns
    -------
    G : ndarray, shape (Nw, Norb, Norb)
        Retarded Green function G_ab(ω) at each frequency.
    Z : float
        Partition function.
    """

    # Dense Hamiltonian
    H = H_.toarray() if hasattr(H_, "toarray") else np.array(H_, dtype=complex)

    # Full diagonalization H = U diag(E) U^†
    es, U = eigh(H)
    es = es.real
    dim = H.shape[0]
    norb = len(c)
    ws = np.asarray(ws, dtype=float)
    n_ω = len(ws)

    # Transform operators to eigenbasis: O_eig = U^† O U
    c_eig = []
    cd_eig = []
    Udag = U.conj().T
    for a in range(norb):
        ci = c[a].toarray() if hasattr(c[a], "toarray") else np.array(c[a])
        cdag_i = c_dag[a].toarray() if hasattr(c_dag[a], "toarray") else np.array(c_dag[a])
        c_eig.append(Udag @ ci @ U)
        cd_eig.append(Udag @ cdag_i @ U)

    # Boltzmann factors
    if beta is None or np.isinf(beta):
        # T = 0 limit: only the ground state contributes
        boltz = np.zeros_like(es)
        boltz[0] = 1.0
        Z = 1.0
    else:
        # Shift by E_min to improve numerical stability
        E0 = es.min()
        b_unnorm = np.exp(-beta * (es - E0))
        Z = b_unnorm.sum()
        boltz = b_unnorm / Z

    # Precompute energy differences ΔE_{n m} = E_n - E_m
    En = es
    dE = En[:, None] - En[None, :]                   # shape (dim, dim)
    boltz_mat = boltz[:, None]                       # shape (dim, 1), broadcast over m

    G = np.zeros((n_ω, norb, norb), dtype=complex)

    for a in range(norb):
        Ci = c_eig[a]
        for b in range(norb):
            Cbd = cd_eig[b]

            # term1_nm = <n| c_a |m><m| c_b^† |n> = Ci[n,m] * Cbd[m,n]
            term1 = Ci * Cbd.T                        # elementwise

            # term2_nm = <n| c_b^† |m><m| c_a |n> = Cbd[n,m] * Ci[m,n]
            term2 = Cbd * Ci.T

            for iw, w in enumerate(ws):
                z = w + 1j * eta

                denom1 = z + dE                       # ω + iη + E_n − E_m
                denom2 = z - dE                       # ω + iη − E_n + E_m

                G[iw, a, b] = np.sum(
                    boltz_mat * (term1 / denom1 + term2 / denom2)
                )

    return G, Z

# --- Helpers for Operator Terms ---
def get_one_body_terms(h1, M):
    terms = []
    for i in range(2*M):
        for j in range(2*M):
            if abs(h1[i, j]) > 1e-14:
                si = Spin.Alpha if i < M else Spin.Beta
                sj = Spin.Alpha if j < M else Spin.Beta
                oi, oj = (i % M), (j % M)
                terms.append((oi, oj, si, sj, complex(h1[i, j])))
    return terms

def get_two_body_terms(U, M):
    terms = []
    for i,j,k,l in np.argwhere(np.abs(U) > 1e-14):
        spins = [Spin.Alpha if idx < M else Spin.Beta for idx in [i,j,k,l]]
        orbs  = [idx % M for idx in [i,j,k,l]]
        terms.append((orbs[0],orbs[1],orbs[2],orbs[3],
                      spins[0],spins[1],spins[2],spins[3], complex(0.5*U[i,j,k,l])))
    return terms
    
# --- The Matrix-Free Hamiltonian Operator ---
def make_H_on_psi(one_body_terms, two_body_terms):
    """Factory function to create the H|psi> operator."""
    def H_on_psi(psi_in):
        psi_out1 = apply_one_body_operator(psi_in, one_body_terms)
        psi_out2 = apply_two_body_operator(psi_in, two_body_terms)
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
        zero_wf = Wavefunction(Qk_list[0].n_spatial)
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
    S_add = [apply_creation(psi0_wf, i % M, Spin.Alpha if i < M else Spin.Beta) for i in range(Norb)]
    S_rem = [apply_annihilation(psi0_wf, i % M, Spin.Alpha if i < M else Spin.Beta) for i in range(Norb)]
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


#----
def plot_all_components_grid(omega, A_1, A_2, title_prefix, out_prefix):
    Nw, M, _ = A_1.shape
    # real parts
    fig_re, axes_re = plt.subplots(M, M, figsize=(3*M, 2.2*M), squeeze=False)
    for i in range(M):
        for j in range(M):
            ax = axes_re[i, j]
            ax.plot(omega, np.real(A_1[:, i, j]), label="Re 1", lw=1.5)
            ax.plot(omega, np.real(A_2[:, i, j]), label="Re 2", lw=1.0, ls="--")
            if i == M-1: ax.set_xlabel("ω")
            if j == 0:   ax.set_ylabel(f"{i},{j}")
            ax.tick_params(axis="both", labelsize=8)
    axes_re[0,0].legend(loc="upper right", fontsize=8)
    fig_re.suptitle(f"{title_prefix} real parts", fontsize=12)
    fig_re.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    #fig_re.savefig(f"{out_prefix}_real.png", dpi=150)
    plt.close(fig_re)
    # imaginary parts
    fig_im, axes_im = plt.subplots(M, M, figsize=(3*M, 2.2*M), squeeze=False)
    for i in range(M):
        for j in range(M):
            ax = axes_im[i, j]
            ax.plot(omega, np.imag(A_1[:, i, j]), label="Im 1", lw=1.5)
            ax.plot(omega, np.imag(A_2[:, i, j]), label="Im 2", lw=1.0, ls="--")
            if i == M-1: ax.set_xlabel("ω")
            if j == 0:   ax.set_ylabel(f"{i},{j}")
            ax.tick_params(axis="both", labelsize=8)
    axes_im[0,0].legend(loc="upper right", fontsize=8)
    fig_im.suptitle(f"{title_prefix} imaginary parts", fontsize=12)
    fig_im.tight_layout(rect=[0, 0, 1, 0.97])
    #fig_im.savefig(f"{out_prefix}_imag.png", dpi=150)
    plt.show()
    plt.close(fig_im)

# --- Main Test Execution ---
if __name__ == "__main__":
 
    print("--- Final Test: Hubbard Dimer (N=2, full subspace) ---")
    
    # Model parameters
    M = 2
    K = 2 * M
    Nelec = 2
    t = 1.0
    U = 2.0

    # --- Part 1: ED Tools Reference ---
    H_ed_full,h0,c,c_dag = get_random_model_and_ED(U, M)
    efull ,_= eigh(H_ed_full.toarray())
    print(len(efull))
    print(f"\nED tools (full, U={U}) eigenvalues:\n{np.round(np.sort(efull), 4)}")

    states_2e_indices = [i for i in range(2**K) if bin(i).count('1') == Nelec]
    H_ed_2e = H_ed_full[np.ix_(states_2e_indices, states_2e_indices)]
    eigvals_ed, _ = eigh(H_ed_2e.toarray())
    print(f"\nED tools (N=2, U={U}) eigenvalues:\n{np.round(np.sort(eigvals_ed), 4)}")

    # --- Part 2: Slater-Condon Builder ---
    basis = get_fci_basis(M, Nelec)
    
    #h0 = np.zeros((K, K), dtype=np.complex128)
    #h0[0, 1] = h0[1, 0] = -t
    #h0[2, 3] = h0[3, 2] = -t
    #for i in range(K):
    #    h0[i,i] = -mu


    V = create_hubbard_V(M, U) # This gives 2U

    print("\nBuilding Hamiltonian with Slater-Condon rules...")
    H_openmp = build_hamiltonian_openmp(basis, h0, V, True)
    eigvals, eigvecs = eigh(H_openmp.toarray())
    print(f"OpenMP builder (U={U}) eigenvalues:\n{np.round(np.sort(eigvals), 4)}")
    
    # Final check
    #np.testing.assert_allclose(np.sort(eigvals_ed), np.sort(eigvals), atol=1e-12)
    #print("\n✅ SUCCESS: OpenMP builder matches ED tools.")

    # --- Find Ground State ---
    e0 = eigvals[0]
    psi0_wf = Wavefunction(M, basis, eigvecs[:, 0])
    
    print(f"Ground State Energy: {e0:.6f}")

    # --- Matrix-Free Lanczos Calculation ---
    one_body_terms = get_one_body_terms(h0, M)
    two_body_terms = get_two_body_terms(V, M)
    H_op = make_H_on_psi(one_body_terms, two_body_terms)
    
    ws = np.linspace(-6, 6, 1001)
    eta = 0.02
    L = 150
    
    print("\nRunning matrix-free block Lanczos for the impurity Green's function...")
    t_start = time.time()
    #G_mat_lanc_wf = green_function_block_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta)
    #G_mat_lanc_wf,_ =  green_function_block_lanczos_fixed_basis(
    #                M, psi0_wf, e0, ws, eta, [i for i in range(2*M)], 2,
    #                h0, V, one_body_terms, two_body_terms, coeff_thresh=1e-12, L=100, reorth=False
    #            )
    
    target_indices = [i for i in range(2*M)]
    gfmeth = "scalar_continued_fraction"
    gfmeth = "block"
    gfmeth = "time_prop"
    coeff_thresh=1e-12
    nappH = 2
    G_mat_lanc_wf = get_green_block(M, psi0_wf, e0, nappH, eta, h0, V, ws,target_indices, gfmeth, 
                    one_body_terms,two_body_terms,coeff_thresh ,L) 
    t_end = time.time()
    print(f"  Calculation finished in {t_end-t_start:.2f}s")

    G_mat_exact, Z = get_exact_green(H_ed_full,c,c_dag,ws,eta,10)
    
    A_mat_lanc_wf = -(1/np.pi) * np.imag(G_mat_lanc_wf)
    A_mat_exact = -(1/np.pi) * np.imag(G_mat_exact)
    
    diff = np.sum(np.abs(G_mat_exact - G_mat_lanc_wf))
    print(f"difference to exact result : {diff}")

    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    for i in range(K):
        for j in range(K):
            if i==j:
                print(f"i,j = {i,j}")
                print(f"exact first freq: {G_mat_exact[:10,i,i]}")
                print(f"lanczos first freq: {G_mat_lanc_wf[:10,i,i]}")

                #plt.plot(ws, A_mat_lanc_wf[:, i, j], label="A_"+str(i)+"(ω)")
                #plt.plot(ws, A_mat_exact[:, i, j], label="A_ex"+str(i)+"(ω)")
                plt.plot(ws, A_mat_lanc_wf[:, i, j]-A_mat_exact[:, i, j], label="A_"+str(i)+"(ω)")
    plt.title("Spectral Function for Hubbard Dimer (Matrix-Free Lanczos)")
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_all_components_grid(ws, G_mat_lanc_wf, G_mat_exact, "comp", "comp")
    
    print("\nScript finished. Check the plot for the spectral function.")