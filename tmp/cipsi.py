import numpy as np
from scipy.linalg import eigh, block_diag, eig
from scipy.sparse.linalg import eigsh
import numpy as np
from typing import Tuple, Literal
import matplotlib.pyplot as plt

from clic import * 


# --- Main script ---

np.set_printoptions(precision=4, suppress=True)


# 1. System Parameters
nb = 31
M = 1 + nb
u = 2.0
mu = u / 2
Nelec = M 
Nelec_half = M // 2
e_bath = np.linspace(-0.2, 0.2, nb)
if nb == 1: e_bath = [0.0]
V_bath = np.full(nb, 0.1)


    
h0, U_mat = get_impurity_integrals(M, u, e_bath, V_bath, mu)


# -------------------------

def lanczos_tridiagonalization(H, v0):
    """
    Performs the Lanczos algorithm to tridiagonalize a Hermitian matrix H.
    """
    N = H.shape[0]
    if N == 0:
        return np.array([[]]), np.array([[]])
        
    Q = np.zeros((N, N), dtype=float)
    v = v0 / np.linalg.norm(v0)
    Q[:, 0] = v

    alphas = []
    betas = []

    for j in range(N):
        w = H @ v
        alpha = np.dot(v.conj(), w)
        alphas.append(alpha)

        if j < N - 1:
            w = w - alpha * v
            if j > 0:
                w = w - betas[-1] * Q[:, j-1]
            
            beta = np.linalg.norm(w)
            
            if beta < 1e-12: # Space is exhausted
                N = j + 1 # Truncate the dimension
                break
            
            betas.append(beta)
            v = w / beta
            if j + 1 < Q.shape[1]:
                Q[:, j + 1] = v
    
    T = np.diag(alphas) + np.diag(betas, k=1) + np.diag(betas, k=-1)
    return T[:N, :N], Q[:, :N]

def perform_natural_orbital_transform(h_spin, u, Nelec):
    """
    Performs the 5-step Natural Orbital transformation on a single-particle Hamiltonian.
    """
    M = h_spin.shape[0]
    final_params = {}

    # (i) Mean-Field
    h_mf = h_spin.copy()
    h_mf[0, 0] += u * 0.5 # <n_down> = 0.5
    e_mf, C_mf = eigh(h_mf)
    rho_mf = C_mf[:, :Nelec] @ C_mf[:, :Nelec].T

    # (ii) Diagonalize Bath Density Matrix
    rho_bath = rho_mf[1:, 1:]
    n_no, W = eigh(rho_bath)
    
    occupations_dist_from_integer = np.minimum(n_no, 1 - n_no)
    b_idx = np.argmax(occupations_dist_from_integer)
    final_params['b_occupation'] = n_no[b_idx]
    
    filled_indices = [i for i, n in enumerate(n_no) if i != b_idx and n > 0.5]
    empty_indices = [i for i, n in enumerate(n_no) if i != b_idx and n <= 0.5]

    ordered_indices = [b_idx] + filled_indices + empty_indices
    W_ordered = W[:, ordered_indices]
    U1 = block_diag(1, W_ordered)
    h_no_basis = U1.T @ h_mf @ U1

    # (iii) Bonding/Anti-bonding
    rho_no_basis = U1.T @ rho_mf @ U1
    rho_ib = rho_no_basis[:2, :2]
    n_ab, U_bond = eigh(rho_ib)
    final_params['ab_occupations'] = n_ab

    U2 = np.identity(M)
    U2[:2, :2] = U_bond
    h_decoupled = U2.T @ h_no_basis @ U2

    # (iv) Lanczos
    h_conduction_indices = [0] + list(range(2 + len(filled_indices), M))
    h_conduction = h_decoupled[np.ix_(h_conduction_indices, h_conduction_indices)]
    v0_c = np.zeros(h_conduction.shape[0]); v0_c[0] = 1.0
    T_c, _ = lanczos_tridiagonalization(h_conduction, v0_c)

    h_valence_indices = [1] + list(range(2, 2 + len(filled_indices)))
    h_valence = h_decoupled[np.ix_(h_valence_indices, h_valence_indices)]
    v0_v = np.zeros(h_valence.shape[0]); v0_v[0] = 1.0
    T_v, _ = lanczos_tridiagonalization(h_valence, v0_v)

    # (v) Recover final parameters
    E_A = T_c[0, 0] if T_c.shape[0] > 0 else 0.0
    E_B = T_v[0, 0] if T_v.shape[0] > 0 else 0.0
    h_ib_final = U_bond @ np.diag([E_A, E_B]) @ U_bond.T
    
    final_params['e_i'] = h_ib_final[0, 0]
    final_params['e_b'] = h_ib_final[1, 1]
    final_params['t_ib'] = h_ib_final[0, 1]

    V_A_to_chain = T_c[0, 1] if T_c.shape[0] > 1 else 0.0
    V_B_to_chain = T_v[0, 1] if T_v.shape[0] > 1 else 0.0

    final_params['V_i_c'] = V_A_to_chain * U_bond[0, 0]
    final_params['V_b_c'] = V_A_to_chain * U_bond[1, 0]
    final_params['V_i_v'] = V_B_to_chain * U_bond[0, 1]
    final_params['V_b_v'] = V_B_to_chain * U_bond[1, 1]

    final_params['conduction_e'] = np.diag(T_c)[1:]
    final_params['conduction_t'] = np.diag(T_c, k=1)[1:]
    final_params['valence_e'] = np.diag(T_v)[1:]
    final_params['valence_t'] = np.diag(T_v, k=1)[1:]

    return final_params

def construct_final_hamiltonian_matrix(params, M):
    """
    Constructs the final M x M Hamiltonian matrix from the parameter dictionary.
    The basis order is [i, b, c1, c2, ..., v1, v2, ...].
    """
    H_final = np.zeros((M, M))
    
    # i, b block
    H_final[0, 0] = params['e_i']
    H_final[1, 1] = params['e_b']
    H_final[0, 1] = H_final[1, 0] = params['t_ib']

    len_c_chain = len(params['conduction_e'])
    len_v_chain = len(params['valence_e'])

    # -- Conduction Chain --
    if len_c_chain > 0:
        c_start_idx = 2
        c_indices = np.arange(c_start_idx, c_start_idx + len_c_chain)
        # Couplings to chain
        H_final[0, c_start_idx] = H_final[c_start_idx, 0] = params['V_i_c']
        H_final[1, c_start_idx] = H_final[c_start_idx, 1] = params['V_b_c']
        # On-site energies
        # **CORRECTED LINE**: Use direct assignment, not fill_diagonal on a 1D view
        H_final[c_indices, c_indices] = params['conduction_e']
        # Hoppings
        if len_c_chain > 1:
            # **CORRECTED LINE**
            H_final[c_indices[:-1], c_indices[1:]] = params['conduction_t']
            H_final[c_indices[1:], c_indices[:-1]] = params['conduction_t']
            
    # -- Valence Chain --
    if len_v_chain > 0:
        v_start_idx = 2 + len_c_chain
        v_indices = np.arange(v_start_idx, v_start_idx + len_v_chain)
        # Couplings to chain
        H_final[0, v_start_idx] = H_final[v_start_idx, 0] = params['V_i_v']
        H_final[1, v_start_idx] = H_final[v_start_idx, 1] = params['V_b_v']
        # On-site energies
        # **CORRECTED LINE**
        H_final[v_indices, v_indices] = params['valence_e']
        # Hoppings
        if len_v_chain > 1:
            # **CORRECTED LINE**
            H_final[v_indices[:-1], v_indices[1:]] = params['valence_t']
            H_final[v_indices[1:], v_indices[:-1]] = params['valence_t']
            
    return H_final


# --- Find Ground State ---
do_fci = False
if do_fci:
    basis = get_fci_basis(M, Nelec)
    #inds, blocks = partition_by_Sz(basis)    # lists of indices + Sz values
    basis, idxs0 = subbasis_by_Sz(basis, 0.0)  # S_z = 0 sector
    print(f"basis size = {len(basis)}")


    H_sparse = build_hamiltonian_openmp(basis, h0, U_mat)
    eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
    e0 = eigvals[0]
    print(f"Impurity Model Ground State Energy: {e0:.6f}")
    psi0star = eigvecs[:, 0]


# -----------------------

do_hf = False 
if do_hf :
    hmf, es, Vs, rho = mfscf(h0,U_mat,M)
    h0,U_mat = basis_change_h0_U(h0,U_mat,Vs)

    basis0 = get_starting_basis(np.real(h0), M)
    H = get_ham(basis0,h0,U_mat)
    eigvals, eigvecs = eig(H.toarray())

    e0 = eigvals[0]
    psi0 = Wavefunction(M, basis0, eigvecs[:,0])
    print(f"e0 = {e0}")

one_bh = get_one_body_terms(h0, M)
two_bh = get_two_body_terms(U_mat, M)



do_no = False 
if do_no:

    res_cisd = selective_ci(h0,U_mat,M,Nelec,hamiltonian_generator,cipsi_one_iter,max_iter=1,prune_thr=1e-12)
    print("CISD E0 =", res_cisd["energy"])
    print("CISD dim =", len(res_cisd["basis"]))
    psi_cisd = res_cisd["wavefunction"]
    rdm1 = one_rdm(psi_cisd,M)
    eno,nos = eig(rdm1[:M,:M])
    nos = double_h(nos,M)
    h0,U_mat = basis_change_h0_U(h0,U_mat,nos)

do_db = True
if do_db :
    h0_spin = np.real(h0[:M,:M])
    final_hamiltonian_params = perform_natural_orbital_transform(h0_spin, u, Nelec_half)
    H_final_matrix = construct_final_hamiltonian_matrix(final_hamiltonian_params, M)
    H_final_matrix[0,0] = -u/2

    h0 = double_h(H_final_matrix,M)
    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    sb = get_starting_basis(h0, M)

cipsi_max_iter = 10
res = selective_ci(h0,U_mat,M,Nelec,hamiltonian_generator,cipsi_one_iter,Nmul=1.0,max_iter=cipsi_max_iter,prune_thr=1e-6)

print("Final E0 =", res["energy"])
print("Final dim =", len(res["basis"]))


comp_green = True 
if comp_green:

    psi0_wf = res["wavefunction"]
    e0 = res["energy"]

    ws = np.linspace(-6, 6, 1001)

    NappH = 1   # for example; 0 means "just the seed support"
    L = 150
    eta = 0.1
    impurity_indices = [0, M]  # same as before

    print(f"\nRunning MATRIX-FREE BLOCK Lanczos for impurity orbitals {impurity_indices}...")

    G_block, meta = green_function_block_lanczos_fixed_basis(
        M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
        h0, U_mat, one_bh, two_bh,
        coeff_thresh=1e-7, L=L, reorth=False
    )
    print("Fixed-basis sizes:", meta)
   
    A_block = -(1/np.pi) * np.imag(G_block)
    
    dos = np.sum(A_block[:, impurity_indices, impurity_indices], axis=1)
    
    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    #for (i,ii) in enumerate(impurity_indices):
    #    plt.plot(ws, i*np.max(A_block[:,ii,ii])+(A_block[:, ii, ii]), label="A_bl_"+str(i)+"(ω)")
    plt.plot(ws,dos)
    plt.title("Impurity Spectral Function for Anderson Model (Matrix-Free Lanczos)")
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Gimp.png")
    plt.show()