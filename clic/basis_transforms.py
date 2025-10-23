import numpy as np
from scipy.linalg import eigh, block_diag


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
