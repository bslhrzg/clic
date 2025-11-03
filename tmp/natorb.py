import numpy as np
from scipy.linalg import eigh, block_diag, eig
from scipy.sparse.linalg import eigsh
from clic import * 
import numpy as np
from typing import Tuple, Literal

def _probs(c: np.ndarray) -> np.ndarray:
    """
    Return normalized probabilities p_i = |c_i|^2 / sum_j |c_j|^2.
    Handles complex c and the all-zero edge case.
    """
    c = np.asarray(c)
    w = np.abs(c)**2
    Z = w.sum()
    if Z == 0:
        # convention: uniform if the vector is identically zero
        return np.ones_like(w, dtype=float) / max(1, w.size)
    return w / Z

def ipr(c: np.ndarray) -> float:
    """
    Inverse Participation Ratio: sum_i p_i^2, where p_i = |c_i|^2 / sum |c|^2.
    Larger IPR => more concentrated.
    """
    p = _probs(c)
    return float(np.sum(p**2))

def pr(c: np.ndarray) -> float:
    """
    Participation number: 1 / IPR. Interpreted as the 'effective' number of components.
    """
    I = ipr(c)
    return float(1.0 / I) if I > 0 else np.inf

def ipr_sparsity_score(c: np.ndarray) -> float:
    """
    Normalized IPR sparsity score in [0,1]:
        S_IPR = (IPR - 1/n) / (1 - 1/n)
    Returns 1 for a basis vector, 0 for the uniform vector.
    """
    n = max(1, int(np.size(c)))
    if n == 1:
        return 1.0
    I = ipr(c)
    return float((I - 1.0/n) / (1.0 - 1.0/n))

def shannon_entropy(c: np.ndarray, base: float = np.e) -> float:
    """
    Shannon entropy H = -sum p_i log p_i (with log in the given base).
    Zeros are handled with the usual 0*log0 -> 0 convention.
    """
    p = _probs(c)
    # mask zeros to avoid log issues
    mask = p > 0
    logp = np.log(p[mask])
    if base != np.e:
        logp = logp / np.log(base)
    H = -float(np.dot(p[mask], logp))
    return H

def shannon_sparsity_score(c: np.ndarray, base: float = np.e) -> float:
    """
    Normalized Shannon sparsity score in [0,1]:
        S_Sh = 1 - H / log_base(n)
    Returns 1 for a basis vector, 0 for the uniform vector.
    """
    n = max(1, int(np.size(c)))
    if n == 1:
        return 1.0
    H = shannon_entropy(c, base=base)
    logn = np.log(n) / (np.log(base) if base != np.e else 1.0)
    return float(1.0 - H / logn)

def top_k_coverage(c: np.ndarray, alpha: float = 0.9) -> Tuple[int, float]:
    """
    Smallest K such that sum of the K largest p_i >= alpha.
    Returns (K, K/n). If alpha <= 0, returns (0, 0.0). If alpha > 1, clamps to 1.
    """
    p = _probs(c)
    n = int(p.size)
    if n == 0:
        return 0, 0.0
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0:
        return 0, 0.0
    ps = np.sort(p)[::-1]
    cs = np.cumsum(ps)
    K = int(np.searchsorted(cs, a, side="left") + 1)
    K = min(K, n)
    return K, K / n

def epsilon_support_size(
    c: np.ndarray,
    epsilon: float,
    mode: Literal["amplitude", "prob"] = "amplitude"
) -> int:
    """
    Count of indices above a threshold:
      - mode="amplitude": |c_i| >= epsilon     (threshold on amplitudes)
      - mode="prob":      p_i   >= epsilon     (threshold on probabilities)
    """
    if mode == "amplitude":
        return int(np.count_nonzero(np.abs(c) >= epsilon))
    elif mode == "prob":
        p = _probs(c)
        return int(np.count_nonzero(p >= epsilon))
    else:
        raise ValueError("mode must be 'amplitude' or 'prob'")

# -------------

def get_impurity_integrals_(M, u, e_bath, V_bath, mu):
    """
    Constructs the single-particle Hamiltonian matrix for one spin channel,
    ensuring correct particle-hole symmetric setup for the Anderson model.
    """
    nb = M - 1
    h_spin = np.zeros((M, M))
    
    # For particle-hole symmetry, impurity on-site energy epsilon_d = -u/2.
    # The Hamiltonian term is (epsilon_d - mu) * n_imp.
    epsilon_d = -u / 2.0
    h_spin[0, 0] = epsilon_d - mu
    
    if nb > 0:
        np.fill_diagonal(h_spin[1:, 1:], e_bath - mu)
        
    h_spin[0, 1:] = V_bath
    h_spin[1:, 0] = V_bath
    return h_spin, u

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


def get_double_chain_transform(h_spin, u, Nelec):
    """
    Performs the 5-step Natural Orbital transformation, yielding a Hamiltonian
    with a double-chain structure and the corresponding unitary transformation matrix C.

    Args:
        h_spin (np.ndarray): The initial single-particle Hamiltonian (M x M).
        u (float): The on-site interaction strength.
        Nelec (int): The number of electrons in the spin sector (Nelec_total / 2).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - h_final_matrix (np.ndarray): The final Hamiltonian in the double-chain basis.
            - C_total (np.ndarray): The total unitary transformation matrix C.
    """
    M = h_spin.shape[0]

    # --- (i) Mean-Field ---
    h_mf = h_spin.copy()
    h_mf[0, 0] += u * 0.5

    # --- (ii) Natural Orbital Basis for the Bath ---
    e_mf, C_mf = eigh(h_mf)
    rho_mf = C_mf[:, :Nelec] @ C_mf[:, :Nelec].T
    rho_bath = rho_mf[1:, 1:]
    n_no, W = eigh(rho_bath)
    
    occupations_dist_from_integer = np.minimum(n_no, 1 - n_no)
    b_idx = np.argmax(occupations_dist_from_integer)
    
    other_indices = [i for i in range(M - 1) if i != b_idx]
    # Sorting ensures a deterministic basis ordering
    filled_indices = sorted([i for i in other_indices if n_no[i] > 0.5])
    empty_indices = sorted([i for i in other_indices if n_no[i] <= 0.5])
    
    ordered_bath_indices = [b_idx] + filled_indices + empty_indices
    W_ordered = W[:, ordered_bath_indices]
    C1 = block_diag(1, W_ordered)

    # --- (iii) Bonding/Anti-bonding Transformation ---
    rho_no_basis = C1.T @ rho_mf @ C1
    rho_ib = rho_no_basis[:2, :2]
    _, U_bond = eigh(rho_ib)
    C2 = np.identity(M)
    C2[:2, :2] = U_bond
    
    C_upto_decoupled = C1 @ C2

    # --- (iv) Lanczos Tridiagonalization (Chain Transformation) ---
    h_decoupled = C_upto_decoupled.T @ h_mf @ C_upto_decoupled

    num_filled = len(filled_indices)
    conduction_indices = [0] + list(range(2 + num_filled, M))
    valence_indices = [1] + list(range(2, 2 + num_filled))

    h_conduction = h_decoupled[np.ix_(conduction_indices, conduction_indices)]
    v0_c = np.zeros(h_conduction.shape[0]); v0_c[0] = 1.0
    T_c, Q_c = lanczos_tridiagonalization(h_conduction, v0_c)

    h_valence = h_decoupled[np.ix_(valence_indices, valence_indices)]
    v0_v = np.zeros(h_valence.shape[0]); v0_v[0] = 1.0
    T_v, Q_v = lanczos_tridiagonalization(h_valence, v0_v)
    
    C_lanczos = np.identity(M)
    if Q_c.shape[1] > 0:
      C_lanczos[np.ix_(conduction_indices, conduction_indices)] = Q_c
    if Q_v.shape[1] > 0:
      C_lanczos[np.ix_(valence_indices, valence_indices)] = Q_v
      
    # C_to_chains transforms from original basis to the basis of Lanczos vectors
    # (permuted).
    C_to_chains = C_upto_decoupled @ C_lanczos

    # --- (v) Construct Final Basis and Transformation Matrix C_total ---
    # The final basis is {|i>, |b>, |c1>, |c2>, ..., |v1>, |v2>, ...}
    # where |i> and |b> are the impurity and special NO in the C1 basis.
    # The other vectors are the Lanczos chain vectors (excluding chain heads).
    
    # Get the vector representations of the chain states in the original basis
    # These are the columns of the C_to_chains matrix
    len_c = Q_c.shape[1]
    len_v = Q_v.shape[1]
    
    # The vector for the head of the conduction chain, |c0> = |A>, in the original basis
    c0_vec = C_to_chains[:, conduction_indices[0]] if len_c > 0 else np.zeros(M)
    # The vector for the head of the valence chain, |v0> = |B>, in the original basis
    v0_vec = C_to_chains[:, valence_indices[0]] if len_v > 0 else np.zeros(M)

    C_total = np.zeros((M, M))
    
    # The first two columns of C_total are the impurity |i> and special NO |b>
    # We recover them by rotating |c0> and |v0> back with U_bond.T
    # |i> = U_bond[0,0]|c0> + U_bond[0,1]|v0>
    # |b> = U_bond[1,0]|c0> + U_bond[1,1]|v0>
    C_total[:, 0] = U_bond[0, 0] * c0_vec + U_bond[0, 1] * v0_vec
    C_total[:, 1] = U_bond[1, 0] * c0_vec + U_bond[1, 1] * v0_vec

    # The remaining columns are the rest of the Lanczos chain vectors
    c_chain_start_idx = 2
    if len_c > 1:
        c_rest_indices_in_decoupled_basis = [conduction_indices[i] for i in range(1, len_c)]
        C_total[:, c_chain_start_idx : c_chain_start_idx + len_c - 1] = C_to_chains[:, c_rest_indices_in_decoupled_basis]

    v_chain_start_idx = c_chain_start_idx + (len_c - 1 if len_c > 0 else 0)
    if len_v > 1:
        v_rest_indices_in_decoupled_basis = [valence_indices[i] for i in range(1, len_v)]
        C_total[:, v_chain_start_idx : v_chain_start_idx + len_v - 1] = C_to_chains[:, v_rest_indices_in_decoupled_basis]

    # --- Transform the Hamiltonian and correct for the mean-field shift ---
    h_mf_final_basis = C_total.T @ h_mf @ C_total
    
    mean_field_correction_term = np.zeros_like(h_spin)
    mean_field_correction_term[0, 0] = u * 0.5
    transformed_correction = C_total.T @ mean_field_correction_term @ C_total
    
    h_final_matrix = h_mf_final_basis - transformed_correction
    
    return h_final_matrix, C_total


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

# --- Main script ---

np.set_printoptions(precision=4, suppress=True)


# 1. System Parameters
nb = 5
M = 1 + nb
u = 4.0
mu = u / 2
Nelec = M 
Nelec_half = M // 2
e_bath = np.linspace(-2.0, 2.0, nb)
if nb == 1: e_bath = [0.0]
V_bath = np.full(nb, 0.5)
    
h0, U_mat = get_impurity_integrals(M, u, e_bath, V_bath, mu)


# ------------------------------------------------------------------
print('-'*42)

print("Star geometry")
print('-'*42)


# --- Find Ground State ---
basis = get_fci_basis(M, Nelec)
basis, idxs0 = subbasis_by_Sz(basis, 0.0)  # S_z = 0 sector


print(f"basis size = {len(basis)}")
H_sparse = build_hamiltonian_openmp(basis, h0, U_mat)
eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
e0 = eigvals[0]
print(f"Impurity Model Ground State Energy: {e0:.6f}")
psi0star = eigvecs[:, 0]

sb = get_starting_basis(h0, M+1)
print("star starting basis : ")
for d in sb:
    print(d)

# ------------------------------------------------------------------
print('-'*42)
print("NO geometry")
print('-'*42)


psi0_wf = Wavefunction(M, basis, psi0star)
rdm1 = one_rdm(psi0_wf,M)
eno,nos = eig(rdm1[:M,:M])
nos = double_h(nos,M)
h0no,Uno = basis_change_h0_U(h0,U_mat,nos)


print("h0 in rdm1")
print(np.real(h0no[:M,:M]))

H_sparse = get_ham(basis, h0no, Uno)
eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
e0 = eigvals[0]
print(f"Impurity Model Ground State Energy: {e0:.6f}")
psi0no = eigvecs[:, 0]

# ------------------------------------------------------------------
print('-'*42)
print("HF geometry")
print('-'*42)



hmf, es, Vs, rho = mfscf(h0,U_mat,M)
print("Vs^+ * Vs = ")
print(Vs.T.conj() @ Vs)
h0hf,Uhf = basis_change_h0_U(h0,U_mat,Vs)



H_sparse = get_ham(basis, h0hf, Uhf)
eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
e0 = eigvals[0]
print(f"Impurity Model Ground State Energy: {e0:.6f}")
psi0hf = eigvecs[:, 0]

#assert 1 == 0

print("h0hf(diag) = ")
print(np.diag(h0hf))
sb = get_starting_basis(h0hf, M)
print("hf starting basis : ")
for d in sb:
    print(d)

# ------------------------------------------------------------------
print('-'*42)
print("Bath No ")
print('-'*42)

def get_natural_orbital_transform(h_spin, u, Nelec):
    """
    Performs a simple Natural Orbital transformation
    This corresponds to steps (i) and (ii) of the more complex procedure.
    """
    M = h_spin.shape[0]

    # Step 1: Mean-field Hamiltonian and its density matrix
    h_mf = h_spin.copy()
    h_mf[0, 0] += u * 0.5
    
    e_mf, C_mf = eigh(h_mf)
    rho_mf = C_mf[:, :Nelec] @ C_mf[:, :Nelec].T

    # Step 2: Diagonalize the bath part of the density matrix
    rho_bath = rho_mf[1:, 1:]
    # W contains the eigenvectors of rho_bath, which define the new bath basis
    _, W = eigh(rho_bath)
    
    # The transformation matrix C_no leaves the impurity alone and transforms the bath
    C_no = block_diag(1, W)
    
    # Apply the transformation to the original non-interacting Hamiltonian
    h_new_basis = C_no.T @ h_spin @ C_no
    
    return h_new_basis, C_no


h0_spin = np.real(h0[:M,:M])
print("h0_spin = ")
print(h0_spin)
h0bno,Cbno = get_natural_orbital_transform(h0_spin,u,Nelec_half)
print("h0bno = ")
print(h0bno)
Cbno = block_diag(Cbno,Cbno)
h0bno,Ubno = basis_change_h0_U(h0,U_mat,Cbno)


H_sparse = get_ham(basis, h0bno, Ubno)
eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
e0 = eigvals[0]
print(f"Impurity Model Ground State Energy: {e0:.6f}")
psi0bno = eigvecs[:, 0]


# ------------------------------------------------------------------
print('-'*42)
print("Double chain geometry")
print('-'*42)


h0_spin = np.real(h0[:M,:M])

# 2. Get initial Hamiltonian and perform transformation
#h0_spin, U = get_impurity_integrals(M, u, e_bath, V_bath, mu)
final_hamiltonian_params = perform_natural_orbital_transform(h0_spin, u, Nelec_half)

# 3. Print the results
print("="*45)
print(" Final Natural-Orbital Hamiltonian Parameters ")
print("="*45)
print(f"Special bath orbital 'b' occupation: {final_hamiltonian_params['b_occupation']:.4f}")
print(f"Anti/Bonding site occupations: {np.round(final_hamiltonian_params['ab_occupations'], 4)} (expect ~[0, 1])")
# ... (rest of the printing logic is the same)
print("\n--- Central Block ---")
print(f"  e_i (impurity): {final_hamiltonian_params['e_i']:.4f}")
print(f"  e_b (special):  {final_hamiltonian_params['e_b']:.4f}")
print(f"  t_ib (hopping): {final_hamiltonian_params['t_ib']:.4f}")
print("\n--- Couplings to Chains ---")
print(f"  V_i_c (i -> conduction): {final_hamiltonian_params['V_i_c']:.4f}")
print(f"  V_b_c (b -> conduction): {final_hamiltonian_params['V_b_c']:.4f}")
print(f"  V_i_v (i -> valence):    {final_hamiltonian_params['V_i_v']:.4f}")
print(f"  V_b_v (b -> valence):    {final_hamiltonian_params['V_b_v']:.4f}")
print("\n--- Conduction Chain ('Empty' Bath) ---")
print(f"  Energies: {np.round(final_hamiltonian_params['conduction_e'], 4)}")
print(f"  Hoppings: {np.round(final_hamiltonian_params['conduction_t'], 4)}")
print("\n--- Valence Chain ('Filled' Bath) ---")
print(f"  Energies: {np.round(final_hamiltonian_params['valence_e'], 4)}")
print(f"  Hoppings: {np.round(final_hamiltonian_params['valence_t'], 4)}")
print("="*45, "\n")


# 4. Construct and print the final Hamiltonian Matrix
H_final_matrix = construct_final_hamiltonian_matrix(final_hamiltonian_params, M)
H_final_matrix[0,0] = -u/2
print("--- Final Transformed Hamiltonian Matrix ---")
print("Basis: [i, b, c1, c2, c3, v1, v2, v3]")
print(H_final_matrix)
print("-" * 42)

#-------------

h0 = double_h(H_final_matrix,M)
h0 = np.ascontiguousarray(h0, dtype=np.complex128)

sb = get_starting_basis(h0, M)
print("double chain starting basis : ")
for d in sb:
    print(d)

# --- Find Ground State ---
print(f"basis size = {len(basis)}")
H_sparse = build_hamiltonian_openmp(basis, h0, U_mat)
eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
e0 = eigvals[0]
psi0dbchain = eigvecs[:, 0]

print(f"Impurity Model Ground State Energy: {e0:.6f}")

for name, v in [("star", psi0star), ("hf", psi0hf), ("dbl chain", psi0dbchain), ("NO", psi0no), ("BNO", psi0bno)]:
        print(f"\n{name}:")
        print("  IPR =", ipr(v))
        print("  PR  =", pr(v))
        print("  S_IPR =", ipr_sparsity_score(v))
        print("  H =", shannon_entropy(v))
        print("  S_Sh =", shannon_sparsity_score(v))
        print("  K_0.9 =", top_k_coverage(v, alpha=0.9))
        print("  N_eps(|c|>=0.01) =", epsilon_support_size(v, 0.01, mode="amplitude"))
        print("  N_eps(p>=0.1)   =", epsilon_support_size(v, 0.1, mode="prob"))



