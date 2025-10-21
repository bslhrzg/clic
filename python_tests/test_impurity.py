import numpy as np
import numpy.linalg as npl
from scipy.sparse.linalg import eigsh
import clic_clib as qc
import time
import matplotlib.pyplot as plt
from itertools import combinations


# --- Integral Generation for Anderson Impurity Model ---
def get_impurity_integrals(M, u, e_bath, V_bath, mu):
    """
    Builds the h0 and U integrals for the Anderson Impurity Model.
    """
    K = 2 * M
    h_spatial = np.zeros((M, M))
    diagonal_elements = np.concatenate(([-mu], e_bath))
    np.fill_diagonal(h_spatial, diagonal_elements)
    h_spatial[0, 1:] = V_bath
    h_spatial[1:, 0] = np.conj(V_bath)

    h0 = np.zeros((K, K))
    h0[0:M, 0:M] = h_spatial
    h0[M:K, M:K] = h_spatial
    
    U = np.zeros((K, K, K, K))
    imp_alpha_idx, imp_beta_idx = 0, M
    U[imp_alpha_idx, imp_beta_idx, imp_alpha_idx, imp_beta_idx] = u
    U[imp_beta_idx, imp_alpha_idx, imp_beta_idx, imp_alpha_idx] = u
    return h0, U

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
def make_H_on_psi(one_body_terms, two_body_terms, prune_thresh=1e-6):
    """Factory function to create the H|psi> operator."""
    def H_on_psi(psi_in):
        psi_out1 = qc.apply_one_body_operator(psi_in, one_body_terms)
        psi_out2 = qc.apply_two_body_operator(psi_in, two_body_terms)
        psi_res = psi_out1 + psi_out2
        psi_res.prune(prune_thresh)
        return psi_res
    return H_on_psi

# --- Gram-Schmidt for Wavefunction objects (HIGH-PERFORMANCE VERSION) ---
def gram_schmidt_qr(block_V):
    """Performs QR decomposition on a block of Wavefunction vectors efficiently."""
    Q = []
    num_vecs = len(block_V)
    if num_vecs == 0:
        return Q, np.array([[]])
    R = np.zeros((num_vecs, num_vecs), dtype=np.complex128)
    
    # Make copies of the input vectors so we don't modify the originals
    V_copy = [qc.Wavefunction(v.n_spatial, v.data()) for v in block_V]
    
    for j in range(num_vecs):
        vj = V_copy[j]
        # Subtract projections onto previous orthonormal vectors
        for i in range(len(Q)):
            qi = Q[i]
            # Perform the projection and subtraction IN-PLACE
            proj_coeff = qi.dot(vj)
            R[i, j] = proj_coeff
            vj.add_wavefunction(qi, -proj_coeff) # The critical change!
        
        # Normalize
        norm_vj_sq = abs(vj.dot(vj))
        if norm_vj_sq > 1e-24:
            norm_vj = np.sqrt(norm_vj_sq)
            R[j, j] = norm_vj # Note: Using R[j,j] is more standard for QR
            
            # This is the only place a new WF is made, for scaling.
            # A future optimization could be an in-place scale method.
            qj = (1.0 / norm_vj) * vj
            Q.append(qj)
        else:
            # Linear dependence detected, break and return the rank-deficient result
            break

    rank = len(Q)
    # Return the correctly shaped R matrix for the calculated rank
    return Q, R[:rank, :num_vecs]


# --- Block Lanczos with Wavefunction objects (HIGH-PERFORMANCE VERSION) ---
def get_block_lanczos_wf(H_op, S_block, L, reorth=False):
    if not S_block: raise ValueError("Seed block is empty.")

    Q0_list, R0 = gram_schmidt_qr(S_block)
    if not Q0_list: return [], [], [], R0
    
    Qs, As, Bs = [Q0_list], [], []
    
    # A_k = Q_k^dag * H * Q_k
    HQ0 = [H_op(q) for q in Q0_list]
    A0 = np.array([[qi.dot(hqj) for hqj in HQ0] for qi in Q0_list])
    As.append(A0)
    
    Q_prev_list = []
    
    for k in range(L):
        Qk_list, Ak = Qs[-1], As[-1]
        
        # W = H*Q_k - Q_k*A_k - Q_{k-1}*B_k^dag
        W_list = [H_op(q) for q in Qk_list] # Start with W = H*Q_k
        
        # --- Perform linear combinations IN-PLACE ---
        # W -= Q_k * A_k
        for j in range(len(W_list)):
            for i in range(len(Qk_list)):
                W_list[j].add_wavefunction(Qk_list[i], -Ak[i, j])

        # W -= Q_{k-1} * B_k^dag
        if k > 0 and Q_prev_list:
            Bk_dagger = Bs[-1].conj().T
            for j in range(len(W_list)):
                for i in range(len(Q_prev_list)):
                    W_list[j].add_wavefunction(Q_prev_list[i], -Bk_dagger[i, j])
        
        # Full re-orthogonalization against all previous Q blocks
        if reorth:
            for Qj_list in Qs:
                for w_vec in W_list:
                    for qj_vec in Qj_list:
                        proj = qj_vec.dot(w_vec)
                        w_vec.add_wavefunction(qj_vec, -proj)

        Q_next_list, B_next = gram_schmidt_qr(W_list)
        if not Q_next_list: break # Breakdown: vectors are linearly dependent

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

def green_function_block_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta, impurity_indices):
    """
    Calculates the Green's function matrix G_ij(ω) for a specified subset of orbitals.
    """
    Norb = 2 * M
    
    # 1. Build seed blocks ONLY for the requested impurity orbitals
    # This is the core optimization.
    
    # Create raw seeds (orbital_index, wavefunction)
    raw_add = [(i, qc.apply_creation(psi0_wf, i % M,
                                     qc.Spin.Alpha if i < M else qc.Spin.Beta))
               for i in impurity_indices]
    raw_rem = [(i, qc.apply_annihilation(psi0_wf, i % M,
                                         qc.Spin.Alpha if i < M else qc.Spin.Beta))
               for i in impurity_indices]

    # Filter out empty wavefunctions and get the corresponding orbital indices
    add_pairs = [(i, wf) for (i, wf) in raw_add if wf.data()]
    rem_pairs = [(i, wf) for (i, wf) in raw_rem if wf.data()]
    
    # These indices map from the small Lanczos block back to the full orbital space
    add_idx_map = [i for (i, _) in add_pairs]
    rem_idx_map = [i for (i, _) in rem_pairs]
    
    S_add = [wf for (_, wf) in add_pairs]
    S_rem = [wf for (_, wf) in rem_pairs]

    # 2. Run Block Lanczos in each branch on the smaller seed blocks
    As_g, Bs_g, Qs_g, R0_g = get_block_lanczos_wf(H_op, S_add, L) if S_add else ([], [], [], np.array([]))
    As_l, Bs_l, Qs_l, R0_l = get_block_lanczos_wf(H_op, S_rem, L) if S_rem else ([], [], [], np.array([]))

    have_g = (R0_g.size != 0 and len(As_g) > 0)
    have_l = (R0_l.size != 0 and len(As_l) > 0)

    Nw = len(ws)
    G_all = np.zeros((Nw, Norb, Norb), dtype=np.complex128)

    # 3. Calculate G(w) and place results into the full matrix
    for iw, w in enumerate(ws):
        z_g = (w + e0) + 1j*eta
        z_l = (-w + e0) - 1j*eta

        # Gg_full will hold the result for this frequency
        Gg_full = np.zeros((Norb, Norb), dtype=np.complex128)
        if have_g:
            G00_g = block_cf_top_left(As_g, Bs_g, z_g)
            if G00_g.size != 0 and not np.isnan(G00_g).any():
                # Gg_eff is the small (n_imp x n_imp) Green's matrix
                Gg_eff = R0_g.conj().T @ G00_g @ R0_g
                # Use np.ix_ to place this small matrix into the correct sub-block
                # of the full-sized matrix.
                Gg_full[np.ix_(add_idx_map, add_idx_map)] = Gg_eff

        Gl_full = np.zeros((Norb, Norb), dtype=np.complex128)
        if have_l:
            G00_l = block_cf_top_left(As_l, Bs_l, z_l)
            if G00_l.size != 0 and not np.isnan(G00_l).any():
                Gl_eff = R0_l.conj().T @ G00_l @ R0_l
                Gl_full[np.ix_(rem_idx_map, rem_idx_map)] = Gl_eff

        G_all[iw, :, :] = Gg_full - Gl_full

    return G_all

def get_scalar_lanczos_wf(H_op, v_init, L):
    """
    Robust scalar Lanczos for a single Wavefunction vector.
    """
    alphas, betas = [], []
    v_list = []

    # Normalize initial vector
    norm_v = np.sqrt(abs(v_init.dot(v_init)))
    if norm_v < 1e-14:
        return np.array(alphas), np.array(betas)
    
    v_list.append((1.0 / norm_v) * v_init)

    for j in range(L):
        q = v_list[j]
        w = H_op(q)
        
        # Re-orthogonalize against previous two vectors (maintains orthogonality)
        if j > 0:
            w = w - betas[j-1] * v_list[j-1]
        
        alpha = q.dot(w)
        alphas.append(alpha)
        
        w = w - alpha * q
        
        beta = np.sqrt(abs(w.dot(w)))
        
        if beta < 1e-12:
            break # Breakdown
        
        betas.append(beta)
        v_list.append((1.0 / beta) * w)

    return np.array(alphas), np.array(betas)

def green_function_scalar_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta, impurity_indices):
    """
    Calculates DIAGONAL elements G_ii(ω) using a corrected scalar Lanczos.
    """
    Norb = 2 * M
    Nw = len(ws)
    G_all = np.zeros((Nw, Norb, Norb), dtype=np.complex128)

    for i in impurity_indices:
        # --- Greater Green's function (particle addition) ---
        seed_g = qc.apply_creation(psi0_wf, i % M, qc.Spin.Alpha if i < M else qc.Spin.Beta)
        norm_g_sq = abs(seed_g.dot(seed_g))
        
        if norm_g_sq > 1e-12:
            a_g, b_g = get_scalar_lanczos_wf(H_op, seed_g, L)
            z = (ws + e0) + 1j * eta
            
            # CORRECTED Continued Fraction
            g_g = np.zeros_like(z)
            for k in range(len(a_g) - 1, -1, -1):
                b2 = b_g[k]**2 if k < len(b_g) else 0.0
                g_g = 1.0 / (z - a_g[k] - b2 * g_g)
            
            G_all[:, i, i] += norm_g_sq * g_g

        # --- Lesser Green's function (particle removal) ---
        seed_l = qc.apply_annihilation(psi0_wf, i % M, qc.Spin.Alpha if i < M else qc.Spin.Beta)
        norm_l_sq = abs(seed_l.dot(seed_l))
        
        if norm_l_sq > 1e-12:
            a_l, b_l = get_scalar_lanczos_wf(H_op, seed_l, L)
            z = (-ws + e0) - 1j * eta
            
            # CORRECTED Continued Fraction
            g_l = np.zeros_like(z)
            for k in range(len(a_l) - 1, -1, -1):
                b2 = b_l[k]**2 if k < len(b_l) else 0.0
                g_l = 1.0 / (z - a_l[k] - b2 * g_l)
            
            G_all[:, i, i] -= norm_l_sq * g_l
            
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
    # --- System Setup ---
    nb = 7
    M = 1 + nb
    u = 1
    mu = u/2
    Nelec = M
    e_bath = np.linspace(-2.0, 2.0, nb)
    if nb == 1 :
        e_bath = [0.0]
    print("e_bath = ",e_bath)
    V_bath = np.full(nb, 0.1)
    
    h0, U_mat = get_impurity_integrals(M, u, e_bath, V_bath, mu)
    h0_clean = np.ascontiguousarray(h0, dtype=np.complex128)
    U_clean = np.ascontiguousarray(U_mat, dtype=np.complex128)

    print("h0 = ")
    print(h0)

    # --- Find Ground State ---
    basis = get_fci_basis(M, Nelec)
    print(f"basis size = {len(basis)}")
    H_sparse = qc.build_hamiltonian_openmp(basis, h0_clean, U_clean)
    eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
    e0 = eigvals[0]
    psi0_wf = qc.Wavefunction(M, basis, eigvecs[:, 0])
    
    print(f"Impurity Model Ground State Energy: {e0:.6f}")

    # --- Matrix-Free Lanczos Calculation ---
    one_body_terms = get_one_body_terms(h0_clean, M)
    two_body_terms = get_two_body_terms(U_clean, M)
    H_op = make_H_on_psi(one_body_terms, two_body_terms)
    
    ws = np.linspace(-6, 6, 1001)
    eta = 0.02
    L = 100

    
    # Call the BLOCK version
    impurity_indices = [0, M] 
    print(f"\nRunning MATRIX-FREE BLOCK Lanczos for impurity orbitals {impurity_indices}...")
    t_start = time.time()
    G_block = green_function_block_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta, impurity_indices)
    t_end = time.time()
    print(f"  Calculation finished in {t_end-t_start:.3f}s")
    A_block = -(1/np.pi) * np.imag(G_block)
    
    # Call the SCALAR version
    print(f"\nRunning EFFICIENT SCALAR Lanczos for impurity orbitals {impurity_indices}...")
    t_start = time.time()
    G_scalar = green_function_scalar_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta, impurity_indices)
    t_end = time.time()
    print(f"  Calculation finished in {t_end-t_start:.3f}s")
    A_scalar = -(1/np.pi) * np.imag(G_scalar)
    
    # --- Verify they give the same diagonal elements ---
    #np.testing.assert_allclose(A_block, A_scalar, atol=1e-6)
    #print("\n✅ SUCCESS: Scalar Lanczos results match Block Lanczos diagonal.")
    
    A_mat_lanc_wf = -(1/np.pi) * np.imag(G_scalar)
    
    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    for (i,ii) in enumerate(impurity_indices):
        plt.plot(ws, i*10+(A_mat_lanc_wf[:, ii, ii]), label="A_sc_"+str(i)+"(ω)")
        plt.plot(ws, i*10+(A_block[:, ii, ii]), label="A_bl_"+str(i)+"(ω)")
    plt.title("Impurity Spectral Function for Anderson Model (Matrix-Free Lanczos)")
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nScript finished. Check the plot for the spectral function.")