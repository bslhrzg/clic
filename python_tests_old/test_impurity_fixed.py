import numpy as np
import numpy.linalg as npl
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
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

# -----------------------------
# Helpers: wavefunction <-> basis
# -----------------------------

def wavefunction_support(wf, coeff_thresh=1e-14):
    """
    Return the set of determinants in a Wavefunction with |coeff|>coeff_thresh.
    Assumes wf.data() is a mapping {SlaterDeterminant: complex}.
    If your qc exposes a different interface, adapt here.
    """
    data = wf.data()
    # If data is an array, you need to provide (basis_list, coeffs) instead.
    if isinstance(data, dict):
        return {det for det, c in data.items() if abs(c) > coeff_thresh}
    raise RuntimeError("wavefunction_support: adapt to your qc.Wavefunction storage.")

def wf_to_vec(wf, basis_list):
    """
    Project wavefunction onto a given determinant basis ordering -> dense vector.
    Assumes wf.data() is {det: coeff}. If it is an array, adapt here.
    """
    data = wf.data()
    if isinstance(data, dict):
        idx = {det: k for k, det in enumerate(basis_list)}
        v = np.zeros(len(basis_list), dtype=np.complex128)
        for det, c in data.items():
            k = idx.get(det, None)
            if k is not None:
                v[k] += c
        return v
    raise RuntimeError("wf_to_vec: adapt to your qc.Wavefunction storage.")

# -----------------------------
# Build fixed Krylov basis by repeated H-applications at determinant level
# -----------------------------

def expand_basis_by_H(seed_dets, one_body_terms, two_body_terms, NappH):
    """
    Determinant-level expansion:
    B_0 = seed_dets
    B_{t+1} = B_t ∪ H*B_t connections (via one- and two-body graph expansion)
    Stop after NappH expansions.
    """
    current = set(seed_dets)
    for _ in range(NappH):
        conn1 = qc.get_connections_one_body(list(current), one_body_terms)
        conn2 = qc.get_connections_two_body(list(current), two_body_terms)
        current |= set(conn1)
        current |= set(conn2)
    return sorted(list(current))

def build_sector_basis_from_seeds(seeds_wf, one_body_terms, two_body_terms, NappH, coeff_thresh=1e-14):
    """
    seeds_wf: list of seed Wavefunctions for a given particle sector.
    1) collect support determinants from all seeds
    2) expand that set by NappH applications of H
    """
    if not seeds_wf:
        return []
    seed_support = set()
    for wf in seeds_wf:
        seed_support |= wavefunction_support(wf, coeff_thresh=coeff_thresh)
    return expand_basis_by_H(seed_support, one_body_terms, two_body_terms, NappH)

# -----------------------------
# Hamiltonian restricted to a fixed determinant basis
# -----------------------------

def build_H_in_basis(basis_dets, h0_clean, U_clean):
    """
    Use your fast OpenMP builder on the restricted basis.
    Returns a scipy.spmatrix (CSR).
    """
    if len(basis_dets) == 0:
        return sp.csr_matrix((0,0), dtype=np.complex128)
    H = qc.build_hamiltonian_openmp(basis_dets, h0_clean, U_clean)
    return H

# -----------------------------
# Block Lanczos on a fixed basis with dense/sparse @
# -----------------------------

def gram_schmidt_qr_dense(block_V, reorth=False, eps=1e-24):
    """
    Gram-Schmidt QR on a list of dense vectors (numpy arrays), returns (Q_list, R).
    Q_list contains orthonormal dense vectors (same length as v).
    """
    n = len(block_V)
    if n == 0:
        return [], np.array([[]], dtype=np.complex128)
    # Copy inputs
    V = [v.astype(np.complex128, copy=True) for v in block_V]
    Q = []
    R = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        vj = V[j]
        for i, qi in enumerate(Q):
            hij = np.vdot(qi, vj)
            R[i, j] = hij
            vj -= hij * qi
        if reorth:
            # one extra pass
            for i, qi in enumerate(Q):
                hij = np.vdot(qi, vj)
                R[i, j] += hij
                vj -= hij * qi
        nrm2 = np.vdot(vj, vj).real
        if nrm2 > eps:
            nrm = np.sqrt(nrm2)
            R[j, j] = nrm
            Q.append(vj / nrm)
        else:
            break
    r = len(Q)
    return Q, R[:r, :n]

def block_lanczos_fixed_basis(H, seed_vecs, L, reorth=False):
    """
    Standard block-Lanczos where H is a matrix in the fixed basis (dense or sparse),
    and seed_vecs is a list of dense vectors (already in that basis).
    Returns As, Bs, Qblocks, R0, where:
      - Qblocks[k] is a list of orthonormal vectors (columns) at block k
      - As[k], Bs[k] are the small block matrices like in your previous routine
    """
    Q0, R0 = gram_schmidt_qr_dense(seed_vecs, reorth=reorth)
    if len(Q0) == 0:
        return [], [], [], R0

    # Convert block list to column-block matrix helpers
    def block_to_mat(Qblock):
        return np.column_stack(Qblock) if len(Qblock) else np.zeros((H.shape[0], 0), dtype=np.complex128)

    Qblocks = [Q0]
    As, Bs = [], []

    Qk = block_to_mat(Q0)
    HQk = H @ Qk
    Ak = Qk.conj().T @ HQk
    As.append(Ak)

    Qkm1 = np.zeros((H.shape[0], 0), dtype=np.complex128)  # empty previous block

    for k in range(L):
        # W = H Qk - Qk Ak - Q_{k-1} B_{k-1}^H
        W = HQk - Qk @ Ak
        if k > 0:
            W -= Qkm1 @ Bs[-1].conj().T

        # Orthonormalize W
        W_cols = [W[:, j].copy() for j in range(W.shape[1])]
        Qnext_list, Bk = gram_schmidt_qr_dense(W_cols, reorth=reorth)
        if len(Qnext_list) == 0:
            break

        Bs.append(Bk)
        Qkm1 = Qk
        Qk = np.column_stack(Qnext_list)

        # Update Ak+1
        HQk = H @ Qk
        Ak = Qk.conj().T @ HQk
        As.append(Ak)

        Qblocks.append(Qnext_list)

        if len(Qblocks) >= L + 1:
            break

    return As, Bs, Qblocks, R0

# -----------------------------
# Continued fraction for top-left block (unchanged)
# -----------------------------

def block_cf_top_left(As, Bs, z):
    if not As:
        return np.array([[]], dtype=np.complex128)
    m0 = As[0].shape[0]
    Id = np.eye(m0, dtype=np.complex128)
    Sigma = np.zeros_like(As[-1], dtype=np.complex128)
    for k in range(len(As)-2, -1, -1):
        Bkp1 = Bs[k]
        Ikp1 = np.eye(As[k+1].shape[0], dtype=np.complex128)
        try:
            Sigma = Bkp1.conj().T @ npl.inv(z*Ikp1 - As[k+1] - Sigma) @ Bkp1
        except npl.LinAlgError:
            return np.full_like(Id, np.nan)
    return npl.inv(z*Id - As[0] - Sigma)

# -----------------------------
# Top-level: fixed-basis block-Lanczos Green's function
# -----------------------------

def green_function_block_lanczos_fixed_basis(
    M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
    h0_clean, U_clean, one_body_terms, two_body_terms, coeff_thresh=1e-12, L=100, reorth=False
):
    """
    For each sector (N+1 and N-1):
      1) Build seed wavefunctions by applying creation/annihilation on psi0
      2) Build fixed Krylov determinant basis by NappH H-expansions
      3) Build H_in_basis once
      4) Run block-Lanczos using plain matrix multiplications with H_in_basis
      5) Assemble G(ω) on the subspace of selected impurity indices, embedded back
         into the full spin-orbital space (size 2M), like your previous routine.
    """
    Norb = 2*M
    Nw = len(ws)

    # 1) seeds in each sector
    seed_add_wf = []
    seed_rem_wf = []
    for i in impurity_indices:
        si = qc.Spin.Alpha if i < M else qc.Spin.Beta
        oi = i % M
        wa = qc.apply_creation(psi0_wf, oi, si)
        if wa.data():  # non-empty
            seed_add_wf.append(wa)
        wr = qc.apply_annihilation(psi0_wf, oi, si)
        if wr.data():
            seed_rem_wf.append(wr)

    # 2) fixed determinant bases
    basis_add = build_sector_basis_from_seeds(seed_add_wf, one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh)
    basis_rem = build_sector_basis_from_seeds(seed_rem_wf, one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh)

    # 3) build H in those bases
    H_add = build_H_in_basis(basis_add, h0_clean, U_clean) if len(basis_add) else sp.csr_matrix((0,0), dtype=np.complex128)
    H_rem = build_H_in_basis(basis_rem, h0_clean, U_clean) if len(basis_rem) else sp.csr_matrix((0,0), dtype=np.complex128)

    # 4) initial block vectors Q0 in each sector: project seeds to the fixed bases
    seed_vecs_add = [wf_to_vec(wf, basis_add) for wf in seed_add_wf] if len(basis_add) else []
    seed_vecs_rem = [wf_to_vec(wf, basis_rem) for wf in seed_rem_wf] if len(basis_rem) else []

    # Run block-Lanczos on the fixed bases
    As_g, Bs_g, Qs_g, R0_g = ([], [], [], np.array([]))
    As_l, Bs_l, Qs_l, R0_l = ([], [], [], np.array([]))

    if len(seed_vecs_add):
        # convert CSR to linear operator by dense/sparse @ in the iteration
        As_g, Bs_g, Qs_g, R0_g = block_lanczos_fixed_basis(H_add, seed_vecs_add, L=L, reorth=reorth)

    if len(seed_vecs_rem):
        As_l, Bs_l, Qs_l, R0_l = block_lanczos_fixed_basis(H_rem, seed_vecs_rem, L=L, reorth=reorth)

    have_g = (R0_g.size != 0 and len(As_g) > 0)
    have_l = (R0_l.size != 0 and len(As_l) > 0)

    # Indices in the *small* blocks correspond one-to-one to the seed order
    add_idx_map = list(range(len(seed_vecs_add)))
    rem_idx_map = list(range(len(seed_vecs_rem)))

    # 5) Evaluate G(ω) in the seed subspace and embed back to [Norb x Norb] using impurity_indices
    G_all = np.zeros((Nw, Norb, Norb), dtype=np.complex128)

    for iw, w in enumerate(ws):
        z_g = (w + e0) + 1j*eta
        z_l = (-w + e0) - 1j*eta

        Gg_eff = None
        if have_g:
            G00_g = block_cf_top_left(As_g, Bs_g, z_g)
            if G00_g.size != 0 and not np.isnan(G00_g).any():
                # Note: here R0_g is the QR R factor. The "impurity block" in this fixed-basis variant
                #       corresponds to the seed subspace, so the correct coupling into CF top-left is
                #       exactly like before: G_eff = R^H G00 R.
                Gg_eff = R0_g.conj().T @ G00_g @ R0_g

        Gl_eff = None
        if have_l:
            G00_l = block_cf_top_left(As_l, Bs_l, z_l)
            if G00_l.size != 0 and not np.isnan(G00_l).any():
                Gl_eff = R0_l.conj().T @ G00_l @ R0_l

        # place back into the full Norb x Norb with the seed ordering matching impurity_indices
        # seed ordering for addition/removal is exactly the order we built seeds_wf with
        if Gg_eff is not None:
            for a, ia in enumerate(impurity_indices):
                for b, ib in enumerate(impurity_indices):
                    if a < len(add_idx_map) and b < len(add_idx_map):
                        G_all[iw, ia, ib] += Gg_eff[a, b]

        if Gl_eff is not None:
            for a, ia in enumerate(impurity_indices):
                for b, ib in enumerate(impurity_indices):
                    if a < len(rem_idx_map) and b < len(rem_idx_map):
                        G_all[iw, ia, ib] -= Gl_eff[a, b]

    return G_all, dict(
        basis_add_size=len(basis_add), basis_rem_size=len(basis_rem),
        have_g=have_g, have_l=have_l
    )    

# ==================

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
    nb = 9
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
    
    ws = np.linspace(-6, 6, 1001)

    # Precompute operator terms once (you already do this)
    one_body_terms = get_one_body_terms(h0_clean, M)
    two_body_terms = get_two_body_terms(U_clean, M)

    NappH = 1   # for example; 0 means "just the seed support"
    L = 100
    eta = 0.02
    impurity_indices = [0, M]  # same as before

    print(f"\nRunning MATRIX-FREE BLOCK Lanczos for impurity orbitals {impurity_indices}...")
    t_start = time.time()

    G_block, meta = green_function_block_lanczos_fixed_basis(
        M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
        h0_clean, U_clean, one_body_terms, two_body_terms,
        coeff_thresh=1e-12, L=L, reorth=False
    )
    print("Fixed-basis sizes:", meta)

    
   
    t_end = time.time()
    print(f"  Calculation finished in {t_end-t_start:.3f}s")
    A_block = -(1/np.pi) * np.imag(G_block)
    
   
    
    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    for (i,ii) in enumerate(impurity_indices):
        plt.plot(ws, i*np.max(A_block[:,ii,ii])+(A_block[:, ii, ii]), label="A_bl_"+str(i)+"(ω)")
    plt.title("Impurity Spectral Function for Anderson Model (Matrix-Free Lanczos)")
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nScript finished. Check the plot for the spectral function.")