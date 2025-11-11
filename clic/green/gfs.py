# gfs.py
import numpy as np
from .. import clic_clib as cc
import scipy.sparse as sp
import numpy.linalg as npl
from numpy.linalg import norm

# -----------------------------
# Helpers: wavefunction <-> basis
# -----------------------------

def wavefunction_support(wf, coeff_thresh=1e-14):
    """
    Return the set of determinants in a Wavefunction with |coeff|>coeff_thresh.
    Assumes wf.data() is a mapping {SlaterDeterminant: complex}.
    """
    data = wf.data()
    return {det for det, c in data.items() if abs(c) > coeff_thresh}

def wf_to_vec(wf, basis_list):
    """
    Project wavefunction onto a given determinant basis ordering -> dense vector.
    """
    data = wf.data()
    idx = {det: k for k, det in enumerate(basis_list)}
    v = np.zeros(len(basis_list), dtype=np.complex128)
    for det, c in data.items():
        k = idx.get(det, None)
        if k is not None:
            v[k] += c
    return v

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
        conn1 = cc.get_connections_one_body(list(current), one_body_terms)
        conn2 = cc.get_connections_two_body(list(current), two_body_terms)
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
    H = cc.build_hamiltonian_openmp(basis_dets, h0_clean, U_clean)
    return H

# -----------------------------
# Block Lanczos on a fixed basis with dense/sparse @
# -----------------------------

def gram_schmidt_qr_dense(block_V, reorth=False, eps=1e-20):
    """
    Gram-Schmidt QR on a list of dense vectors (numpy arrays), returns (Q_list, R).
    Q_list contains orthonormal dense vectors (same length as v).
    R has shape (len(Q_list), n). If columns are dependent, len(Q_list) < n.
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
        if reorth and len(Q):
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

    for _ in range(L):
        # W = H Qk - Qk Ak - Q_{k-1} B_{k-1}^H
        W = HQk - Qk @ Ak
        if len(Bs) > 0:
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

    return As, Bs, Qblocks, R0

# -----------------------------
# Continued fraction for top-left block (unchanged)
# -----------------------------

def block_cf_top_left(As, Bs, z):
    """
    Evaluate the top left block of the block tridiagonal resolvent using a
    backward continued fraction with linear solves.
    """
    if not As:
        return np.array([[]], dtype=np.complex128)
    m0 = As[0].shape[0]
    Id0 = np.eye(m0, dtype=np.complex128)
    Sigma = np.zeros_like(As[-1], dtype=np.complex128)
    for k in range(len(As) - 2, -1, -1):
        Bkp1 = Bs[k]
        Ikp1 = np.eye(As[k+1].shape[0], dtype=np.complex128)
        try:
            # Solve (z I - A_{k+1} - Sigma) X = B rather than invert
            X = npl.solve(z * Ikp1 - As[k+1] - Sigma, Bkp1)
        except npl.LinAlgError:
            return np.full_like(Id0, np.nan)
        Sigma = Bkp1.conj().T @ X
    try:
        return npl.solve(z * Id0 - As[0] - Sigma, np.eye(m0, dtype=np.complex128))
    except npl.LinAlgError:
        return np.full_like(Id0, np.nan)

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

    # 1) seeds in each sector, keep maps to the global spin-orbital indices
    seed_add_wf, add_src_idx = [], []
    seed_rem_wf, rem_src_idx = [], []
    for i in impurity_indices:
        si = cc.Spin.Alpha if i < M else cc.Spin.Beta
        oi = i % M
        wa = cc.apply_creation(psi0_wf, oi, si)
        if wa.data():  # non-empty
            seed_add_wf.append(wa)
            add_src_idx.append(i)
        wr = cc.apply_annihilation(psi0_wf, oi, si)
        if wr.data():
            seed_rem_wf.append(wr)
            rem_src_idx.append(i)

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

    # Drop a sector if all projected seeds are numerically zero
    if seed_vecs_add and all(norm(v) < 1e-30 for v in seed_vecs_add):
        seed_vecs_add, add_src_idx, basis_add, H_add = [], [], [], sp.csr_matrix((0,0), dtype=np.complex128)
    if seed_vecs_rem and all(norm(v) < 1e-30 for v in seed_vecs_rem):
        seed_vecs_rem, rem_src_idx, basis_rem, H_rem = [], [], [], sp.csr_matrix((0,0), dtype=np.complex128)

    if len(seed_vecs_add):
        # convert CSR to linear operator by dense/sparse @ in the iteration
        As_g, Bs_g, Qs_g, R0_g = block_lanczos_fixed_basis(H_add, seed_vecs_add, L=L, reorth=reorth)

    if len(seed_vecs_rem):
        As_l, Bs_l, Qs_l, R0_l = block_lanczos_fixed_basis(H_rem, seed_vecs_rem, L=L, reorth=reorth)

    have_g = (R0_g.size != 0 and len(As_g) > 0)
    have_l = (R0_l.size != 0 and len(As_l) > 0)

    # Indices in the small blocks correspond one-to-one to the surviving seeds order
    # We preserved the mapping to the global spin-orbital indices in add_src_idx and rem_src_idx

    # 5) Evaluate G(ω) in the seed subspace and embed back to [Norb x Norb] using the preserved maps
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
        # but only for those seeds that actually survived
        if Gg_eff is not None:
            for a, ia in enumerate(add_src_idx):
                for b, ib in enumerate(add_src_idx):
                    G_all[iw, ia, ib] += Gg_eff[a, b]

        if Gl_eff is not None:
            for a, ia in enumerate(rem_src_idx):
                for b, ib in enumerate(rem_src_idx):
                    G_all[iw, ia, ib] -= Gl_eff[a, b]

    return G_all, dict(
        basis_add_size=len(basis_add), basis_rem_size=len(basis_rem),
        have_g=have_g, have_l=have_l
    )


def green_function_scalar_fixed_basis(
    M, psi0_wf, e0, ws, eta, i, NappH,
    h0_clean, U_clean, one_body_terms, two_body_terms,
    coeff_thresh=1e-12, L=100, reorth=False
):
    """
    Compute the single-diagonal element G_ii(ω) for a given spin-orbital index i (0..2M-1)
    using the fixed-basis block-Lanczos approach specialized to a scalar seed subspace.

    Returns
    -------
    Gii : np.ndarray of shape (Nw,), complex128
        The diagonal Green's function element G_ii(ω) for each ω in `ws`.
    info : dict
        Diagnostics: sizes of addition/removal bases, flags for having each sector, etc.
    """
    Norb = 2*M
    assert 0 <= i < Norb, "i must be in [0, 2M)"

    Nw = len(ws)
    Gii = np.zeros(Nw, dtype=np.complex128)

    # --- Build the two sector seeds only for index i
    si = cc.Spin.Alpha if i < M else cc.Spin.Beta
    oi = i % M

    wf_add = cc.apply_creation(psi0_wf, oi, si)
    wf_rem = cc.apply_annihilation(psi0_wf, oi, si)

    have_add_seed = bool(wf_add.data())
    have_rem_seed = bool(wf_rem.data())

    # Early exit if both seeds vanish (matrix element zero)
    if not have_add_seed and not have_rem_seed:
        return Gii, dict(
            basis_add_size=0, basis_rem_size=0,
            have_g=False, have_l=False,
            seed_nonzero=False
        )

    # --- Determinant bases by H-expansion from the single seeds
    basis_add = build_sector_basis_from_seeds(
        [wf_add] if have_add_seed else [],
        one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh
    )
    basis_rem = build_sector_basis_from_seeds(
        [wf_rem] if have_rem_seed else [],
        one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh
    )

    # --- Restrict H
    H_add = build_H_in_basis(basis_add, h0_clean, U_clean) if len(basis_add) else sp.csr_matrix((0,0), dtype=np.complex128)
    H_rem = build_H_in_basis(basis_rem, h0_clean, U_clean) if len(basis_rem) else sp.csr_matrix((0,0), dtype=np.complex128)

    # --- Project seeds into their bases (each is a single dense vector)
    seed_vecs_add = [wf_to_vec(wf_add, basis_add)] if have_add_seed and len(basis_add) else []
    seed_vecs_rem = [wf_to_vec(wf_rem, basis_rem)] if have_rem_seed and len(basis_rem) else []

    # Drop sector if the projected seed is numerically zero after expansion
    if seed_vecs_add and norm(seed_vecs_add[0]) < 1e-30:
        seed_vecs_add = []
    if seed_vecs_rem and norm(seed_vecs_rem[0]) < 1e-30:
        seed_vecs_rem = []

    # --- Block Lanczos in each sector (block size will be 1 if present)
    As_g, Bs_g, R0_g = [], [], np.array([])
    As_l, Bs_l, R0_l = [], [], np.array([])

    if seed_vecs_add:
        As_g, Bs_g, _, R0_g = block_lanczos_fixed_basis(H_add, seed_vecs_add, L=L, reorth=reorth)

    if seed_vecs_rem:
        As_l, Bs_l, _, R0_l = block_lanczos_fixed_basis(H_rem, seed_vecs_rem, L=L, reorth=reorth)

    have_g = (R0_g.size != 0 and len(As_g) > 0)
    have_l = (R0_l.size != 0 and len(As_l) > 0)

    # --- Evaluate scalar CFs and assemble G_ii(ω) = (R* G00 R)_{00}^add - (R* G00 R)_{00}^rem
    # For 1x1 blocks: G_eff = |R0|^2 * G00
    for iw, w in enumerate(ws):
        # particle sector: z_g = w + e0 + iη
        if have_g:
            z_g = (w + e0) + 1j*eta
            G00_g = block_cf_top_left(As_g, Bs_g, z_g)  # shape (1,1)
            if G00_g.size and not np.isnan(G00_g).any():
                r = R0_g[0, 0] if R0_g.ndim == 2 else R0_g
                Gii[iw] += (abs(r)**2) * G00_g[0, 0]

        # hole sector: z_l = -w + e0 - iη
        if have_l:
            z_l = (-w + e0) - 1j*eta
            G00_l = block_cf_top_left(As_l, Bs_l, z_l)  # shape (1,1)
            if G00_l.size and not np.isnan(G00_l).any():
                r = R0_l[0, 0] if R0_l.ndim == 2 else R0_l
                Gii[iw] -= (abs(r)**2) * G00_l[0, 0]

    return Gii, dict(
        basis_add_size=len(basis_add), basis_rem_size=len(basis_rem),
        have_g=have_g, have_l=have_l, seed_nonzero=True
    )


# ---------------------------------------------------------------------------
# TIME PROPAGATION 
# ---------------------------------------------------------------------------

def _lanczos_tridiagonal(H, v0, L=200, reorth=False):
    """
    Perform Hermitian Lanczos iteration starting from vector v0.
    Constructs an orthonormal Krylov basis Q and the tridiagonal matrix T.

    Parameters
    ----------
    H : (N×N) linear operator (csr_matrix or LinearOperator)
        Hermitian Hamiltonian, must support `@` for matrix-vector product.
    v0 : ndarray (N,)
        Initial vector to start the Krylov subspace.
    L : int
        Maximum number of Lanczos steps.
    reorth : bool
        If True, apply a (single-pass) reorthogonalization against all previous vectors
        to improve numerical stability.

    Returns
    -------
    Q : ndarray (N×m)
        Orthonormal Lanczos basis vectors.
    T : ndarray (m×m)
        Tridiagonal Lanczos matrix.
    v0_norm : float
        Norm of the initial vector v0.
    """

    # Ensure v0 is a complex128 array
    v0 = np.asarray(v0, dtype=np.complex128)
    N = v0.size

    # Normalize the initial vector
    v0_norm = np.linalg.norm(v0)
    if v0_norm == 0:
        # If v0 is zero, return empty basis and matrix
        return np.zeros((N,0), dtype=np.complex128), np.zeros((0,0), dtype=np.complex128), 0.0

    # Initialize first basis vector q = v0 / ||v0||
    q_prev = np.zeros_like(v0)   # "ghost" previous vector, initially zero
    q = v0 / v0_norm
    Qcols = [q.copy()]           # list to store basis vectors
    alphas, betas = [], []       # diagonals and off-diagonals of T

    beta = 0.0
    for _ in range(L):
        # Apply Hamiltonian
        w = H @ q
        # Rayleigh quotient α = <q|H|q>
        alpha = np.vdot(q, w)
        # Remove components along current and previous q (three-term recurrence)
        w = w - alpha * q - beta * q_prev

        # Optional reorthogonalization against all previous Q vectors
        if reorth and Qcols:
            Qmat = np.column_stack(Qcols)
            w -= Qmat @ (Qmat.conj().T @ w)

        # Next off-diagonal β = ||w||
        beta = np.linalg.norm(w)
        alphas.append(alpha)

        # If β ≈ 0, Krylov subspace has closed (breakdown)
        if beta < 1e-14:
            break

        # Store β and normalize new q
        betas.append(beta)
        q_prev, q = q, w / beta
        Qcols.append(q.copy())

    # Number of steps actually performed
    m = len(alphas)
    if m == 0:
        # Should not normally happen, but guard against empty Krylov space
        return np.zeros((N,0), dtype=np.complex128), np.zeros((0,0), dtype=np.complex128), v0_norm

    # Construct Q matrix (columns are q0, q1, …, q_{m-1})
    Q = np.column_stack(Qcols[:m])

    # Construct tridiagonal T from α and β coefficients
    T = np.zeros((m, m), dtype=np.complex128)
    for k in range(m):
        T[k, k] = alphas[k]      # diagonal entries
        if k+1 < m:
            T[k, k+1] = betas[k] # upper diagonal
            T[k+1, k] = betas[k] # lower diagonal

    return Q, T, v0_norm

def lanczos_tridiagonal_stable(H, v0, L=200, reorth=True, symmetrize=True, scale=True,
                               reorth_tol=1e-10, breakdown_tol=1e-14, powerit=20):
    """
    Hermitian Lanczos with optional symmetrization, spectral scaling, and two-pass MGS reorth.
    Returns Q (N×m), T (m×m), ||v0||.
    """

    # optional symmetrization
    if symmetrize:
        if sp.issparse(H):
            H = 0.5*(H + H.conj().T)
            H.setdiag(H.diagonal().real)
        else:
            H = 0.5*(H + H.conj().T)
            np.fill_diagonal(H, np.real(np.diag(H)))

    # estimate spectral radius for scaling
    rho = 1.0
    if scale:
        # power iteration on a random unit vector
        rng = np.random.default_rng(12345)
        x = rng.standard_normal(H.shape[0]) + 1j*rng.standard_normal(H.shape[0])
        x /= norm(x)
        n = 1.0
        for _ in range(powerit):
            x = H @ x
            n = norm(x)
            if not np.isfinite(n) or n < 1e-300:
                break
            x /= n
        rho = max(n, 1.0, 1e-12)
    Hdot = (lambda x: (H @ x) / rho) if scale else (lambda x: H @ x)

    v0 = np.asarray(v0, dtype=np.complex128)
    N = v0.size
    v0_norm = norm(v0)
    if v0_norm == 0.0:
        return np.zeros((N,0), np.complex128), np.zeros((0,0), np.complex128), 0.0

    q_prev = np.zeros_like(v0)
    q = v0 / v0_norm

    # preallocate Q with an upper bound of L+1 columns
    Q = np.empty((N, L+1), dtype=np.complex128)
    Q[:, 0] = q
    k = 0
    alphas = np.empty(L, dtype=np.float64)
    betas  = np.empty(L, dtype=np.float64)

    beta = 0.0
    while k < L:
        w = Hdot(q)
        alpha = np.vdot(q, w)
        # Hermitian guard: take the real part
        alpha = float(np.real(alpha))
        # three term recurrence
        w = w - alpha*q - beta*q_prev

        # selective reorth trigger
        if reorth:
            # test loss of orthogonality: max|Q[:,:k+1]^H w|
            hk = Q[:, :k+1].conj().T @ w
            maxlost = np.max(np.abs(hk))
            if maxlost > reorth_tol*norm(w):
                # two-pass MGS
                w -= Q[:, :k+1] @ hk
                hk2 = Q[:, :k+1].conj().T @ w
                w -= Q[:, :k+1] @ hk2

        beta = norm(w)
        alphas[k] = alpha

        if not np.isfinite(beta) or beta < breakdown_tol:
            # happy breakdown or numerical failure
            break

        q_prev, q = q, w / beta
        k += 1
        Q[:, k] = q
        betas[k-1] = beta

    m = max(1, k)  # at least one alpha
    Q = Q[:, :m]
    T = np.zeros((m, m), dtype=np.complex128)
    for i in range(m):
        # scale back the spectrum if we scaled H
        T[i, i] = alphas[i]*rho
        if i+1 < m:
            T[i, i+1] = betas[i]*rho
            T[i+1, i] = betas[i]*rho

    return Q, T, v0_norm

def green_function_from_time_propagation(
    i, j,
    M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
    h0_clean, U_clean, one_body_terms, two_body_terms,
    coeff_thresh=1e-12, L=100, reorth=True
):
    """
    Compute the retarded Green's function G_ij(ω) using time propagation + Lanczos.

    Strategy:
      - Build N+1 and N-1 sector bases starting from |a_j> = c_j†|ψ0> and |r_j> = c_j|ψ0>.
      - Restrict H to those bases and shift by E0 so ground state is stationary.
      - Run Lanczos on each sector to get tridiagonal T matrices.
      - Use T’s spectral decomposition to compute time overlaps S_add(t), S_rem(t).
      - Fourier-integrate with damping exp(-ηt) to get G_ij(ω).

    Parameters
    ----------
    i, j : int
        Indices of the Green's function G_ij(ω) to compute (0 ≤ i,j < 2M).
    M : int
        Number of spatial orbitals (so Norb = 2M including spin).
    psi0_wf : Wavefunction
        Ground state wavefunction.
    e0 : float
        Ground state energy.
    ws : ndarray
        Frequency grid.
    eta : float
        Broadening parameter (imaginary shift).
    impurity_indices : list[int]
        Indices of impurity orbitals (not directly needed here).
    NappH : int
        Number of H-applications to expand determinant basis.
    h0_clean, U_clean, one_body_terms, two_body_terms :
        Hamiltonian input data.
    coeff_thresh : float
        Threshold for determinant coefficients.
    L : int
        Maximum Lanczos steps.
    reorth : bool
        Reorthogonalize Lanczos vectors.

    Returns
    -------
    g : ndarray (len(ws),)
        Complex retarded Green's function values at given ω grid.
    """

    ws = np.asarray(ws, dtype=float)
    Norb = 2*M
    assert 0 <= i < Norb and 0 <= j < Norb

    # Helper: convert global index to (orbital, spin)
    def _index_to_spin_orb(i, M):
        si = cc.Spin.Alpha if i < M else cc.Spin.Beta
        oi = i % M
        return oi, si

    # Spin/orbital for indices i and j
    oj, sj = _index_to_spin_orb(j, M)
    oi, si = _index_to_spin_orb(i, M)

    # Seeds: |a_j> = c_j†|ψ0>, |r_j> = c_j|ψ0>
    wf_add_j = cc.apply_creation(psi0_wf, oj, sj)
    wf_rem_j = cc.apply_annihilation(psi0_wf, oj, sj)
    have_add = bool(wf_add_j.data()); have_rem = bool(wf_rem_j.data())
    if not have_add and not have_rem:
        # If both vanish, G_ij(ω) = 0
        return np.zeros_like(ws, dtype=np.complex128)

    # Bra seeds for overlap evaluation (|a_i>, |r_i>)
    wf_add_i = cc.apply_creation(psi0_wf, oi, si)
    wf_rem_i = cc.apply_annihilation(psi0_wf, oi, si)

    # Build determinant bases for N+1 and N-1 sectors by H-expansion
    print(f"DEBUG: coeff_thress in green : {coeff_thresh} NappH = {NappH}")
    basis_add = build_sector_basis_from_seeds(
        [wf_add_j] if have_add else [], one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh
    )
    basis_rem = build_sector_basis_from_seeds(
        [wf_rem_j] if have_rem else [], one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh
    )

    print(f"DEBUG: rem size : {len(basis_rem)}, add size : {len(basis_add)}")

    # Build restricted Hamiltonians
    H_add = build_H_in_basis(basis_add, h0_clean, U_clean) if have_add and len(basis_add) else sp.csr_matrix((0,0), dtype=np.complex128)
    H_rem = build_H_in_basis(basis_rem, h0_clean, U_clean) if have_rem and len(basis_rem) else sp.csr_matrix((0,0), dtype=np.complex128)

    print(f"DEBUG: Hamiltonian computed")

    # Shift by ground state energy e0 so the ground state is stationary
    if H_add.shape[0] > 0:
        H_add = H_add - e0 * sp.eye(H_add.shape[0], dtype=np.complex128, format='csr')
    if H_rem.shape[0] > 0:
        H_rem = H_rem - e0 * sp.eye(H_rem.shape[0], dtype=np.complex128, format='csr')

    # Project seeds into those bases
    a_j_vec = wf_to_vec(wf_add_j, basis_add) if have_add and H_add.shape[0] else np.zeros((0,), dtype=np.complex128)
    r_j_vec = wf_to_vec(wf_rem_j, basis_rem) if have_rem and H_rem.shape[0] else np.zeros((0,), dtype=np.complex128)
    a_i_vec = wf_to_vec(wf_add_i, basis_add) if have_add and H_add.shape[0] else np.zeros((0,), dtype=np.complex128)
    r_i_vec = wf_to_vec(wf_rem_i, basis_rem) if have_rem and H_rem.shape[0] else np.zeros((0,), dtype=np.complex128)

    # Lanczos on each sector starting from |a_j>, |r_j>
    Qp=Tp=n0p=None; Qm=Tm=n0m=None
    if have_add and a_j_vec.size:
        # stable variant with symmetrization and spectral scaling
        Qp, Tp, n0p = lanczos_tridiagonal_stable(H_add, a_j_vec, L=L, reorth=True, symmetrize=True, scale=True)
    if have_rem and r_j_vec.size:
        Qm, Tm, n0m = lanczos_tridiagonal_stable(H_rem, r_j_vec, L=L, reorth=True, symmetrize=True, scale=True)

    # Helper: pick time grid from ws and η
    def _compute_time_grid(ws, eta):
        ws = np.asarray(ws, dtype=float)
        wmin, wmax = ws.min(), ws.max()
        span = max(wmax - wmin, 1e-6)
        # target time scales: from η and from frequency spacing
        t_eta = 8.0 / max(eta, 1e-6)
        dw = np.median(np.diff(np.unique(np.round(ws, 12)))) if ws.size > 1 else span
        t_dw = 2.0*np.pi / max(dw, 1e-6)
        t_max = max(t_eta, t_dw)
        # dt to avoid aliasing: dt ≲ π/(4Ω)
        wabs = max(abs(wmin), abs(wmax), 1.0)
        dt_alias = np.pi / (4.0 * wabs)
        dt = min(dt_alias, t_max/4096.0)
        # clamp Nt between 512 and 8192
        Nt = int(min(max(np.ceil(t_max/dt), 512), 8192))
        ts = dt * np.arange(Nt, dtype=float)
        return ts, dt

    ts, dt = _compute_time_grid(ws, eta)

    # Helper: compute time overlaps S(t) = <bra| exp(i sign T t) e1 > * v0_norm
    def _time_overlaps_from_lanczos(T, Q, bra_vec, v0_norm, ts, sign):
        if T is None or T.size == 0:
            return np.zeros_like(ts, dtype=np.complex128)
        # Project bra onto Lanczos basis
        c = Q.conj().T @ bra_vec  # coefficients in Lanczos basis
        # Diagonalize small tridiagonal T
        evals, U = np.linalg.eigh(T)
        # e1 = (1,0,0,…)
        e1 = np.zeros((T.shape[0],), dtype=np.complex128); e1[0] = 1.0
        Udag_e1 = U.conj().T @ e1         # components of e1 in eigenbasis
        cdag_U  = np.conj(c) @ U          # bra projected in eigenbasis
        # Time evolution exp(i sign λ t) for each eigenvalue
        phases = np.exp(1j * sign * np.outer(evals, ts))   # (m, Nt)
        # Overlap sum
        return v0_norm * (cdag_U[:,None] * Udag_e1[:,None] * phases).sum(axis=0)

    # Time overlaps for addition (sign=-1) and removal (sign=+1)
    S_add = _time_overlaps_from_lanczos(Tp, Qp, a_i_vec, n0p, ts, sign=-1) if Qp is not None else np.zeros_like(ts, dtype=np.complex128)
    S_rem = _time_overlaps_from_lanczos(Tm, Qm, r_i_vec, n0m, ts, sign=+1) if Qm is not None else np.zeros_like(ts, dtype=np.complex128)

    # Fourier integration to G(ω): retarded integral with damping exp(-ηt)
    phase = np.exp(1j * np.outer(ws, ts)) * np.exp(-eta * ts)[None, :]
    g = -1j * dt * (phase @ (S_add + S_rem))  # -i factor for retarded GF

    return g