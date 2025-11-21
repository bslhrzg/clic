# gfs.py
import numpy as np
import clic_clib as cc
import scipy.sparse as sp
import numpy.linalg as npl
from numpy.linalg import norm

from .green_utils import *
from ..lanczos.block_lanczos import * 
from ..lanczos.scalar_lanczos import *

# -----------------------------
# Continued fraction for top-left block
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


def scalar_cf_from_T(T, z):
    """
    Evaluate the (0,0) element of the resolvent (zI - T)^{-1}
    for a tridiagonal matrix T using a scalar continued fraction.
    """
    alphas = np.diag(T).real
    betas_sq = np.diag(T, 1).real**2
    
    # Backward recurrence for g_k = z - alpha_k - beta_k^2 / g_{k+1}
    g = np.complex128(z - alphas[-1])
    for k in range(len(alphas) - 2, -1, -1):
        # Add a small epsilon to avoid division by zero if g is pathologically small
        g = z - alphas[k] - betas_sq[k] / (g + 1e-100j)
    return 1.0 / (g + 1e-100j)


# -----------------------------
# Top-level: fixed-basis block-Lanczos Green's function
# -----------------------------

def green_function_block_lanczos_fixed_basis_(
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
        Q0_add = np.column_stack(seed_vecs_add)
        #s_g, Bs_g, Qs_g, R0_g = block_lanczos_fixed_basis(H_add, seed_vecs_add, L=L, reorth=reorth)
        As_g, Bs_g, _, R0_g = block_lanczos_matrix(
            H_add, r=Q0_add.shape[1], seed=Q0_add, max_steps=L, reorth=reorth
        )

    if len(seed_vecs_rem):
        #As_l, Bs_l, Qs_l, R0_l = block_lanczos_fixed_basis(H_rem, seed_vecs_rem, L=L, reorth=reorth)
        Q0_rem = np.column_stack(seed_vecs_rem)
        As_l, Bs_l, _, R0_l = block_lanczos_matrix(
            H_rem, r=Q0_rem.shape[1], seed=Q0_rem, max_steps=L, reorth=reorth
        )

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


def green_function_block_lanczos_fixed_basis(
    M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
    h0_clean, U_clean, one_body_terms, two_body_terms, 
    iws = None,
    coeff_thresh=1e-12, L=100, reorth=False
):
    """
    Calculates the Green's function block corresponding ONLY to the provided
    impurity_indices. Returns a dense matrix in the basis of those indices.
    """
    Nw = len(ws)
    num_imp = len(impurity_indices)

    if iws is not None : 
        Niw = len(iws)

    seed_add_wf, add_src_idx = [], []
    seed_rem_wf, rem_src_idx = [], []
    for i in impurity_indices:
        si = cc.Spin.Alpha if i < M else cc.Spin.Beta
        oi = i % M
        wa = cc.apply_creation(psi0_wf, oi, si)
        if wa.data():
            seed_add_wf.append(wa)
            add_src_idx.append(i)
        wr = cc.apply_annihilation(psi0_wf, oi, si)
        if wr.data():
            seed_rem_wf.append(wr)
            rem_src_idx.append(i)

    basis_add = build_sector_basis_from_seeds(seed_add_wf, one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh)
    basis_rem = build_sector_basis_from_seeds(seed_rem_wf, one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh)

    H_add = build_H_in_basis(basis_add, h0_clean, U_clean) if len(basis_add) else sp.csr_matrix((0,0), dtype=np.complex128)
    H_rem = build_H_in_basis(basis_rem, h0_clean, U_clean) if len(basis_rem) else sp.csr_matrix((0,0), dtype=np.complex128)

    seed_vecs_add = [wf_to_vec(wf, basis_add) for wf in seed_add_wf] if len(basis_add) else []
    seed_vecs_rem = [wf_to_vec(wf, basis_rem) for wf in seed_rem_wf] if len(basis_rem) else []
    
    As_g, Bs_g, Qs_g, R0_g = ([], [], [], np.array([]))
    As_l, Bs_l, Qs_l, R0_l = ([], [], [], np.array([]))
    if seed_vecs_add and not all(norm(v) < 1e-30 for v in seed_vecs_add):
        Q0_add = np.column_stack(seed_vecs_add)
        As_g, Bs_g, _, R0_g = block_lanczos_matrix(H_add, r=Q0_add.shape[1], seed=Q0_add, max_steps=L, reorth=reorth)
    if seed_vecs_rem and not all(norm(v) < 1e-30 for v in seed_vecs_rem):
        Q0_rem = np.column_stack(seed_vecs_rem)
        As_l, Bs_l, _, R0_l = block_lanczos_matrix(H_rem, r=Q0_rem.shape[1], seed=Q0_rem, max_steps=L, reorth=reorth)

    have_g = (R0_g.size != 0 and len(As_g) > 0)
    have_l = (R0_l.size != 0 and len(As_l) > 0)
    
    G = np.zeros((Nw, num_imp, num_imp), dtype=np.complex128)
    if iws is not None : 
        G_iws = np.zeros((Niw, num_imp, num_imp), dtype=np.complex128)

    # Create a map from the global spin-orbital index to its position (0, 1, 2...) 
    # within the impurity_indices list.
    impurity_map = {idx: i for i, idx in enumerate(impurity_indices)}

    for iw, w in enumerate(ws):
        z_g, z_l = (w + e0) + 1j*eta, (-w + e0) - 1j*eta
        
        # Calculate G_eff in the basis of SURVIVING seeds (this part is correct)
        Gg_eff = None
        if have_g:
            G00_g = block_cf_top_left(As_g, Bs_g, z_g)
            if G00_g.size != 0 and not np.isnan(G00_g).any():
                Gg_eff = R0_g.conj().T @ G00_g @ R0_g

        Gl_eff = None
        if have_l:
            G00_l = block_cf_top_left(As_l, Bs_l, z_l)
            if G00_l.size != 0 and not np.isnan(G00_l).any():
                Gl_eff = R0_l.conj().T @ G00_l @ R0_l
        
        # Place the results from the surviving seed basis into the impurity basis
        if Gg_eff is not None:
            for a, ia in enumerate(add_src_idx):      # 'a' is index in Gg_eff, 'ia' is global index
                for b, ib in enumerate(add_src_idx):  # 'b' is index in Gg_eff, 'ib' is global index
                    out_i = impurity_map[ia]          # Find where 'ia' lives in the output matrix
                    out_j = impurity_map[ib]          # Find where 'ib' lives in the output matrix
                    G[iw, out_i, out_j] += Gg_eff[a, b]

        if Gl_eff is not None:
            for a, ia in enumerate(rem_src_idx):
                for b, ib in enumerate(rem_src_idx):
                    out_i = impurity_map[ia]
                    out_j = impurity_map[ib]
                    G[iw, out_i, out_j] -= Gl_eff[a, b]

    if iws is not None : 
        for iiw, iw in enumerate(iws):
            z_g, z_l = (iw + e0), (-iw + e0) 
            
            # Calculate G_eff in the basis of SURVIVING seeds (this part is correct)
            Gg_eff = None
            if have_g:
                G00_g = block_cf_top_left(As_g, Bs_g, z_g)
                if G00_g.size != 0 and not np.isnan(G00_g).any():
                    Gg_eff = R0_g.conj().T @ G00_g @ R0_g

            Gl_eff = None
            if have_l:
                G00_l = block_cf_top_left(As_l, Bs_l, z_l)
                if G00_l.size != 0 and not np.isnan(G00_l).any():
                    Gl_eff = R0_l.conj().T @ G00_l @ R0_l
            
            # Place the results from the surviving seed basis into the impurity basis
            if Gg_eff is not None:
                for a, ia in enumerate(add_src_idx):      # 'a' is index in Gg_eff, 'ia' is global index
                    for b, ib in enumerate(add_src_idx):  # 'b' is index in Gg_eff, 'ib' is global index
                        out_i = impurity_map[ia]          # Find where 'ia' lives in the output matrix
                        out_j = impurity_map[ib]          # Find where 'ib' lives in the output matrix
                        G_iws[iiw, out_i, out_j] += Gg_eff[a, b]

            if Gl_eff is not None:
                for a, ia in enumerate(rem_src_idx):
                    for b, ib in enumerate(rem_src_idx):
                        out_i = impurity_map[ia]
                        out_j = impurity_map[ib]
                        G_iws[iiw, out_i, out_j] -= Gl_eff[a, b]

    if iws is None:
        return G, dict(
            basis_add_size=len(basis_add), basis_rem_size=len(basis_rem)
        )
    else : 
        return G, G_iws, dict(
            basis_add_size=len(basis_add), basis_rem_size=len(basis_rem)
        )

def green_function_scalar_fixed_basis(
    M, psi0_wf, e0, ws, eta, i, NappH,
    h0_clean, U_clean, one_body_terms, two_body_terms,
    iws = None, 
    coeff_thresh=1e-12, L=100, reorth=False
):
    """
    Compute the single-diagonal element G_ii(ω) using a specialized fixed-basis
    scalar Lanczos approach.
    """
    Norb = 2*M
    assert 0 <= i < Norb, "i must be in [0, 2M)"

    Nw = len(ws)
    Gii = np.zeros(Nw, dtype=np.complex128)

    # If provided, we will also return Gii on matsubara frequencies
    if iws is not None: 
        Niw = len(ws)
        Gii_iw = np.zeros(Niw, dtype=np.complex128)


    si = cc.Spin.Alpha if i < M else cc.Spin.Beta
    oi = i % M
    wf_add = cc.apply_creation(psi0_wf, oi, si)
    wf_rem = cc.apply_annihilation(psi0_wf, oi, si)

    # --- Addition (Particle) Sector ---
    have_g = False
    if wf_add.data():
        basis_add = build_sector_basis_from_seeds([wf_add], one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh)
        if len(basis_add) > 0:
            print(f"DEBUG: len(basis_add) = {len(basis_add)}")
            H_add = build_H_in_basis(basis_add, h0_clean, U_clean)
            seed_vec_add = wf_to_vec(wf_add, basis_add)
            
            # Use scalar Lanczos
            _, T_g, v0_norm_g = scalar_lanczos(H_add, seed_vec_add, L=L, reorth=reorth)

            if T_g.size > 0:
                have_g = True
                norm_sq_g = v0_norm_g**2
                for iw, w in enumerate(ws):
                    z_g = (w + e0) + 1j*eta
                    Gii[iw] += norm_sq_g * scalar_cf_from_T(T_g, z_g)

                if iws is not None : 
                    for iiw, iw in enumerate(iws):
                        z_g = (iw + e0)
                        Gii_iw[iiw] += norm_sq_g * scalar_cf_from_T(T_g, z_g)


    # --- Removal (Hole) Sector ---
    have_l = False
    if wf_rem.data():
        basis_rem = build_sector_basis_from_seeds([wf_rem], one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh)
        if len(basis_rem) > 0:
            print(f"DEBUG: len(basis_rem) = {len(basis_rem)}")
            H_rem = build_H_in_basis(basis_rem, h0_clean, U_clean)
            seed_vec_rem = wf_to_vec(wf_rem, basis_rem)
            
            # Use scalar Lanczos
            _, T_l, v0_norm_l = scalar_lanczos(H_rem, seed_vec_rem, L=L, reorth=reorth)

            if T_l.size > 0:
                have_l = True
                norm_sq_l = v0_norm_l**2
                for iw, w in enumerate(ws):
                    z_l = (e0 - w) - 1j*eta
                    Gii[iw] -= norm_sq_l * scalar_cf_from_T(T_l, z_l)

                if iws is not None : 
                    for iiw, iw in enumerate(iws):
                        z_g = (e0 - iw)
                        Gii_iw[iiw] -= norm_sq_l * scalar_cf_from_T(T_l, z_l)


    info = dict(
        basis_add_size=len(basis_add) if 'basis_add' in locals() else 0,
        basis_rem_size=len(basis_rem) if 'basis_rem' in locals() else 0,
        have_g=have_g, have_l=have_l,
        seed_nonzero=(have_g or have_l)
    )
    if iws is None: 
        return Gii, info
    else :
        return Gii, Gii_iw, info

# ---------------------------------------------------------------------------
# TIME PROPAGATION 
# ---------------------------------------------------------------------------
def make_time_grid_from_freq(ws, eta, Nt_min=512, Nt_max=8192):
    """
    Pick a real-time grid suitable for a given frequency grid ws and damping η.

    Returns
    -------
    ts : (Nt,) float
        Time points.
    dt : float
        Time step.
    """
    ws = np.asarray(ws, dtype=float)
    wmin, wmax = ws.min(), ws.max()
    span = max(wmax - wmin, 1e-6)

    # time scale from η and from Δω
    t_eta = 8.0 / max(eta, 1e-6)
    dw = np.median(np.diff(np.unique(np.round(ws, 12)))) if ws.size > 1 else span
    t_dw = 2.0 * np.pi / max(dw, 1e-6)
    t_max = max(t_eta, t_dw)

    # dt to avoid aliasing: dt ≲ π/(4 Ω_max)
    wabs = max(abs(wmin), abs(wmax), 1.0)
    dt_alias = np.pi / (4.0 * wabs)
    dt = min(dt_alias, t_max / Nt_max)

    Nt = int(min(max(np.ceil(t_max / dt), Nt_min), Nt_max))
    ts = dt * np.arange(Nt, dtype=float)
    return ts, dt

def time_overlaps_from_lanczos(T, Q, bra_vec, v0_norm, ts, sign):
    """
    Compute S(t) = <bra| e^{-i sign H t} |v0> using a Lanczos
    representation H ≈ Q T Q^† built from |v0>.

    Parameters
    ----------
    T : (m, m) ndarray
        Tridiagonal Lanczos matrix.
    Q : (dim, m) ndarray
        Orthonormal Lanczos basis.
    bra_vec : (dim,) ndarray
        Bra vector (in the full Hilbert space) to contract with.
    v0_norm : float
        Norm of the initial vector used to build the Lanczos basis.
    ts : (Nt,) ndarray
        Time grid.
    sign : +1 or -1
        +1 for e^{-i H t}, -1 for e^{+i H t} depending on convention.

    Returns
    -------
    S : (Nt,) complex ndarray
        Time-dependent overlaps.
    """
    if T is None or T.size == 0 or Q is None:
        return np.zeros_like(ts, dtype=np.complex128)

    # project bra onto Lanczos basis
    c = Q.conj().T @ bra_vec        # shape (m,)

    # spectral decomposition of T
    evals, U = np.linalg.eigh(T)    # T = U diag(evals) U^†

    # e1 in Krylov space
    m = T.shape[0]
    e1 = np.zeros(m, dtype=np.complex128)
    e1[0] = 1.0

    Udag_e1 = U.conj().T @ e1       # (m,)
    cdag_U  = np.conj(c) @ U        # (m,)

    # phases exp(-i sign λ t)  (note: -sign because H vs our earlier +i sign)
    phases = np.exp(-1j * sign * np.outer(evals, ts))  # (m, Nt)

    # sum over eigenmodes
    S = v0_norm * (cdag_U[:, None] * Udag_e1[:, None] * phases).sum(axis=0)
    return S


def green_from_time_overlaps(ws, eta, ts, S_add, S_rem):
    """
    Build retarded Green's function from time overlaps.

    G^R(ω) = -i ∫_0^∞ dt e^{i ω t - η t} [S_add(t) + S_rem(t)].

    Parameters
    ----------
    ws : (Nw,) ndarray
        Frequency grid.
    eta : float
        Damping parameter.
    ts : (Nt,) ndarray
        Time grid (uniform spacing).
    S_add, S_rem : (Nt,) ndarray
        Time overlaps from addition and removal sectors.

    Returns
    -------
    g : (Nw,) complex ndarray
        Retarded Green's function on ws.
    """
    ws = np.asarray(ws, dtype=float)
    ts = np.asarray(ts, dtype=float)
    dt = ts[1] - ts[0]

    S_total = S_add + S_rem
    phase = np.exp(1j * np.outer(ws, ts)) * np.exp(-eta * ts)[None, :]
    g = -1j * dt * (phase @ S_total)
    return g

def green_function_from_time_propagation(
    i, j,
    M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
    h0_clean, U_clean, one_body_terms, two_body_terms,
    coeff_thresh=1e-12, L=100, reorth=True
):
    ws = np.asarray(ws, dtype=float)
    Norb = 2*M
    assert 0 <= i < Norb and 0 <= j < Norb

    def _index_to_spin_orb(k, M):
        sk = cc.Spin.Alpha if k < M else cc.Spin.Beta
        ok = k % M
        return ok, sk

    oj, sj = _index_to_spin_orb(j, M)
    oi, si = _index_to_spin_orb(i, M)

    # seeds
    wf_add_j = cc.apply_creation(psi0_wf, oj, sj)
    wf_rem_j = cc.apply_annihilation(psi0_wf, oj, sj)
    have_add = bool(wf_add_j.data())
    have_rem = bool(wf_rem_j.data())
    if not have_add and not have_rem:
        return np.zeros_like(ws, dtype=np.complex128)

    wf_add_i = cc.apply_creation(psi0_wf, oi, si)
    wf_rem_i = cc.apply_annihilation(psi0_wf, oi, si)

    # build bases
    basis_add = build_sector_basis_from_seeds(
        [wf_add_j] if have_add else [],
        one_body_terms, two_body_terms,
        NappH,
        coeff_thresh=coeff_thresh
    )
    basis_rem = build_sector_basis_from_seeds(
        [wf_rem_j] if have_rem else [],
        one_body_terms, two_body_terms,
        NappH,
        coeff_thresh=coeff_thresh
    )

    # restricted H
    H_add = build_H_in_basis(basis_add, h0_clean, U_clean) if have_add and len(basis_add) else sp.csr_matrix((0,0), dtype=np.complex128)
    H_rem = build_H_in_basis(basis_rem, h0_clean, U_clean) if have_rem and len(basis_rem) else sp.csr_matrix((0,0), dtype=np.complex128)

    # shift by E0
    if H_add.shape[0] > 0:
        H_add = H_add - e0 * sp.eye(H_add.shape[0], dtype=np.complex128, format='csr')
    if H_rem.shape[0] > 0:
        H_rem = H_rem - e0 * sp.eye(H_rem.shape[0], dtype=np.complex128, format='csr')

    # project seeds
    a_j_vec = wf_to_vec(wf_add_j, basis_add) if have_add and H_add.shape[0] else np.zeros((0,), dtype=np.complex128)
    r_j_vec = wf_to_vec(wf_rem_j, basis_rem) if have_rem and H_rem.shape[0] else np.zeros((0,), dtype=np.complex128)
    a_i_vec = wf_to_vec(wf_add_i, basis_add) if have_add and H_add.shape[0] else np.zeros((0,), dtype=np.complex128)
    r_i_vec = wf_to_vec(wf_rem_i, basis_rem) if have_rem and H_rem.shape[0] else np.zeros((0,), dtype=np.complex128)

    # Lanczos on each sector
    Qp = Tp = n0p = None
    Qm = Tm = n0m = None
    if have_add and a_j_vec.size:
        Qp, Tp, n0p = scalar_lanczos(H_add, a_j_vec, L=L, reorth=reorth, symmetrize=True, scale=True)
    if have_rem and r_j_vec.size:
        Qm, Tm, n0m = scalar_lanczos(H_rem, r_j_vec, L=L, reorth=reorth, symmetrize=True, scale=True)

    # time grid
    ts, dt = make_time_grid_from_freq(ws, eta)

    # time overlaps
    S_add = time_overlaps_from_lanczos(Tp, Qp, a_i_vec, n0p, ts, sign=-1) if Qp is not None else np.zeros_like(ts, dtype=np.complex128)
    S_rem = time_overlaps_from_lanczos(Tm, Qm, r_i_vec, n0m, ts, sign=+1) if Qm is not None else np.zeros_like(ts, dtype=np.complex128)

    # Fourier → G(ω)
    g = green_from_time_overlaps(ws, eta, ts, S_add, S_rem)
    return g

def green_function_from_time_propagation_(
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
        Qp, Tp, n0p = scalar_lanczos(H_add, a_j_vec, L=L, reorth=True, symmetrize=True, scale=True)
    if have_rem and r_j_vec.size:
        Qm, Tm, n0m = scalar_lanczos(H_rem, r_j_vec, L=L, reorth=True, symmetrize=True, scale=True)

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
    print(f"DEBUG: in timeprop lanczos: dt = {dt}, len(ts) = {len(ts)}")

    # Helper: compute time overlaps S(t) = <bra| exp(i sign T t) e1 > * v0_norm
    def _time_overlaps_from_lanczos(T, Q, bra_vec, v0_norm, ts, sign):
        if T is None or T.size == 0:
            return np.zeros_like(ts, dtype=np.complex128)
        # Project bra onto Lanczos basis
        c = Q.conj().T @ bra_vec  # coefficients in Lanczos basis
        # Diagonalize small tridiagonal T
        evals, U = np.linalg.eigh(T)
        # e1 = (1,0,0,…)
        e1 = np.zeros((T.shape[0],), dtype=np.complex128)
        e1[0] = 1.0
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


def lanczos_time_evolution(
    H,
    psi0_vec,
    ts,
    L=100,
    reorth=True,
    symmetrize=True,
    scale=True,
):
    """
    Time evolution |psi(t)> = exp(-i H t) |psi0> using a single
    Lanczos/Krylov space built from |psi0>.

    Parameters
    ----------
    H : (dim, dim) array_like or csr_matrix
        Hamiltonian in the chosen sector (Hermitian).
    psi0_vec : (dim,) ndarray (complex)
        Initial state |psi(0)> in the same basis as H.
    ts : (Nt,) ndarray (float)
        Time grid in the same units as 1/energy in H.
    L : int
        Maximum number of Lanczos iterations (Krylov dimension).
    reorth : bool
        Reorthogonalize Lanczos vectors in `scalar_lanczos`.
    symmetrize, scale : bool
        Passed through to `scalar_lanczos` (as in your GF code).

    Returns
    -------
    psis_t : (Nt, dim) ndarray (complex)
        Wavefunctions at each time t in `ts`, so that
        psis_t[k, :] ≈ |psi(t_k)>.
    """

    ts = np.asarray(ts, dtype=float)
    dim = psi0_vec.shape[0]

    print(f"dim = {dim}")

    # Build Lanczos basis from |psi0>
    Q, T, v0_norm = scalar_lanczos(
        H, psi0_vec,
        L=L,
        reorth=reorth,
        symmetrize=symmetrize,
        scale=scale
    )

    m = T.shape[0]
    if m == 0:
        # degenerate case
        return np.tile(psi0_vec, (ts.size, 1))

    # Diagonalize the small tridiagonal matrix T
    evals, U = np.linalg.eigh(T)  # T = U diag(evals) U^†

    # e1 in Krylov space
    e1 = np.zeros((m,), dtype=np.complex128)
    e1[0] = 1.0

    # α = U^† e1
    alpha = U.conj().T @ e1   # shape (m,)

    # phases: exp(-i λ t_k), shape (m, Nt)
    phases = np.exp(-1j * np.outer(evals, ts))

    # coefficients in eigenbasis for each t: exp(-iλt) * α
    tmp = phases * alpha[:, None]            # (m, Nt)

    # back to Lanczos basis: coeffs(t) = U * tmp
    coeffs = U @ tmp                         # (m, Nt)

    # back to full Hilbert space: |psi(t)> = v0_norm * Q * coeffs(t)
    psis = v0_norm * (Q @ coeffs)            # (dim, Nt)

    # return as (Nt, dim)
    return psis.T

