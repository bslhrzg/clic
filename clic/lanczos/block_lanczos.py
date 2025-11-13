# lanczos/bloc_lanczos.py

import numpy as np
import numpy.linalg as npl

# ---------------- helpers ----------------

def _psd_eigsqrt_and_pinv(G, tol=1e-12):
    """
    Compute the Hermitian positive-semidefinite (PSD) square root and its pseudo-inverse.

    Given a Hermitian matrix G ≽ 0 (typically a Gram matrix), this returns matrices:
        B       = G^{1/2}      (principal Hermitian square root)
        B_pinv  = G^{-1/2}     (Moore–Penrose pseudo-inverse of the square root)
        rank    = numerical rank of G

    Small or negative eigenvalues below a relative/absolute tolerance are clipped to zero,
    making this routine robust for nearly singular Gram matrices.

    Parameters
    ----------
    G : ndarray (n, n)
        Hermitian positive-semidefinite matrix.
    tol : float, optional
        Absolute lower bound for eigenvalue cutoff. Effective threshold is
        max(tol, max(spectrum(G)) * 1e-12).

    Returns
    -------
    B : ndarray (n, n)
        Hermitian square root of G (B @ B^H ≈ G).
    B_pinv : ndarray (n, n)
        Hermitian pseudo-inverse square root (B_pinv @ G @ B_pinv ≈ I on the support of G).
    rank : int
        Effective numerical rank of G after eigenvalue thresholding.

    Notes
    -----
    This is safer than a Cholesky decomposition for ill-conditioned or rank-deficient
    matrices and ensures B, B_pinv remain Hermitian PSD.
    """
    Gh = 0.5 * (G + G.conj().T)
    s, U = npl.eigh(Gh)
    s = np.clip(s, 0.0, None)
    thr = max(tol, (s.max() if s.size else 0.0) * 1e-12)
    mask = s > thr
    r = int(mask.sum())
    if r == 0:
        z = np.zeros_like(Gh)
        return z, z, 0
    Ur = U[:, mask]
    sr = s[mask]
    B = Ur @ np.diag(np.sqrt(sr)) @ Ur.conj().T
    B_pinv = Ur @ np.diag(1.0 / np.sqrt(sr)) @ Ur.conj().T
    return B, B_pinv, r

def _axpy_last(A, C):
    # A shape (..., bA), C shape (bA, k) -> (..., k)
    return np.tensordot(A, C, axes=([-1], [0]))

def _zeros_like_last(X, b):
    shape = list(X.shape)
    shape[-1] = b
    return np.zeros(shape, dtype=X.dtype)

def _truncate_last(X, k):
    return X[..., :k].copy()

def _orthonormalize_block(W, bases, gram, reorth=True, tol=1e-12):
    """
    Orthonormalize a block of vectors W against an existing set of orthonormal bases.

    This routine takes a block W (matrix or tensor whose last axis indexes columns),
    orthogonalizes it with respect to all blocks in `bases` using the inner product
    defined by the callable `gram(A, B)`, and normalizes it via the Hermitian square
    root of its local Gram matrix.

    Handles rank deficiency gracefully by projecting out linearly dependent directions.

    Parameters
    ----------
    W : ndarray
        Block of candidate vectors to be orthonormalized. Shape (..., b_W).
    bases : list of ndarrays
        List of previously orthonormalized blocks (same leading dimensions as W).
    gram : callable
        Function gram(A, B) -> ndarray (b_A, b_B) computing the Hermitian inner product.
        For the Euclidean case, this is A.conj().T @ B.
    reorth : bool, optional
        If True, perform a second reorthogonalization pass (recommended for stability).
    tol : float, optional
        Tolerance for numerical rank detection when normalizing.

    Returns
    -------
    Q : ndarray
        Orthonormalized block spanning the independent directions of W.
    B : ndarray
        Hermitian positive-definite square root of the Gram matrix of W (local overlap).
    rank : int
        Effective number of independent columns retained in Q.

    Notes
    -----
    The algorithm:
      1. Removes projections of W onto all previous bases (1 or 2 passes if reorth=True).
      2. Computes the local Gram matrix G = gram(W, W).
      3. Diagonalizes G to obtain its square root B = G^{1/2} and pseudo-inverse B^{-1}.
      4. Constructs Q = W @ B^{-1}, ensuring Q†Q = I on the support of G.
         Rank-deficient directions are automatically dropped.

    Used inside the block Lanczos iteration to keep each block orthonormal under
    an arbitrary inner product, even if near-linear dependencies appear.
    """
    # reorth against existing bases
    if bases:
        passes = 2 if reorth else 1
        for _ in range(passes):
            for P in bases:
                C = gram(P, W)           # shape (bP, bW)
                W = _axpy_last(P, -C) + W
    # local Gram and sqrt
    G = gram(W, W)
    B, B_pinv, r = _psd_eigsqrt_and_pinv(G, tol=tol)
    if r == 0:
        return None, None, 0
    Q = _axpy_last(W, B_pinv)  # W @ B_pinv
    # compress to rank r if needed
    if Q.shape[-1] != r:
        s, U = npl.eigh(0.5 * (G + G.conj().T))
        thr = max(tol, (s.max() if s.size else 0.0) * 1e-12)
        Uc = U[:, s > thr]  # bW x r
        Q = _axpy_last(Q, Uc)    # keep r independent dirs
        B = Uc.conj().T @ B @ Uc
    return Q, B, Q.shape[-1]

# ---------------- definitive generic block Lanczos ----------------

def block_lanczos(apply_op, gram, Q0, K=None, reorth=True, tol=1e-12, cap_dim=None):
    """
    Generic block Lanczos for Hermitian problems with a custom inner product.

    Inputs
        apply_op: function(X) -> same shape as X, operator application
        gram    : function(A, B) -> A^H * B in the chosen inner product, shape (bA, bB)
        Q0      : initial block, last axis indexes block columns
        K       : max iterations, default very large
        reorth  : do second-pass reorthogonalization
        tol     : Gram tolerance
        cap_dim : optional cap on total accumulated columns

    Returns
        A_blocks, B_blocks, Q_blocks, R0
        A_k has shape (b_k, b_k)
        B_k has shape (b_{k+1}, b_k)
        Q_blocks is a list of blocks
        R0 is the initial block sqrt Gram
    """
    if K is None:
        K = 10**9

    Q_blocks = []
    Q0c = Q0.copy()
    Q0o, R0, r0 = _orthonormalize_block(Q0c, [], gram, reorth=reorth, tol=tol)
    if r0 == 0:
        return [], [], [], None
    Q_blocks.append(Q0o)

    A_blocks, B_blocks = [], []
    Qkm1 = _zeros_like_last(Q0o, 0)
    Bkm1 = np.zeros((0, Q0o.shape[-1]), dtype=np.complex128)
    total_cols = Q0o.shape[-1]

    for _ in range(K):
        Qk = Q_blocks[-1]
        HQk = apply_op(Qk)
        Ak = gram(Qk, HQk)
        Ak = 0.5 * (Ak + Ak.conj().T)
        A_blocks.append(Ak)

        # W = H Qk - Qk Ak - Q_{k-1} B_{k-1}^H
        W = HQk + _axpy_last(Qk, -Ak)
        if Bkm1.size:
            W = W + _axpy_last(Qkm1, -Bkm1.conj().T)

        Qnext, Bk, rk = _orthonormalize_block(W, Q_blocks, gram, reorth=reorth, tol=tol)
        if rk == 0:
            break

        if cap_dim is not None and total_cols + rk > cap_dim:
            keep = max(0, cap_dim - total_cols)
            if keep == 0:
                break
            Qnext = _truncate_last(Qnext, keep)
            Bk = Bk[:keep, :]
            rk = keep

        B_blocks.append(Bk)
        Qkm1, Bkm1 = Qk, Bk
        Q_blocks.append(Qnext)
        total_cols += rk

    return A_blocks, B_blocks, Q_blocks, R0

# ---------------- convenience wrappers that replace variants ----------------
def block_lanczos_matrix(H, r, max_steps=None, seed=None, reorth=True, tol=1e-12):
    """
    Dense or sparse Hermitian H in C^{n x n}.
    Returns A_blocks, B_blocks, Q with Q shape n x m.
    """
    n = H.shape[0]
    if r <= 0:
        return [], [], np.eye(n)[:, :0]
    if max_steps is None:
        max_steps = int(np.ceil(n / r))

    Q0 = np.eye(n, r, dtype=np.complex128) if seed is None else np.asarray(seed, dtype=np.complex128)

    def apply_op(X):
        return H @ X

    def gram(A, B):
        return A.conj().T @ B

    A, B, Qblocks, R = block_lanczos(apply_op, gram, Q0, K=max_steps, reorth=reorth, tol=tol, cap_dim=n)
    Q = np.concatenate(Qblocks, axis=1) if Qblocks else np.zeros((n, 0), dtype=np.complex128)
    return A, B, Q, R

def block_lanczos_grid(x_vals, weight_mats, K, b, reorth=True, tol=1e-12):
    """
    Block Lanczos tridiagonalization for a weighted grid representation.

    Performs the block Lanczos algorithm where the operator acts as
    multiplication by x on a discrete grid, and the inner product is defined
    by position-dependent Hermitian weights μ_s at each grid point.

    This generates the block three-term recurrence for matrix-valued
    orthogonal polynomials Φ_k(x) satisfying:
        x Φ_k(x) = Φ_k(x) A_k + Φ_{k-1}(x) B_{k-1}^† + Φ_{k+1}(x) B_k

    Parameters
    ----------
    x_vals : ndarray (S,)
        Grid points representing the variable x (e.g. energy mesh).
    weight_mats : list[ndarray (M, M)]
        Hermitian positive-semidefinite weight matrices μ_s defining the
        inner product ⟨Φ, Ψ⟩ = Σ_s Φ(s)† μ_s Ψ(s).
    K : int
        Maximum number of Lanczos steps (recurrence depth).
    b : int
        Block size (number of initial orthogonal functions).
    reorth : bool, optional
        If True, perform a second reorthogonalization pass for numerical stability.
    tol : float, optional
        Tolerance for rank detection when orthonormalizing blocks.

    Returns
    -------
    A_blocks : list[ndarray (b_k, b_k)]
        Diagonal block matrices of the block-tridiagonal representation.
    B_blocks : list[ndarray (b_{k+1}, b_k)]
        Off-diagonal block matrices of the recurrence.
    Phi_blocks : list[ndarray (S, M, b_k)]
        List of orthonormal block functions Φ_k on the grid.
    M0 : ndarray (M, M)
        Total weight matrix M0 = Σ_s μ_s (symmetrized).

    Notes
    -----
    This routine is equivalent to constructing a block-tridiagonal representation
    of the multiplication operator x in a weighted inner-product space. The
    resulting (A, B) blocks can be used to approximate integrals or moments of
    spectral densities, or to build continued-fraction representations of
    Green's functions from grid data.
    """
    x = np.asarray(x_vals)
    mu = [np.asarray(W) for W in weight_mats]
    S = len(x)
    M = mu[0].shape[0]

    Phi0 = np.zeros((S, M, b), dtype=np.complex128)
    eyeMb = np.eye(M, b, dtype=np.complex128)
    for s in range(S):
        Phi0[s] = eyeMb

    mu_stack = np.stack(mu, axis=0)

    def apply_op(Phi):
        return x[:, None, None] * Phi

    def gram(Phi, Psi):
        acc = np.einsum('smb,smn,snk->bk', Phi.conj(), mu_stack, Psi, optimize=True)
        return 0.5 * (acc + acc.conj().T)

    A, B, Phi_blocks, _ = block_lanczos(apply_op, gram, Phi0, K=K, reorth=reorth, tol=tol, cap_dim=None)
    M0 = sum(mu).astype(np.complex128)
    M0 = 0.5 * (M0 + M0.conj().T)
    return A, B, Phi_blocks, M0