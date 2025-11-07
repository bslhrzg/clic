import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm

def scalar_lanczos(H, v0, L=200, reorth=True, symmetrize=True, scale=True,
                        reorth_tol=1e-10, breakdown_tol=1e-14, powerit=20, rng_seed=12345):
    """
    Hermitian Lanczos tridiagonalization with optional symmetrization, scaling,
    and selective two-pass reorthogonalization.

    Parameters
    ----------
    H : ndarray or scipy.sparse matrix (N,N) or LinearOperator
        Hermitian operator supporting '@'.
        If symmetrize=True, H is explicitly symmetrized.
    v0 : ndarray (N,)
        Starting vector.
    L : int
        Maximum number of Lanczos steps.
    reorth : bool
        Enable selective reorthogonalization (two-pass MGS when needed).
    symmetrize : bool
        Symmetrize H by 0.5*(H + H^H) and force real diagonal.
    scale : bool
        Scale the operator by an estimate of its spectral radius to improve stability.
    reorth_tol : float
        Trigger threshold for reorth: if max|Q^H w| > reorth_tol*||w||, perform two passes.
    breakdown_tol : float
        Stop if beta < breakdown_tol.
    powerit : int
        Power-iteration steps to estimate spectral radius when scale=True.
    rng_seed : int
        Seed for power-iteration start vector.

    Returns
    -------
    Q : ndarray (N,m)
        Orthonormal Lanczos basis (m <= L+1).
    T : ndarray (m,m)
        Real-symmetric tridiagonal matrix.
    v0_norm : float
        ||v0||_2.

    Notes
    -----
    - If scale=True, the returned T is rescaled back to the spectrum of H.
    - Produces real alphas and nonnegative betas for Hermitian H.
    """
    # optional symmetrization
    if symmetrize:
        if sp.issparse(H):
            H = 0.5 * (H + H.conj().T)
            H.setdiag(H.diagonal().real)
        else:
            H = 0.5 * (H + H.conj().T)
            d = np.diag(H).real
            np.fill_diagonal(H, d)

    # spectral scaling
    rho = 1.0
    if scale:
        rng = np.random.default_rng(rng_seed)
        if sp.issparse(H):
            N = H.shape[0]
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            x /= norm(x) if norm(x) > 0 else 1.0
            nlast = 1.0
            for _ in range(powerit):
                x = H @ x
                n = norm(x)
                if not np.isfinite(n) or n < 1e-300:
                    break
                x /= n
                nlast = n
            rho = max(nlast, 1.0, 1e-12)
        else:
            # dense path
            N = H.shape[0]
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            x /= norm(x) if norm(x) > 0 else 1.0
            nlast = 1.0
            for _ in range(powerit):
                x = H @ x
                n = norm(x)
                if not np.isfinite(n) or n < 1e-300:
                    break
                x /= n
                nlast = n
            rho = max(nlast, 1.0, 1e-12)

    Hdot = (lambda x: (H @ x) / rho) if scale else (lambda x: H @ x)

    v0 = np.asarray(v0, dtype=np.complex128)
    N = v0.size
    v0_norm = norm(v0)
    if v0_norm == 0.0:
        return np.zeros((N, 0), dtype=np.complex128), np.zeros((0, 0), dtype=np.complex128), 0.0

    q_prev = np.zeros_like(v0)
    q = v0 / v0_norm

    # preallocate
    Q = np.empty((N, L + 1), dtype=np.complex128)
    Q[:, 0] = q
    alphas = np.empty(L, dtype=np.float64)
    betas = np.empty(L, dtype=np.float64)

    beta = 0.0
    k = 0
    while k < L:
        w = Hdot(q)
        alpha = np.vdot(q, w)
        alpha = float(np.real(alpha))
        w = w - alpha * q - beta * q_prev

        if reorth:
            hk = Q[:, :k + 1].conj().T @ w
            maxlost = np.max(np.abs(hk)) if hk.size else 0.0
            if maxlost > reorth_tol * (norm(w) + 1e-300):
                w -= Q[:, :k + 1] @ hk
                hk2 = Q[:, :k + 1].conj().T @ w
                w -= Q[:, :k + 1] @ hk2

        beta = norm(w)
        alphas[k] = alpha

        if not np.isfinite(beta) or beta < breakdown_tol:
            break

        q_prev, q = q, w / beta
        k += 1
        Q[:, k] = q
        betas[k - 1] = beta

    m = max(1, k)
    Q = Q[:, :m]
    T = np.zeros((m, m), dtype=np.complex128)
    for i in range(m):
        T[i, i] = alphas[i] * rho
        if i + 1 < m:
            T[i, i + 1] = betas[i] * rho
            T[i + 1, i] = betas[i] * rho
    return Q, T, v0_norm


def scalar_lanczos_grid(x, mu, N, v_init=None, breakdown_tol=1e-14):
    """
    Scalar Lanczos on a weighted grid for the multiplication operator f(x) -> x f(x).

    Inner product is <a,b> = sum_s mu_s a_s b_s with mu_s >= 0.
    Produces the three-term recurrence coefficients for orthonormal polynomials
    with respect to the discrete measure {x_s, mu_s}.

    Parameters
    ----------
    x : ndarray (S,)
        Grid points.
    mu : ndarray (S,)
        Nonnegative weights defining the measure.
        Example: mu = w * rho where w are quadrature weights and rho is a density.
    N : int
        Number of Lanczos steps (size of T).
    v_init : ndarray (S,), optional
        Initial vector. If None, uses all-ones and normalizes under the weighted norm.
    breakdown_tol : float
        Stop if beta < breakdown_tol.

    Returns
    -------
    T : ndarray (N,N)
        Real-symmetric tridiagonal matrix with diagonals alpha and off-diagonals beta.
    v0_norm : float
        Weighted norm of the initial vector: sqrt(sum mu_s v_s^2).

    Notes
    -----
    This is the scalar analogue of the grid-based block routine, used to build
    continued-fraction or Gauss-type quadrature from {x, mu}.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    assert x.ndim == 1 and mu.ndim == 1 and x.shape == mu.shape
    S = x.size

    def ip(a, b):
        return float(np.dot(mu, a * b))

    def nrm(a):
        return np.sqrt(max(ip(a, a), 0.0))

    if v_init is None:
        v = np.ones(S, dtype=np.float64)
    else:
        v = np.asarray(v_init, dtype=np.float64).copy()

    v_im1 = np.zeros_like(v)
    v0n = nrm(v)
    if v0n == 0.0:
        return np.zeros((0, 0), dtype=np.float64), 0.0
    v /= v0n

    al = np.zeros(N, dtype=np.float64)
    be = np.zeros(N - 1, dtype=np.float64)

    beta_im1 = 0.0
    m = 0
    for j in range(N):
        w = x * v
        a = ip(v, w)
        al[j] = a
        w = w - a * v - beta_im1 * v_im1
        beta = nrm(w)
        if j < N - 1:
            be[j] = beta
        if beta < breakdown_tol:
            m = j + 1
            break
        v_im1, v = v, w / max(beta, 1e-300)
        beta_im1 = beta
        m = j + 1

    # build T of size m
    T = np.zeros((m, m), dtype=np.float64)
    idx = np.arange(m)
    T[idx, idx] = al[:m]
    if m > 1:
        j = np.arange(m - 1)
        T[j, j + 1] = be[:m - 1]
        T[j + 1, j] = be[:m - 1]
    return T, v0n