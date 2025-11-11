import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.linalg import eigh

from scipy.sparse.linalg import aslinearoperator, LinearOperator


def _build_linear_operator(A):
    if hasattr(A, "shape") and hasattr(A, "dot"):
        return aslinearoperator(A)
    raise TypeError("A must be a scipy sparse matrix or LinearOperator")


def _get_diag(A):
    if issparse(A):
        d = A.diagonal()
        return np.array(d).reshape(-1)
    if hasattr(A, "diagonal"):
        return np.array(A.diagonal()).reshape(-1)
    return None


def _result_dtype(A, v0, diag):
    dts = []
    if getattr(A, "dtype", None) is not None:
        dts.append(np.dtype(A.dtype))
    if v0 is not None:
        dts.append(np.asarray(v0).dtype)
    if diag is not None:
        dts.append(np.asarray(diag).dtype)
    dts.append(np.float64)
    return np.result_type(*dts)


def _block_orth(X, B=None, tol=1e-12):
    """
    Orthonormalize columns of X against columns of B (if any) using
    block projection + QR. Returns Qx with ortho columns; may return
    zero columns if X lies in span(B).
    """
    if B is not None and B.size:
        G = B.conj().T @ X
        X = X - B @ G
    # QR on the block
    if X.size == 0:
        return np.empty((0, 0), dtype=X.dtype)
    Qx, R = np.linalg.qr(X, mode="reduced")
    # drop numerically rank deficient columns
    diagR = np.abs(np.diag(R))
    keep = diagR > tol * max(1.0, diagR.max(initial=0.0))
    if not np.any(keep):
        return np.empty((X.shape[0], 0), dtype=X.dtype)
    return Qx[:, keep]


def davidson(
    A,
    num_roots=1,
    max_subspace=128,
    max_iter=1000,
    tol=1e-10,
    v0=None,
    diag=None,
    block_size=16,
    verbose=False,
):
    """
    Block Davidson for Hermitian A with fast block orth and SpMM-friendly kernels.
    Returns (eigvals, eigvecs) with eigvecs as columns.
    """
    LA = _build_linear_operator(A)
    n = LA.shape[0]

    if diag is None:
        diag = _get_diag(A)
    if diag is not None and np.iscomplexobj(diag):
        if np.max(np.abs(diag.imag)) <= 1e-12 * max(1.0, np.max(np.abs(diag.real))):
            diag = diag.real

    work_dtype = _result_dtype(A, v0, diag)

    # initial block
    if v0 is None:
        V0 = np.random.randn(n, max(1, min(block_size, num_roots))).astype(work_dtype, copy=False)
    else:
        V0 = v0 if v0.ndim == 2 else v0.reshape(-1, 1)
        V0 = V0.astype(work_dtype, copy=False)

    Q = _block_orth(V0, None)
    if Q.shape[1] == 0:
        raise ValueError("Initial guess produced an empty orthonormal basis")

    AQ = (LA @ Q).astype(work_dtype, copy=False)

    locked_vals = []
    locked_vecs = np.empty((n, 0), dtype=work_dtype)

    it = 0
    while len(locked_vals) < num_roots and it < max_iter:
        it += 1

        # Rayleigh–Ritz
        T = Q.conj().T @ AQ
        T = 0.5 * (T + T.conj().T)
        w_all, y_all = eigh(T)
        order = np.argsort(w_all)
        w_all = w_all[order]
        y_all = y_all[:, order]

        take = max(num_roots, 1)
        w = w_all[:take]
        y = y_all[:, :take]
        U = Q @ y
        AU = AQ @ y

        # residuals and soft locking
        ritz_to_expand = []
        for i in range(take):
            r = AU[:, i] - w[i] * U[:, i]
            rn = norm(r)
            if verbose:
                print(f"iter {it:3d} root {i:2d} λ={w[i]:.12e} ||r||={rn:.3e}")
            if rn < tol:
                locked_vals.append(float(np.real(w[i])))
                locked_vecs = np.hstack([locked_vecs, U[:, [i]]])
            else:
                ritz_to_expand.append((i, r))

        if len(locked_vals) >= num_roots:
            break

        # anticipate restart before expansion
        n_add = min(block_size, max(1, num_roots - len(locked_vals)))
        if Q.shape[1] + n_add > max_subspace:
            keep = min(max(num_roots + 2 * block_size, 32), max_subspace // 2)
            yk = y_all[:, :keep]
            Qc = Q @ yk
            # block reorth against locked
            B = locked_vecs if locked_vecs.size else None
            Q = _block_orth(Qc, B)
            if Q.shape[1] == 0:
                Q = Qc  # fallback
            AQ = (LA @ Q).astype(work_dtype, copy=False)
            if verbose:
                print(f"Restart to subspace dim {Q.shape[1]}")

            # refresh Ritz for the restarted subspace
            T = Q.conj().T @ AQ
            T = 0.5 * (T + T.conj().T)
            w_all, y_all = eigh(T)
            order = np.argsort(w_all)
            w_all = w_all[order]
            y_all = y_all[:, order]
            w = w_all[:take]
            y = y_all[:, :take]
            U = Q @ y
            AU = AQ @ y

        # build one block of expansion directions
        b = min(block_size, len(ritz_to_expand)) or 1
        D = np.empty((n, 0), dtype=work_dtype)
        if b > 0:
            cols = []
            for i, r in ritz_to_expand[:b]:
                if diag is not None:
                    denom = diag - w[i]
                    eps = np.finfo(float).eps * max(1.0, float(np.abs(w[i])))
                    tiny = np.abs(denom) < eps
                    if np.any(tiny):
                        denom = denom.astype(np.complex128 if np.iscomplexobj(denom) or np.iscomplexobj(w) else denom.dtype, copy=True)
                        denom[tiny] = eps
                    t = r / denom
                else:
                    t = r
                cols.append(t.astype(work_dtype, copy=False))
            D = np.column_stack(cols)

        # orthonormalize directions in a single block against [Q, locked]
        B = Q if locked_vecs.size == 0 else np.hstack([Q, locked_vecs])
        D = _block_orth(D, B)
        if D.shape[1] == 0:
            # emergency random block
            D = np.random.randn(n, 1).astype(work_dtype, copy=False)
            D = _block_orth(D, B)
            if D.shape[1] == 0:
                if verbose:
                    print("Stalled. Exiting.")
                break

        # single SpMM expansion
        AD = LA @ D
        Q = np.hstack([Q, D])
        AQ = np.hstack([AQ, AD])

    # finalize from current subspace if needed
    if len(locked_vals) < num_roots:
        T = Q.conj().T @ AQ
        T = 0.5 * (T + T.conj().T)
        w_all, y_all = eigh(T)
        order = np.argsort(w_all)
        w_all = w_all[order]
        y_all = y_all[:, order]
        need = num_roots - len(locked_vals)
        U = Q @ y_all[:, :need]
        locked_vals.extend(list(np.real(w_all[:need])))
        locked_vecs = np.hstack([locked_vecs, U])

    order = np.argsort(locked_vals)[:num_roots]
    eigvals = np.array(locked_vals, dtype=float)[order]
    eigvecs = locked_vecs[:, order]
    return eigvals, eigvecs


def davidson_linearop(
    A,
    num_roots=1,
    max_subspace=128,
    max_iter=200,
    tol=1e-10,
    v0=None,
    diag=None,          # ndarray diagonal for Davidson t = r / (diag - lambda)
    prec=None,          # callable: prec(X) returns approx M^{-1} X (block)
    block_size=16,
    verbose=False,
):
    """
    Block Davidson for Hermitian problems with a LinearOperator.

    Parameters
    ----------
    A : LinearOperator or anything accepted by aslinearoperator
        Hermitian operator (n x n). Only matvecs are required.
    num_roots : int
        Number of lowest eigenpairs to compute.
    max_subspace : int
        Max dimension of the working subspace before restart.
    max_iter : int
        Max outer iterations.
    tol : float
        Residual norm tolerance per Ritz vector.
    v0 : ndarray or None
        Initial guesses, shape (n,) or (n, k0). If None, random block.
    diag : ndarray or None
        Diagonal of A for the classic Davidson preconditioner.
    prec : callable or None
        Preconditioner apply: Y = prec(X). If provided, used instead of diag.
        X and Y are (n, b) blocks.
    block_size : int
        Number of expansion directions per outer iteration.
    verbose : bool
        Print per-iteration residuals.

    Returns
    -------
    eigenvalues : (num_roots,) ndarray
    eigenvectors : (n, num_roots) ndarray  (columns are eigenvectors)
    """
    LA = aslinearoperator(A)
    n = LA.shape[0]

    # dtype bookkeeping
    work_dtype = _result_dtype(LA, v0, diag)

    # handle nearly-real diagonals: keep real if tiny imag
    if diag is not None and np.iscomplexobj(diag):
        if np.max(np.abs(diag.imag)) <= 1e-12 * max(1.0, np.max(np.abs(diag.real))):
            diag = diag.real

    # initial block
    if v0 is None:
        V0 = np.random.randn(n, max(1, min(block_size, num_roots))).astype(work_dtype, copy=False)
    else:
        V0 = v0 if v0.ndim == 2 else v0.reshape(-1, 1)
        V0 = V0.astype(work_dtype, copy=False)

    Q = _block_orth(V0, None)
    if Q.shape[1] == 0:
        raise ValueError("Initial guess produced an empty orthonormal basis")
    AQ = (LA @ Q).astype(work_dtype, copy=False)

    locked_vals = []
    locked_vecs = np.empty((n, 0), dtype=work_dtype)

    it = 0
    while len(locked_vals) < num_roots and it < max_iter:
        it += 1

        # Rayleigh–Ritz
        T = Q.conj().T @ AQ
        T = 0.5 * (T + T.conj().T)
        w_all, y_all = eigh(T)
        order = np.argsort(w_all)
        w_all = w_all[order]
        y_all = y_all[:, order]

        take = max(num_roots, 1)
        w = w_all[:take]
        y = y_all[:, :take]
        U = Q @ y
        AU = AQ @ y

        # residuals and soft-locking
        unconverged = []
        for i in range(take):
            r = AU[:, i] - w[i] * U[:, i]
            rn = norm(r)
            if verbose:
                print(f"iter {it:3d} root {i:2d} λ={w[i]:.12e} ||r||={rn:.3e}")
            if rn < tol:
                locked_vals.append(float(np.real(w[i])))
                locked_vecs = np.hstack([locked_vecs, U[:, [i]]])
            else:
                unconverged.append((i, r))

        if len(locked_vals) >= num_roots:
            break

        # anticipate restart before expansion
        n_add = min(block_size, max(1, num_roots - len(locked_vals)))
        if Q.shape[1] + n_add > max_subspace:
            keep = min(max(num_roots + 2 * block_size, 32), max_subspace // 2)
            yk = y_all[:, :keep]
            Qc = Q @ yk
            B = locked_vecs if locked_vecs.size else None
            Q = _block_orth(Qc, B)
            if Q.shape[1] == 0:
                Q = Qc
            AQ = (LA @ Q).astype(work_dtype, copy=False)

            # refresh Ritz on restarted subspace
            T = Q.conj().T @ AQ
            T = 0.5 * (T + T.conj().T)
            w_all, y_all = eigh(T)
            order = np.argsort(w_all)
            w_all = w_all[order]
            y_all = y_all[:, order]
            w = w_all[:take]
            y = y_all[:, :take]
            U = Q @ y
            AU = AQ @ y
            if verbose:
                print(f"Restart to subspace dim {Q.shape[1]}")

        # build corrections as a block
        b = min(block_size, len(unconverged)) or 1
        if b == 0:
            # emergency random direction
            D = np.random.randn(n, 1).astype(work_dtype, copy=False)
        else:
            Rcols = [r.astype(work_dtype, copy=False) for _, r in unconverged[:b]]
            R = np.column_stack(Rcols)

            # apply preconditioner: either user-supplied `prec` or Davidson diag
            if prec is not None:
                Tcorr = prec(R)  # should return shape (n, b)
                if Tcorr.shape != R.shape:
                    raise ValueError("prec(X) must return an array of same shape as X")
                D = Tcorr.astype(work_dtype, copy=False)
            elif diag is not None:
                # diag preconditioner column-wise: t_j = r_j / (diag - lambda_j)
                D = np.empty_like(R, dtype=np.result_type(work_dtype, np.complex128 if np.iscomplexobj(diag) else work_dtype))
                for j, (i_root, _) in enumerate(unconverged[:b]):
                    denom = diag - w[i_root]
                    eps = np.finfo(float).eps * max(1.0, float(np.abs(w[i_root])))
                    tiny = np.abs(denom) < eps
                    if np.any(tiny):
                        denom = denom.astype(np.complex128 if np.iscomplexobj(denom) or np.iscomplexobj(w) else denom.dtype, copy=True)
                        denom[tiny] = eps
                    D[:, j] = R[:, j] / denom
                D = D.astype(work_dtype, copy=False)
            else:
                D = R  # unpreconditioned

        # block orth against current basis and locked ones
        B = Q if locked_vecs.size == 0 else np.hstack([Q, locked_vecs])
        D = _block_orth(D, B)
        if D.shape[1] == 0:
            # try a random direction if everything got projected out
            D = np.random.randn(n, 1).astype(work_dtype, copy=False)
            D = _block_orth(D, B)
            if D.shape[1] == 0:
                if verbose:
                    print("Stalled. Exiting.")
                break

        # single SpMM expansion
        AD = LA @ D
        Q = np.hstack([Q, D])
        AQ = np.hstack([AQ, AD])

    # finalize if needed
    if len(locked_vals) < num_roots:
        T = Q.conj().T @ AQ
        T = 0.5 * (T + T.conj().T)
        w_all, y_all = eigh(T)
        order = np.argsort(w_all)
        w_all = w_all[order]
        y_all = y_all[:, order]
        need = num_roots - len(locked_vals)
        U = Q @ y_all[:, :need]
        locked_vals.extend(list(np.real(w_all[:need])))
        locked_vecs = np.hstack([locked_vecs, U])

    order = np.argsort(locked_vals)[:num_roots]
    eigvals = np.array(locked_vals, dtype=float)[order]
    eigvecs = locked_vecs[:, order]
    return eigvals, eigvecs

# ------------------ quick self test ------------------
dotest=False 
if dotest:
    # 1D Laplacian, diagonally dominant tridiagonal
    from scipy.sparse import diags

    n = 10_000
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1) #+ 1e-3 * np.random.randn(n-1)
    randscale = 1e-2 
    random_pot = randscale * np.random.randn(n)

    H = diags([off, main + random_pot, off], offsets=[-1, 0, 1], format="csr")

    from time import time 

    t0=time()
    d = H.diagonal()

    k = 6
    w, v = davidson(
        H,
        num_roots=k,
        max_subspace=120,
        tol=1e-10,
        diag=d,
        block_size=2*k,
        verbose=False
    )
    t1=time()

    from scipy.sparse.linalg import eigsh

    t2=time()
    w1, v1 = eigsh(H, k=k, which='SA', ncv=max(4*k, 40), tol=1e-10, maxiter=1000)
    t3=time()

    print("davidson eigenvalues:", w)
    print("arpack eigenvalues: ",w1)
    print(f"diff with arpack : {np.sort(w)-np.sort(w1)}")
    for i in range(4):
        r = H @ v[:, i] - w[i] * v[:, i]
        print(f"||r_{i}|| = {norm(r):.3e}")

    print(f"davidson time : {t1-t0}")
    print(f"arpack time: {t3-t2}")



    def mv(x):
        # your matvec here
        # e.g., 1D Laplacian with random off-diagonals (toy):
        y = np.zeros_like(x)
        y[0] = 2*x[0] - x[1]
        y[-1] = 2*x[-1] - x[-2]
        y[1:-1] = 2*x[1:-1] - x[0:-2] - x[2:]
        return y

    Aop = LinearOperator((n, n), matvec=mv, rmatvec=mv, dtype=np.float64)

    t0 = time()
    w, V = davidson_linearop(Aop, num_roots=k, max_subspace=128, block_size=16, tol=1e-10)
    t1 = time()
    print("lindav eigenvalues: ",w)
    print(f"diff with arpack : {np.sort(w)-np.sort(w1)}")

    print(f"lindav time : {t1-t0}")
