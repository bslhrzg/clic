import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh  # or eigs if complex non-Hermitian
from scipy.sparse import csr_matrix, diags
from scipy.linalg import eigh
import clic_clib as cc
from .davidson import davidson_linearop,davidson
from clic.basis.basis_Np import partition_by_Sz
from time import time



def get_ham(basis,h0,U,method="1",tables=None):
    """TO ADD : DETECT SPIN FLIPS TERMS"""
    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    U = np.ascontiguousarray(U, dtype=np.complex128)

    if method == "0":
        t0 = time()
        H = cc.build_hamiltonian_openmp(basis, h0, U)
        t1 = time()
        print(f"DEBUG: build ham time = {t1-t0}")

    elif method == "1":
        t0 = time()
        M = np.shape(h0)[0] // 2
        if tables is None:
            toltables = 1e-12
            tables = cc.build_hamiltonian_tables(h0,U,toltables)
        
        tol_el = 1e-16
        ftables = cc.build_fixed_basis_tables(tables,basis,M)
        H = cc.build_hamiltonian_matrix_fixed_basis(
            ftables, basis, h0, U, tol_el)
        t1 = time()
        print(f"DEBUG: build ham time = {t1-t0}")
    else : 
        print("method not implemented yet")
        assert 1==2
    return H 



def diagH_(basis, h0, U, M, num_roots, vguess='hf', option='csr', sh_full=None):

    if option == 'csr':  # builds the matrix once, then matvec
        if sh_full is None:
            sh_full = cc.build_hamiltonian_tables(h0, U, 1e-10)
        sh_fb = cc.build_fixed_basis_tables(sh_full, basis, M)
        A_csr_native = cc.build_fixed_basis_csr(sh_fb, basis, h0, U)
        def matvec(x):
            return cc.csr_matvec(A_csr_native, np.asarray(x, dtype=np.complex128))

    elif option == 'onthefly':
        # build once
        print("DEBUG before Hop")
        Hop = cc.FixedBasisMatvec(basis, h0, U, enable_magnetic=False, tol=1e-12)
        print("DEBUG coucou")
        def matvec(x):
            x = np.asarray(x, dtype=np.complex128, order="C")
            return Hop.apply(x)
        

    elif option == 'wavefun':  # only if v is very sparse
        # convert dense vector -> sparse Wavefunction on the given basis
        assert 1==0,"not implemented yet"
        def matvec(x):
            psi = cc.Wavefunction(M)
            x = np.asarray(x, dtype=np.complex128)
            for i, ci in enumerate(x):
                if ci != 0:
                    psi.add_term(basis[i], ci, 0.0)
            phi = cc.apply_hamiltonian_fixed_basis(psi, sh_fb, basis, h0, U, 0.0)
            y = np.zeros(len(basis), dtype=np.complex128)
            for det, val in phi.data().items():
                j = 0 #basis_index[det]  # precompute basis_index dict once outside
                y[j] = val
            return y
    else:
        raise ValueError("option must be 'csr', 'onthefly', or 'wavefun'")

    A = LinearOperator((len(basis), len(basis)), matvec=matvec, dtype=np.complex128)
    #evals, evecs = eigsh(A, k=num_roots, which="SA", ncv=min(2*num_roots+1, len(basis)))
    evals, evecs = davidson_linearop(A, num_roots=num_roots, max_subspace=128, block_size=16, tol=1e-10)
    return evals, evecs


import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh

# assume davidson, get_ham, partition_by_Sz are imported/defined elsewhere


def diagH(H, num_roots, option="arpack", **kwargs):
    """
    Diagonalize H and return the lowest `num_roots` eigenpairs.

    Parameters
    ----------
    H : (N, N) array_like or sparse matrix
        Hamiltonian matrix.
    num_roots : int
        Number of lowest eigenvalues to compute (clipped to matrix size).
    option : {"dense", "arpack", "davidson"}
        Diagonalization method.
    **kwargs :
        Passed to the underlying solver:
          - arpack: forwarded to eigsh (e.g. which="SA", v0=...)
          - davidson: forwarded to davidson(...)
    
    Returns
    -------
    evals : (num_roots,) np.ndarray
    evecs : (N, num_roots) np.ndarray
    """
    dim = H.shape[0]
    if dim == 0:
        return np.array([]), np.zeros((0, 0), dtype=complex)

    # cannot ask more eigenpairs than dimension
    num_roots = min(num_roots, dim)

    if option == "dense":
        # convert sparse → dense if needed
        if issparse(H):
            H_dense = H.toarray()
        else:
            H_dense = np.asarray(H)
        evals, evecs = eigh(H_dense)
        evals = evals[:num_roots]
        evecs = evecs[:, :num_roots]

    elif option == "arpack":
        # eigsh requires k < N; if k == N, fall back to dense
        if num_roots >= dim:
            if issparse(H):
                H_dense = H.toarray()
            else:
                H_dense = np.asarray(H)
            evals, evecs = eigh(H_dense)
            evals = evals[:num_roots]
            evecs = evecs[:, :num_roots]
        else:
            k = num_roots
            which = kwargs.pop("which", "SA")  # smallest algebraic by default
            evals, evecs = eigsh(H, k=k, which=which, **kwargs)

    elif option == "davidson":
        # sensible defaults, can be overridden via kwargs
        davidson_defaults = dict(
            num_roots=num_roots,
            max_subspace=120,
            tol=1e-10,
            diag=H.diagonal() if hasattr(H, "diagonal") else np.diag(H),
            block_size=2 * num_roots,
            verbose=False,
        )
        davidson_defaults.update(kwargs)
        evals, evecs = davidson(H, **davidson_defaults)

    else:
        raise ValueError(f"Unknown diagonalization option: {option}")

    # Ensure global sorting of eigenpairs
    idx = np.argsort(evals)
    evals = np.array(evals)[idx]
    evecs = evecs[:, idx]

    return evals, evecs


def diagonalize_by_blocks(
    basis0,
    blocks,
    build_block_hamiltonian,
    nroots,
    method="arpack",
    max_dense_block=512,
    diag_kwargs=None,
    dtype=complex,
):
    """
    Generic block-diagonalization helper.

    Parameters
    ----------
    basis0 : sequence
        Full basis, only used for sizing / scattering eigenvectors.
    blocks : list[list[int] or np.ndarray]
        List of index arrays/lists, each defining one block in `basis0`.
    build_block_hamiltonian : callable
        Function taking `sub_basis` and returning the corresponding H_block.
        Signature: H_block = build_block_hamiltonian(sub_basis)
    nroots : int
        Number of lowest-energy eigenstates to return globally (across all blocks).
    method : {"dense", "arpack", "davidson"}
        Preferred diagonalization method for large blocks.
    max_dense_block : int
        Blocks with size <= max_dense_block use dense diagonalization.
    diag_kwargs : dict or None
        Extra keyword arguments forwarded to diagH.
    dtype : numpy dtype
        Dtype of global eigenvectors.

    Returns
    -------
    evals : (nroots,) np.ndarray
    evecs : (len(basis0), nroots) np.ndarray
    """
    if diag_kwargs is None:
        diag_kwargs = {}

    full_basis_size = len(basis0)
    if full_basis_size == 0:
        return np.array([]), np.zeros((0, 0), dtype=dtype)

    all_evals = []
    all_evecs = []

    for sub_indices in blocks:
        sub_indices = np.asarray(sub_indices, dtype=int)
        sub_size = len(sub_indices)
        if sub_size == 0:
            continue

        sub_basis = [basis0[i] for i in sub_indices]
        H_block = build_block_hamiltonian(sub_basis)

        # Decide method for this block
        if sub_size <= max_dense_block:
            block_method = "dense"
        else:
            block_method = method

        # we cannot ask for more roots than block dimension
        nroots_block = min(nroots, sub_size)
        if nroots_block == 0:
            continue

        evals_block, evecs_block = diagH(
            H_block, nroots_block, option=block_method, **diag_kwargs
        )

        # Scatter block eigenvectors into full basis
        for i in range(len(evals_block)):
            evec_full = np.zeros(full_basis_size, dtype=dtype)
            evec_full[sub_indices] = evecs_block[:, i]
            all_evals.append(evals_block[i])
            all_evecs.append(evec_full)

    if not all_evals:
        return np.array([]), np.zeros((full_basis_size, 0), dtype=dtype)

    all_evals = np.array(all_evals)
    all_evecs = np.stack(all_evecs, axis=1)  # (dim, nbasis_total)

    # global sort and truncate to requested nroots
    idx = np.argsort(all_evals)
    idx = idx[:nroots]

    sorted_evals = all_evals[idx]
    sorted_evecs = all_evecs[:, idx]

    return sorted_evals, sorted_evecs


def get_roots(basis0, h0, U, nroots, method="davidson", tables=None,
              max_dense_block=512, diag_kwargs=None):
    """
    Wrapper: diagonalize the Hubbard Hamiltonian in S_z blocks and
    return the lowest `nroots` eigenpairs expressed in the full basis.
    """
    if diag_kwargs is None:
        diag_kwargs = {}

    if len(basis0) == 0:
        return np.array([]), np.zeros((0, 0), dtype=complex)

    # Partition basis into S_z blocks
    inds_by_sz, sz_blocks = partition_by_Sz(basis0)
    print(f"Found {len(sz_blocks)} S_z blocks: {sz_blocks}")

    # Build callback for block Hamiltonian construction
    def build_block_hamiltonian(sub_basis):
        # you can of course change the "method" argument of get_ham if needed
        return get_ham(sub_basis, h0, U, method="1", tables=tables)

    # Optional: debug sizes per block
    for sz, sub_indices in zip(sz_blocks, inds_by_sz):
        print(f"  Sz={sz}, block size {len(sub_indices)}")

    evals, evecs = diagonalize_by_blocks(
        basis0=basis0,
        blocks=inds_by_sz,
        build_block_hamiltonian=build_block_hamiltonian,
        nroots=nroots,
        method=method,
        max_dense_block=max_dense_block,
        diag_kwargs=diag_kwargs,
        dtype=complex,
    )

    return evals, evecs