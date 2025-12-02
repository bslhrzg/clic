# diagh.py
import numpy as np
from scipy.sparse.linalg import eigsh  # or eigs if complex non-Hermitian
from scipy.sparse import csr_matrix, diags
from scipy.linalg import eigh
import clic_clib as cc
from .davidson import davidson
from clic.basis.basis_Np import partition_by_Sz
from time import time
from scipy.sparse import issparse


def get_ham(basis,h0,U,
            tables=None, verbose=False):

    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    U = np.ascontiguousarray(U, dtype=np.complex128)

    #if method == "0":
    #    t0 = time()
    #    H = cc.build_hamiltonian_openmp(basis, h0, U)
    #    t1 = time()
    #    print(f"DEBUG: build ham time = {t1-t0}")

    #elif method == "1":
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
    if verbose:
        print(f"DEBUG: build ham time = {t1-t0}")
    #else : 
    #    print("method not implemented yet")
    #    assert 1==2
    return H 


def diagH(H, num_roots, option="auto", **kwargs):
    """
    Diagonalize H and return the lowest `num_roots` eigenpairs.

    Parameters
    ----------
    H : (N, N) array_like or sparse matrix
        Hamiltonian matrix.
    num_roots : int
        Number of lowest eigenvalues to compute (clipped to matrix size).
    option : {"auto", "dense", "arpack", "davidson"}
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

    if option == "auto":
        if dim < 512:
            option = "dense"
        else : 
            option = "davidson"

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
            #evals, evecs = eigsh(H, k=k, which=which, **kwargs)
            try:
                evals, evecs = eigsh(H, k=k, which=which, **kwargs)
            except ArpackNoConvergence as err:
                # If ARPACK already has enough converged roots, use them:
                print("DEBUG: Arpack failed, reverting to davidson")
                if (err.eigenvalues is not None and
                    len(err.eigenvalues) >= num_roots and
                    err.eigenvectors is not None):
                    evals = err.eigenvalues[:num_roots]
                    evecs = err.eigenvectors[:, :num_roots]
                else:
                    # Fallback: use Davidson without reusing ARPACK-specific kwargs
                    diag_vec = (
                        H.diagonal() if hasattr(H, "diagonal") else np.diag(H)
                    )
                    evals, evecs = davidson(
                        H,
                        num_roots=num_roots,
                        max_subspace=120,
                        tol=1e-10,
                        diag=diag_vec,
                        block_size=2 * num_roots,
                        verbose=False,
                    )

    elif option == "davidson":
        # sensible defaults, can be overridden via kwargs
        davidson_defaults = dict(
            num_roots=num_roots,
            max_subspace=120,
            tol=1e-10,
            diag=H.diagonal() if hasattr(H, "diagonal") else np.diag(H),
            block_size=2*num_roots,
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
    h0, U,
    tables,
    nroots,
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
        H_block = get_ham(sub_basis,h0,U,tables=tables)

        # we cannot ask for more roots than block dimension
        nroots_block = min(nroots, sub_size)
        if nroots_block == 0:
            continue

        evals_block, evecs_block = diagH(
            H_block, nroots_block, option="auto", **diag_kwargs
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


def get_roots(
        basis0, 
        h0, U, 
        nroots, 
        do_Sz = True,
        tables=None,
        diag_kwargs=None, 
        verbose = False):
    """
    Wrapper: diagonalize the Hubbard Hamiltonian in S_z blocks and
    return the lowest `nroots` eigenpairs expressed in the full basis.
    """
    if diag_kwargs is None:
        diag_kwargs = {}

    if len(basis0) == 0:
        return np.array([]), np.zeros((0, 0), dtype=complex)
    

    if tables is None:
        toltables = 1e-12
        tables = cc.build_hamiltonian_tables(h0,U,toltables)
     

    # Partition basis into S_z blocks
    if do_Sz : 
        inds_by_sz, sz_blocks = partition_by_Sz(basis0)
        if verbose:
            print(f"Found {len(sz_blocks)} S_z blocks: {sz_blocks}")

    
        # Optional: debug sizes per block
        if verbose:
            for sz, sub_indices in zip(sz_blocks, inds_by_sz):
                print(f"  Sz={sz}, block size {len(sub_indices)}")

        evals, evecs = diagonalize_by_blocks(
            basis0=basis0,
            blocks=inds_by_sz,
            h0=h0,U=U,
            tables=tables,
            nroots=nroots,
            diag_kwargs=diag_kwargs,
            dtype=complex,
        )
    else : 

        H = get_ham(basis0,h0,U,tables=tables)
        evals,evecs = diagH(H, nroots, option="auto", **diag_kwargs)


    return evals, evecs