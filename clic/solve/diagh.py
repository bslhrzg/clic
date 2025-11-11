import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh  # or eigs if complex non-Hermitian
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from ..ops.ops import get_ham
from .. import clic_clib as cc
from .davidson import davidson_linearop
from clic.basis.basis_Np import partition_by_Sz
from clic.ops import ops


def diagH_(basis,h0,U,M,num_roots,vguess = 'hf', option='matvec',sh_full=None):
    
    if sh_full is None:
        sh_full = cc.build_screened_hamiltonian(h0, U, 1e-10)
    
    sh_fb   = cc.build_fixed_basis_tables(sh_full, basis, M)

    
    A_csr_native = cc.build_fixed_basis_csr(sh_fb, basis, h0, U)

    if option == 'native':
        A = csr_matrix((np.asarray(A_csr_native.data),
                        np.asarray(A_csr_native.indices, dtype=np.int64),
                        np.asarray(A_csr_native.indptr, dtype=np.int64)),
                    shape=(A_csr_native.N, A_csr_native.N))

    if option == 'matvec':
        def matvec(x):
            return cc.csr_matvec(A_csr_native, np.asarray(x, dtype=np.complex128))
        A = LinearOperator((A_csr_native.N, A_csr_native.N), matvec=matvec, dtype=np.complex128)

   
    #evals, evecs = eigsh(A, k=num_roots, which="SA", ncv=min(2*num_roots+1, len(basis)))
    evals, evecs = davidson_linearop(A, num_roots=num_roots, max_subspace=128, block_size=16, tol=1e-10)

    return evals,evecs


def diagH(basis, h0, U, M, num_roots, vguess='hf', option='csr', sh_full=None):

    if option == 'csr':  # builds the matrix once, then matvec
        if sh_full is None:
            sh_full = cc.build_screened_hamiltonian(h0, U, 1e-10)
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

def get_roots_(basis0,h0,U,nroots,method='buildH'):
    if len(basis0) <= 64:
        H = ops.get_ham(basis0, h0, U)
        evals, evecs = eigh(H.toarray())
    else:
        H = ops.get_ham(basis0, h0, U)
        evals, evecs = eigsh(H, k=nroots, which='SA')
    indsort = np.argsort(np.real(evals))
    evals=evals[indsort]
    evecs=evecs[:,indsort]
    return evals[:nroots], evecs[:, :nroots]

def get_roots(basis0, h0, U, nroots,method='buildH'):
    """
    Calculates the lowest energy eigenstates by block-diagonalizing the Hamiltonian
    in S_z sectors.

    This is more efficient than building and diagonalizing the full Hamiltonian
    as it solves smaller eigenvalue problems for each block.

    Parameters
    ----------
    basis0 : list[SlaterDeterminant]
        The full list of Slater determinants in the basis.
    h0 : np.ndarray
        One-body part of the Hamiltonian (hopping terms).
    U : float
        Two-body interaction strength (Hubbard U).
    nroots : int
        The number of lowest-energy eigenstates to return.

    Returns
    -------
    evals : np.ndarray
        The `nroots` lowest eigenvalues, sorted.
    evecs : np.ndarray
        The corresponding eigenvectors, with columns ordered by eigenvalue.
        Each eigenvector is expressed in the original `basis0`.
    """
    if not basis0:
        return np.array([]), np.array([[]])

    # 1. Partition the full basis into S_z blocks
    inds_by_sz, sz_blocks = partition_by_Sz(basis0)
    
    print(f"Found {len(sz_blocks)} S_z blocks: {sz_blocks}")

    all_evals = []
    all_evecs_in_full_basis = []
    full_basis_size = len(basis0)

    # 2. Loop over each block, build its Hamiltonian, and diagonalize it
    for sz, sub_indices in zip(sz_blocks, inds_by_sz):
        sub_basis_size = len(sub_indices)
        if sub_basis_size == 0:
            continue
        
        print(f"  Processing Sz={sz} block of size {sub_basis_size}x{sub_basis_size}")

        # Create the sub-basis for this specific S_z block
        sub_basis = [basis0[i] for i in sub_indices]

        # Build the smaller Hamiltonian for this block only
        H_block = ops.get_ham(sub_basis, h0, U)
        
        # Decide on diagonalization method based on the BLOCK size
        # This is much more efficient than checking the full basis size
        if sub_basis_size <= 64:
            # For small blocks, full diagonalization is fast and easy
            try:
                # If H_block is sparse, convert to dense
                H_block_dense = H_block.toarray()
            except AttributeError:
                H_block_dense = H_block
            evals_block, evecs_block = eigh(H_block_dense)
        else:
            # For larger blocks, use an iterative solver
            # We need to find at most nroots, but cannot ask for more roots
            # than the matrix dimension minus one for eigsh.
            k = min(nroots, sub_basis_size - 1)
            evals_block, evecs_block = eigsh(H_block, k=k, which='SA')

        # 3. Reconstruct eigenvectors in the original full basis
        for i in range(len(evals_block)):
            # Start with a zero vector of the full basis size
            evec_full = np.zeros(full_basis_size, dtype=complex)
            
            # Get the eigenvector from the block diagonalization
            evec_sub = evecs_block[:, i]
            
            # "Scatter" the values from the sub-eigenvector into the
            # correct positions in the full-basis eigenvector.
            # This is the crucial step.
            evec_full[sub_indices] = evec_sub
            
            # 4. Collect results
            all_evals.append(evals_block[i])
            all_evecs_in_full_basis.append(evec_full)

    # 5. Sort all the collected eigenvalues and eigenvectors globally
    indsort = np.argsort(all_evals)
    
    # Convert list of arrays to a 2D numpy array and sort
    sorted_evals = np.array(all_evals)[indsort]
    # np.stack creates a matrix from the list of vectors before sorting columns
    sorted_evecs = np.stack(all_evecs_in_full_basis, axis=1)[:, indsort]
    
    # Return only the requested number of roots
    return sorted_evals[:nroots], sorted_evecs[:, :nroots]