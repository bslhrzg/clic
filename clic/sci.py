# sci.py
from . import clic_clib as cc
from itertools import combinations
import numpy as np
from . import basis_Np,hamiltonians,ops,results
from scipy.sparse.linalg import eigsh 

def selective_ci(
    h0, U,
    M, Nelec,
    generator,
    selector,
    num_roots=1,
    one_bh=None,
    two_bh=None,
    max_iter=5,
    conv_tol=1e-6,
    prune_thr=1e-7,
    Nmul = None,
    verbose=True,
):
    """
    Generic selective-CI driver.

    Parameters
    ----------
    h0 : (K,K) array_like
        One-particle Hamiltonian (spin-orbital or spatial; your helpers decide).
    U  : (K,K,K,K) array_like
        Two-particle Hamiltonian (spin-orbital convention matching `get_*_terms`).
    M  : int
        Number of spatial orbitals.
    Nelec : int
        Total number of electrons.
    generator : callable
        Expansion proposal function. Must have signature:
            generator(wf, ewf, one_body_terms, two_body_terms, K, h0, U, thr=...)
        and return an *iterable of SlaterDeterminant* to add.
    num_roots : int 
        Number of eigenvectors computed and return at the FINAL iteration
    one_bh : list
        one body non zero terms of the hamiltonians, computed if not given 
    two_bh : list
        wo body ... 
    max_iter : int
        Maximum number of CI selection iterations.
    conv_tol : float
        Convergence threshold on energy change |E_new - E_old|.
    prune_thr : float
        Wavefunction prune threshold passed to `generator`.  
    Nmul : float 
        If positive, the size of the retained elements in generated basis is Nmul * len(current basis)
        If None, we keep the full generated basis.
    verbose : bool
        Print iteration logs.

    Returns
    -------
    result : dict
        {
          "energy": E0,
          "wavefunction": psi0,
          "basis": basis0,
          "energies": energies_per_iter,
          "sizes": sizes_per_iter,
        }

    Note: called with generator=cipsi_one_iter, max_iter=5, Nmul=None, this is CISD
    """

    if generator == hamiltonian_generator:
        # Precompute operator terms once
        if one_bh == None:
            one_bh = ops.get_one_body_terms(h0, M)
        if two_bh == None:
            two_bh = ops.get_two_body_terms(U, M)

    # Initial basis 
    basis0 = basis_Np.get_starting_basis(np.real(h0), Nelec)  # returns list of SlaterDeterminant

    # Initial Hamiltonian and ground state
    H = ops.get_ham(basis0, h0, U)
    # For small bases, a dense eig can be faster / safer; otherwise eigsh.
    dim0 = H.shape[0]
    if dim0 <= 64:
        from numpy.linalg import eigh
        evals, evecs = eigh(H.toarray())
        e0 = float(evals[0])
        psi0 = cc.Wavefunction(M, basis0, evecs[:, 0])
    else:
        evals, evecs = eigsh(H, k=1, which='SA')
        e0 = float(evals[0])
        psi0 = cc.Wavefunction(M, basis0, evecs[:, 0])

    if verbose:
        print(f"[init] dim={len(basis0)}  E0={e0:.12f}")

    energies = [e0]
    sizes = [len(basis0)]

    # Main selection loop
    for it in range(max_iter):
        # Propose new determinants using the provided generator

        lenb = len(basis0)

        gen_basis,hwf = generator(psi0,one_bh,two_bh,thr=prune_thr,return_hwf=True)
        selected_basis = selector(hwf,e0,gen_basis,2*M,h0,U)

        # If Nmul = 1.0, at each iteration we double the basis size 
        if Nmul != None : 
            nkeep = Nmul * lenb 
            nkeep = int(min(nkeep, len(selected_basis)))
            selected_basis = selected_basis[:nkeep]

        # Merge and sort the basis
        new_basis = set(basis0) | set(selected_basis)
        if len(new_basis) == len(basis0):
            if verbose:
                print(f"[iter {it}] no new determinants proposed; stopping.")
            break
        basis0 = sorted(list(new_basis))


        

        # Rebuild H and solve ground state
        H = ops.get_ham(basis0, h0, U)
        dim = H.shape[0]

        if dim <= 64:
            from numpy.linalg import eigh
            evals, evecs = eigh(H.toarray())
            e_new = float(evals[0])
            psi0 = cc.Wavefunction(M, basis0, evecs[:, 0])
        else:
            evals, evecs = eigsh(H, k=1, which='SA')
            e_new = float(evals[0])
            psi0 = cc.Wavefunction(M, basis0, evecs[:, 0])

        dE = abs(e_new - energies[-1])
        energies.append(e_new)
        sizes.append(dim)

        if verbose:
            print(f"[iter {it}] dim={dim:>6}  E0={e_new:.12f}  |dE|={dE:.3e}")

        if dE < conv_tol:
            if verbose:
                print(f"[conv] reached |dE| < {conv_tol}")
            break

    evals, evecs = eigsh(H, k=num_roots, which='SA')
    psis = [cc.Wavefunction(M, basis0, evecs[:, i]) for i in range(num_roots)]

    sci_res = results.NelecLowEnergySubspace(M=M,Nelec=Nelec,
        energies=evals,
        wavefunctions=psis,
        basis=basis0,
        transformation_matrix=None
    )

    return sci_res


def do_fci(h0,U,M,Nelec,num_roots=1,Sz=0,verbose=True):

    basis = basis_Np.get_fci_basis(M, Nelec)
    #inds, blocks = partition_by_Sz(basis)    # lists of indices + Sz values
    basis, idxs0 = basis_Np.subbasis_by_Sz(basis, Sz)  # S_z = 0 sector
    print(f"fci basis size = {len(basis)}")

    H_sparse = ops.get_ham(basis,h0,U,method="openmp")
    eigvals, eigvecs = eigsh(H_sparse, k=num_roots, which='SA')
    if verbose:
        print(f"Ground State Energy: {eigvals[0]:.6f}")
    psis = [cc.Wavefunction(M, basis, eigvecs[:,i]) for i in range(num_roots)]

    fci_res = results.NelecLowEnergySubspace(M=M,Nelec=Nelec,
        energies=eigvals,
        wavefunctions=psis,
        basis=basis,
        transformation_matrix=None
    )
    return fci_res


# -----------
# Generators
# -----------

def hamiltonian_generator(wf,one_body_terms,two_body_terms,thr=1e-7,return_hwf=True):
    r"""
    The basis is expanded by acting on a state with the hamiltonian

    Args : 
        wf: the input wavefunction :math:`|\psi\rangle`, a Wavefunction object 
        one_body_terms: the one_body part of the hamiltonian 
        two_body_terms: ..
        h0: the one particle hamiltonian 
        U : the two particle hamitlonian 
        thr : Basis elements with absolute coefficient below threshold are pruned before expansion
        return_hwf: if True, return the new wavefunction :math:`H|\psi\rangle` as well

    Returns: 
        diffbasis: the unique new basis terms 
        hwf: optional, the new wavefunction

    """
    wf.prune(thr)

    if return_hwf:
        hwf = cc.apply_one_body_operator(wf,one_body_terms) + cc.apply_two_body_operator(wf,two_body_terms)

        basis_hwf = hwf.get_basis()
        diffbasis = list(set(basis_hwf) - set(wf.get_basis()))

        return diffbasis,hwf 
    
    else:
        print("hamiltonian generator with return_hwf not implemented yet")


# -----------
# Selectors
# -----------

def cipsi_one_iter(hwf,ewf,diffbasis,K, h0,U):
    r"""
    One iteration of the CIPSI method. 

    For a given Slater determinant :math:`|a\rangle` accessible from a given wavefunction :math:`|\psi\rangle`, 
    the estimated coefficient at 2nd order Nesbet perturbation theory is
    .. math::

        c^{PT2}_a = \frac{|\langle a | H | \psi \rangle|^2}
                         {\langle \psi | H | \psi \rangle - \langle a | H | a \rangle}

    Args : 
        hwf: :math:`H|\psi\rangle`, a Wavefunction object 
        ewf: the energy :math:`\langle \psi|H|\psi\rangle`, a scalar
        one_body_terms: the one_body part of the hamiltonian 
        two_body_terms: ..
        K : the number of spin-orbitals 
        h0: the one particle hamiltonian 
        U : the two particle hamitlonian 
        thr : Basis elements with absolute coefficient below threshold are pruned before expansion

    Returns:
        diffbasis_sorted : the new basis elements sorted according to their PT2 coefficients

    """
    c_PT2 = np.zeros(len(diffbasis))

    for i,d in enumerate(diffbasis):
        occ_i = d.get_occupied_spin_orbitals()
        ahwf = hwf.amplitude(d)
        aha = cc.KL(occ_i, occ_i, K, h0, U)

        c_PT2[i] = np.abs(ahwf)**2 / np.real(ewf - aha + 1e-8)

    indsort = np.argsort(np.abs(c_PT2))[::-1]
    diffbasis_sorted = [diffbasis[i] for i in indsort]

    return diffbasis_sorted




