# sci.py
from . import clic_clib as cc
from itertools import combinations
import numpy as np
from . import basis_Np,hamiltonians,ops,results,basis_1p
from scipy.sparse.linalg import eigsh 
from numpy.linalg import eigh,eig
from time import time

def selective_ci(
    h0, U, C,
    M, Nelec,
    seed,
    generator,
    selector,
    num_roots=1,
    one_bh=None,
    two_bh=None,
    max_iter=5,
    conv_tol=1e-6,
    prune_thr=1e-6,
    Nmul = None,
    min_size=512,
    max_size=5e4,
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
    C : (K,K) array_like 
        Transformation matrix (we want to keep track of it)
    M  : int
        Number of spatial orbitals.
    Nelec : int
        Total number of electrons.
    seed : list
        The inital basis
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
    min_size : Int -> to do  
    max_size : Int -> to do 
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


    t_start = time()
    basis0=seed

    # Initial Hamiltonian and ground state
    H = ops.get_ham(basis0, h0, U)
    # For small bases, a dense eig can be faster / safer; otherwise eigsh.
    dim0 = H.shape[0]
    if dim0 <= 256:
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

    if generator == hamiltonian_generator:
        # Precompute operator terms once
        if one_bh == None:
            one_bh = ops.get_one_body_terms(h0, M)
        if two_bh == None:
            two_bh = ops.get_two_body_terms(U, M)

    def get_roots(H,nroots,dim):
        if dim <= 64:
            evals, evecs = eigh(H.toarray())
        else:
            evals, evecs = eigsh(H, k=nroots, which='SA')
        indsort = np.argsort(np.real(evals))
        evals=evals[indsort]
        evecs=evecs[:,indsort]
        return evals[:nroots], evecs[:, :nroots]



    # Main selection loop
    for it in range(max_iter):
        # Propose new determinants using the provided generator

        #t0=time()
        inds,blocks = basis_Np.partition_by_Sz(basis0)
        #t1=time()
        #print(f"blocks = {blocks}, time block = {t1-t0}")

        current_size = len(basis0)

        #t2=time()
        gen_basis,hwf = generator(psi0,one_bh,two_bh,thr=prune_thr,return_hwf=True)
        # generate externals and couplings without normalization or pruning
        #ext, Hpsi_amp = hamiltonian_generator_raw(psi0, one_bh, two_bh)

       
        #t3=time()
        #print(f"gen basis time = {t3-t2}")
        selected_basis = selector(hwf,e0,gen_basis,2*M,h0,U)
         # PT2-guided selection
        #cipsi_thr = 1e-8
        #selected_basis, Ept2, ranked = cipsi_select(ext, Hpsi_amp, e0, 2*M, h0, U,
        #                                    select_cutoff=cipsi_thr)
        #t4=time()
        #print(f"sel basis time = {t4-t3}")


        # --- Determine how many new determinants to keep based on size constraints ---
        num_candidates = len(selected_basis)

        # 1. Calculate target number of new states from Nmul
        if Nmul is not None:
            n_add_nmul = int(Nmul * current_size)
        else:
            # If Nmul is None, we are unconstrained by it, so consider all candidates
            n_add_nmul = num_candidates

        # 2. Calculate number of new states needed to reach min_size
        n_add_min = max(0, min_size - current_size)
        # 3. Take the maximum of the two suggestions
        n_to_add = max(n_add_nmul, n_add_min)

        # 4. Cap by the number of available candidates
        n_to_add = min(n_to_add, num_candidates)
        
        # 5. Cap to not exceed max_size
        n_to_add = min(n_to_add, max_size - current_size)

        n_to_add = int(max(0, n_to_add)) # Ensure it's not negative
        # Truncate the selected basis to the final number of states to add
        final_selected = selected_basis[:n_to_add]
        # Merge and sort the basis
        new_basis = set(basis0) | set(final_selected)

        if len(new_basis) == len(basis0):
            if verbose:
                print(f"[iter {it}] no new determinants proposed; stopping.")
            break

        basis0 = sorted(list(new_basis))


        

        # Rebuild H and solve ground state
        #t5=time()
        H = ops.get_ham(basis0, h0, U)
        #t6=time()
        #print(f"ham construction time = {t6-t5}")
        dim = H.shape[0]

        #t7=time()
        evals, evecs = get_roots(H, num_roots, dim)
        e_new = float(evals[0])
        #t8=time()
        #print(f"ham diag time = {t8-t7}")
        psi0 = cc.Wavefunction(M, basis0, evecs[:, 0])
        #print(f"length before pruning = {len(psi0.get_basis())}")
        psi0.prune(prune_thr)
        #print(f"length after pruning = {len(psi0.get_basis())}")
        psi0.normalize()

        #t9=time()
        #print(f"psi0 construction time = {t9-t8}")

        dE = abs(e_new - energies[-1])
        energies.append(e_new)
        sizes.append(dim)

        if verbose:
            Es_str = " ".join(f"{ev:.12f}" for ev in evals[:num_roots])
            print(f"[iter {it}] dim={dim:>6}  Es=[{Es_str}]  |dE0|={dE:.3e}")

        if dE < conv_tol:
            if verbose:
                print(f"[conv] reached |dE| < {conv_tol}")
            break


    
    if num_roots > len(basis0):
        print("num_roots > length(basis)")
        num_roots = len(basis0)

    evals, evecs = get_roots(H, num_roots, len(basis0))
    psis = [cc.Wavefunction(M, basis0, evecs[:, i], keep_zeros=True) for i in range(num_roots)]

    t_final = time()
    print(f"sci total time : {t_final - t_start}")

    sci_res = results.NelecLowEnergySubspace(M=M,Nelec=Nelec,
        energies=evals,
        wavefunctions=psis,
        basis=basis0,
        transformation_matrix=C
    )

    return sci_res



def do_fci(h0,U,M,Nelec,num_roots=1,Sz=0,verbose=True):

    basis = basis_Np.get_fci_basis(M, Nelec)
    #inds, blocks = partition_by_Sz(basis)    # lists of indices + Sz values
    basis, idxs0 = basis_Np.subbasis_by_Sz(basis, Sz)  # S_z = 0 sector
    print(f"fci basis size = {len(basis)}")

    H_sparse = ops.get_ham(basis,h0,U,method="openmp")
    eigvals, eigvecs = eigsh(H_sparse, k=num_roots, which='SA')
    indsort = np.argsort(np.real(eigvals))
    eigvals=eigvals[indsort]
    eigvecs=eigvecs[:,indsort]
    if verbose:
        Es_str = " ".join(f"{ev:.12f}" for ev in eigvals[:num_roots])
        print(f"Es=[{Es_str}]")
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

def hamiltonian_generator(wf,one_body_terms,two_body_terms,thr=1e-6,return_hwf=True):
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
    #wf.prune(thr)

    if return_hwf:
        hwf = cc.apply_one_body_operator(wf,one_body_terms) + cc.apply_two_body_operator(wf,two_body_terms)
        hwf.normalize()
        hwf.prune(thr)
        basis_hwf = hwf.get_basis()
        diffbasis = list(set(basis_hwf) - set(wf.get_basis()))

        return diffbasis,hwf 
    
    else:
        print("hamiltonian generator with return_hwf not implemented yet")

def hamiltonian_generator_raw(wf, one_body_terms, two_body_terms):
    """
    Expand only by determinants directly connected to |psi> through H.
    Return:
      ext_dets: list of external determinants D_a not already in wf
      Hpsi_amp: dict mapping D_a -> <D_a|H|psi>  (unnormalized)
    """
    # Build H|psi> WITHOUT normalization or pruning
    Hpsi = cc.apply_one_body_operator(wf, one_body_terms)
    Hpsi += cc.apply_two_body_operator(wf, two_body_terms)

    # Extract amplitudes on determinants not in the current wf support
    cur_basis = set(wf.get_basis())
    Hpsi_basis = Hpsi.get_basis()

    ext_dets = []
    Hpsi_amp = {}
    for d in Hpsi_basis:
        if d not in cur_basis:
            amp = Hpsi.amplitude(d)    # this equals <d|H|psi>
            if amp != 0.0:
                ext_dets.append(d)
                Hpsi_amp[d] = amp
    return ext_dets, Hpsi_amp

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
    #diffbasis_sorted = [diffbasis[i] for i in indsort]
    diffbasis_sorted = [diffbasis[i] for i in indsort if np.abs(c_PT2)[i]>1e-16]


    return diffbasis_sorted



def cipsi_select(ext_dets, Hpsi_amp, E_var, K, h0, U, select_cutoff=1e-6,
                 max_to_add=None, level_shift=0.0):
    """
    Rank externals by |ΔE_a^(2)| = |<a|H|ψ>|^2 / (E_var - H_aa + shift).
    Keep those with |ΔE_a^(2)| above threshold, or the top-N if max_to_add is set.
    Return the chosen list and the total PT2 estimate for diagnostics.
    """
    import numpy as np

    denom_eps = 1e-12
    contrib = []
    Ept2_total = 0.0

    for a in ext_dets:
        occ = a.get_occupied_spin_orbitals()
        Haa = cc.KL(occ, occ, K, h0, U)           # diagonal Epstein–Nesbet
        denom = E_var - Haa + level_shift
        if abs(denom) < denom_eps:
            continue                              # skip near intruders
        v = Hpsi_amp[a]
        de2 = (abs(v)**2) / denom
        contrib.append((a, de2))
        Ept2_total += de2

    # sort by absolute PT2 magnitude, largest first
    contrib.sort(key=lambda t: abs(t[1]), reverse=True)
    #print((contrib[1:10]))
    #contrib = np.sort(contrib)[::-1]

    #if max_to_add is not None:
    #    chosen = [a for a, de2 in contrib[:max_to_add]]
    #else:
    chosen = [a for a, de2 in contrib if abs(de2) > select_cutoff]

    return chosen, Ept2_total, contrib


