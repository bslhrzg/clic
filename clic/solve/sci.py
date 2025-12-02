# sci.py
import clic_clib as cc
from itertools import combinations
import numpy as np
from clic.basis import basis_Np
from clic.ops import ops
from clic.results import results
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

from clic.model.model_utils import test_spin_sym

from numpy.linalg import eigh,eig
from time import time
from clic.io_clic.io_utils import vprint 
from .davidson import davidson 
from .diagh import get_roots, get_ham, diagH

applyH=False
dodavidson=False


def selective_ci(
    h0, U, C,
    M, Nelec,
    seed,
    generator,
    selector,
    num_roots=1,
    one_bh=None,
    two_bh=None,
    max_iter=0,
    conv_tol=1e-5,
    prune_thr=1e-5,
    Nmul = None,
    min_size=513,
    max_size=1e5,
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

    Note: called with generator=cipsi_one_iter, max_iter=1, Nmul=None, this is CISD
    """

    print("DEBUG: entering selective_ci()")

    is_spin_sym = test_spin_sym(h0)
    print(f"DEBUG: is_spin_sym: {is_spin_sym}")

    t_start = time()
    basis0=seed

    print("DEBUG: constructing tables")
    toltables = 1e-12
    tables = cc.build_hamiltonian_tables(h0,U,toltables)
    print("DEBUG: tables constructed")


    # Here we need the seed to define the starting point 
    # psi0 is NOT in general the ground state
    # At first, we use a linear combination of ALL eigenstates of the seed basis if possible 
    # This is to keep symmetry allowed states while in CIPSI
    # After this, ????

    # HERE DIAG
    # Initial Hamiltonian and ground state
    H = get_ham(basis0, h0, U, tables = tables)
    # For small bases, a dense eig can be faster / safer; otherwise eigsh.
    dim0 = H.shape[0]
    if dim0 <= 256:
        evals, evecs = eigh(H.toarray())
    else:
        evals, evecs = eigsh(H, k=num_roots, which='SA')

    e0 = float(evals[0])
    #psi0 = cc.Wavefunction(M, basis0, evecs[:, 0])
    psi0 = cc.Wavefunction(M, basis0, 1/np.sqrt(dim0) * (np.sum(evecs,axis=1)))


    if verbose:
        print(f"[init] dim={len(basis0)}  E0={e0:.12f}")

    energies = [e0]
    sizes = [len(basis0)]



    # Main selection loop
    for it in range(max_iter):
        # Propose new determinants using the provided generator

        #t0=time()
        #inds,blocks = basis_Np.partition_by_Sz(basis0)
        #t1=time()
        #print(f"blocks = {blocks}, time block = {t1-t0}")

        current_size = len(basis0)

        t2=time()
        # generate externals and couplings without normalization or pruning
        ext, Hpsi_amp = hamiltonian_generator(psi0, 
                                              #one_body_terms=None, 
                                              #two_body_terms=None, 
                                              tables, 
                                              h0, U)
        t3=time()
        #print(f"gen basis time = {t3-t2}")
        #selected_basis = selector(hwf,e0,gen_basis,2*M,h0,U)
        # PT2-guided selection
        cipsi_thr = 1e-12
        #print(f"DEBUG, using cipsi with threshold {cipsi_thr}")
        selected_basis, Ept2, ranked = cipsi_select(ext, Hpsi_amp, e0, 2*M, h0, U,
                                            select_cutoff=cipsi_thr)
        t4=time()
        #print(f"sel basis time = {t4-t3}")

        def keep_states(selected_basis,Nmul,current_size,min_size,max_size):
            """
            Logic to keep new states from generated ones:
            If Nmul is None, we try to keep everything consistently with min_size 
            and max_size 
            Else, we try to keep Nmul * current_size
            """

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

            return new_basis
            
        
        new_basis = keep_states(selected_basis, Nmul, current_size, min_size, max_size)
        if len(new_basis) == len(basis0):
            if verbose:
                print(f"[iter {it}] no new determinants proposed; stopping.")
            break
            

        basis0 = sorted(list(new_basis))
        dim = len(basis0)

        
        t5=time()
       
        evals, evecs = get_roots(
            basis0, 
            h0, U, 
            num_roots, 
            do_Sz = is_spin_sym,
            tables=tables,
            diag_kwargs=None, 
            verbose = False)

        e_new = float(evals[0])
        t8=time()
        #print(f"DEBUG: for basis size {dim}, ham diag time = {t8-t5}")


        # ADDING THE TWO LOWEST STATES TOGETHER
        if num_roots > 1 :
            psi0 = cc.Wavefunction(M, basis0, 1/np.sqrt(2) * (evecs[:, 0] + evecs[:, 1]))
        else :
            psi0 = cc.Wavefunction(M, basis0, evecs[:, 0])
        psi0.prune(prune_thr)
        psi0.normalize()



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

    evals, evecs = get_roots(
        basis0, 
        h0, U, 
        num_roots, 
        do_Sz = is_spin_sym,
        tables=tables,
        diag_kwargs=None, 
        verbose = True)
    
    psis = [cc.Wavefunction(M, basis0, evecs[:, i], keep_zeros=True) for i in range(num_roots)]

    t_final = time()
    print(f"sci total time : {t_final - t_start}")

    sci_res = results.NelecLowEnergySubspace(M_spatial=M,Nelec=Nelec,
        energies=evals,
        wavefunctions=psis,
        basis=basis0,
        transformation_matrix=C
    )

    return sci_res


# -----------
# Generators
# -----------

def hamiltonian_generator(wf, 
                          #one_body_terms, 
                          #two_body_terms, 
                          screened_H, 
                          h0,
                          U):
    """
    Expand only by determinants directly connected to |psi> through H.
    Return:
      ext_dets: list of external determinants D_a not already in wf
      Hpsi_amp: dict mapping D_a -> <D_a|H|psi>  (unnormalized)
    """
    t0 = time()

    # Build H|psi> WITHOUT normalization or pruning
    #Hpsi = cc.apply_one_body_operator(wf, one_body_terms)
    #Hpsi += cc.apply_two_body_operator(wf, two_body_terms)

    thr = 0  # Use a small threshold for screening

    # Apply the Hamiltonian using the new C++ function
    #print("Applying Hamiltonian to HF state with new C++ kernel...")
    Hpsi = cc.apply_hamiltonian(wf, screened_H, h0, U, thr) # tol_element=0
    t1 = time()
    #print(f"DEBUG: apply_ham time = {t1-t0}")

    # Extract amplitudes on determinants not in the current wf support
    cur_basis = set(wf.get_basis())
    Hpsi_basis = Hpsi.get_basis()
    #print(f"DEBUG: len new basis = {len(Hpsi_basis)}")

    ext_dets = []
    Hpsi_amp = {}
    for d in Hpsi_basis:
        if d not in cur_basis :#and d not in ext_dets:
            amp = Hpsi.amplitude(d)    # this equals <d|H|psi>
            if amp != 0.0:
                ext_dets.append(d)
                Hpsi_amp[d] = amp
    return ext_dets, Hpsi_amp

# -----------
# Selectors
# -----------

def cipsi_select(ext_dets, Hpsi_amp, E_var, K, h0, U, select_cutoff=1e-6,
                 max_to_add=None, level_shift=0.0):
    """
    Rank externals by |ΔE_a^(2)| = |<a|H|ψ>|^2 / (E_var - H_aa + shift).
    Keep those with |ΔE_a^(2)| above threshold, or the top-N if max_to_add is set.
    Return the chosen list and the total PT2 estimate for diagnostics.
    """

    contrib = []
    Ept2_total = 0.0

    for a in ext_dets:
        occ = a.get_occupied_spin_orbitals()
        Haa = cc.KL(occ, occ, K, h0, U)           # diagonal Epstein–Nesbet
        denom = E_var - Haa + level_shift + 1e-8
        v = Hpsi_amp[a]

        #de2 = (abs(v)**2) / denom
        

        try:
            de2 = (abs(v)**2) / denom

        except OverflowError as err:
            # print debug info *once for this determinant*
            print("OverflowError in CIPSI selection:")
            print(f"  det id: {a}")
            print(f"  occ: {occ}")
            print(f"  Haa: {Haa}")
            print(f"  denom: {denom}")
            print(f"  v: {v}")
            print(f"  |v|²: {abs(v)**2}")
            print(f"  exception: {err}")

            # Then either skip or clamp
            continue   # or: de2 = 0.0

        contrib.append((a, de2))
        Ept2_total += de2

    # sort by absolute PT2 magnitude, largest first
    contrib.sort(key=lambda t: abs(t[1]), reverse=True)

    #print(f"DEBUG: Biggest CPT2:")
    #keep=np.min([10,len(contrib)])
    #for a,c in contrib[:keep]:
        #print(f"DEBUG: |amp| = {c} for state {a}")
    

    if max_to_add is not None:
        chosen = [a for a, de2 in contrib[:max_to_add]]
    else:
        chosen = [a for a, de2 in contrib if abs(de2) > select_cutoff]

    return chosen, Ept2_total, contrib


