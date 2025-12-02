
# basis_Np.py
import clic_clib as cc
from itertools import combinations, product
from collections import defaultdict
import numpy as np
from ..model.model_utils import test_spin_sym

def get_fci_basis(num_spatial, num_electrons):
    """
    Return the Full Configuration Interaction (fci) basis 
    for N_electrons among M spatial orbitals, as a list of 
    SlaterDeterminants objects 
    """
    num_spin_orbitals = 2 * num_spatial
    basis_dets = []
    for occupied_indices in combinations(range(num_spin_orbitals), num_electrons):
        occ_a = [i for i in occupied_indices if i < num_spatial]
        occ_b = [i - num_spatial for i in occupied_indices if i >= num_spatial]
        det = cc.SlaterDeterminant(num_spatial, occ_a, occ_b)
        basis_dets.append(det)
    return sorted(basis_dets)

def get_Nelec(det):
    occ_a = det.alpha_occupied_indices()
    occ_b = det.beta_occupied_indices()
    Na = len(occ_a)
    Nb = len(occ_b)
    return Na,Nb,Na+Nb
    


def partition_by_Sz(basis):
    """
    Group a list of SlaterDeterminant objects into S_z sectors.

    Returns
    -------
    inds : list[list[int]]
        For each block (ordered by Sz), the sorted indices of determinants in `basis`.
    blocks : list[float]
        The corresponding S_z values (in the same order as `inds`).
    """
    sz_to_inds = defaultdict(list)

    for idx, det in enumerate(basis):
        nα = len(det.alpha_occupied_indices())
        nβ = len(det.beta_occupied_indices())
        Sz = 0.5 * (nα - nβ)
        sz_to_inds[Sz].append(idx)

    blocks = sorted(sz_to_inds.keys())
    inds = [sorted(sz_to_inds[Sz]) for Sz in blocks]
    return inds, blocks

def subbasis_by_Sz(basis, target_Sz):
    """
    Extract the sub-basis with a given S_z.

    Returns
    -------
    subbasis : list[SlaterDeterminant]
    indices  : list[int]
        Indices (wrt the original `basis`) of the determinants in the subbasis.
    """
    inds, blocks = partition_by_Sz(basis)
    table = dict(zip(blocks, inds))
    idxs = table.get(target_Sz, [])
    return [basis[i] for i in idxs], idxs


def _extract_spatial_energies_(h0, order="AlphaFirst", tol=1e-12):
    """
    Return spatial orbital energies eps (length M) from either
    spatial h0 (M×M) or spin-orbital h0 (2M×2M).
    """
    h0 = np.asarray(h0)
    if h0.ndim != 2 or h0.shape[0] != h0.shape[1]:
        raise ValueError("h0 must be square")

    K = h0.shape[0]
    diag = np.real_if_close(np.diag(h0).astype(float))

    # Already spatial?
    if K % 2 != 0:
        return diag, K

    M = K // 2
    if order == "AlphaFirst":
        d_a = diag[:M]
        d_b = diag[M:]
    elif order == "Interleaved":
        d_a = diag[0::2]
        d_b = diag[1::2]
    else:
        raise ValueError("order must be 'AlphaFirst' or 'Interleaved'")

    if np.allclose(d_a, d_b, atol=tol, rtol=0):
        # Consistent spin-orbital HF: same energies for α and β
        return d_a, M
    else:
        # Just treat as spatial
        #return diag, K
        return d_a, M

def _extract_spatial_energies(h0, order="AlphaFirst", tol=1e-12):
    """
    Return spatial orbital energies eps (length M) from either
    spatial h0 (M×M) or spin-orbital h0 (2M×2M) in the
    spin-symmetric case (eps_alpha == eps_beta).

    In the spin-broken case, raise a ValueError; that case must be
    handled by a spin-orbital filling routine.
    """
    h0 = np.asarray(h0)
    if h0.ndim != 2 or h0.shape[0] != h0.shape[1]:
        raise ValueError("h0 must be square")

    K = h0.shape[0]
    diag = np.real_if_close(np.diag(h0).astype(float))

    # Already spatial: no explicit spin
    if K % 2 != 0:
        M = K
        return diag, M

    # Spin-orbital case: K = 2M
    M = K // 2
    if order == "AlphaFirst":
        d_a = diag[:M]
        d_b = diag[M:]
    elif order == "Interleaved":
        d_a = diag[0::2]
        d_b = diag[1::2]
    else:
        raise ValueError("order must be 'AlphaFirst' or 'Interleaved'")

    if np.allclose(d_a, d_b, atol=tol, rtol=0):
        # Restricted case: same energies for α and β
        return d_a, M
    else:
        # This must be treated with a spin-orbital filling algorithm
        raise ValueError("Spin-broken case (eps_alpha != eps_beta) must be handled separately.")
    

from itertools import combinations

def _get_starting_basis_spin_orbital(h0, Nelec, order="AlphaFirst", tol=1e-12):
    """
    Spin-broken case: build starting determinants by filling the lowest
    spin-orbital energies (α and β treated separately).

    Returns a list of SlaterDeterminant(M, occ_a, occ_b) with M = K//2.
    """
    h0 = np.asarray(h0)
    K = h0.shape[0]
    if K % 2 != 0:
        raise ValueError("Spin-orbital routine called with odd dimension.")

    diag = np.real_if_close(np.diag(h0).astype(float))
    M = K // 2  # spatial orbitals

    # Build spin-orbital energies and mapping
    if order == "AlphaFirst":
        # α: 0..M-1, β: M..2M-1
        d_a = diag[:M]
        d_b = diag[M:]
        eps_spin = np.concatenate([d_a, d_b])
        # map spin-orbital index -> (spin, spatial_index)
        def so_to_spin_spatial(so):
            if so < M:
                return 'a', so
            else:
                return 'b', so - M

    elif order == "Interleaved":
        # diag: [ε_a0, ε_b0, ε_a1, ε_b1,...]
        eps_spin = diag.copy()
        def so_to_spin_spatial(so):
            if so % 2 == 0:
                return 'a', so // 2
            else:
                return 'b', so // 2
    else:
        raise ValueError("order must be 'AlphaFirst' or 'Interleaved'")

    if not (0 <= Nelec <= 2*M):
        raise ValueError(f"Nelec ({Nelec}) must be between 0 and {2*M}.")

    # Sort spin-orbitals by energy
    order_idx = np.argsort(eps_spin, kind="mergesort")
    eps_sorted = eps_spin[order_idx]

    # Degeneracy blocks in spin-orbital space
    blocks = []
    if K > 0:
        s = 0
        for i in range(1, K):
            if abs(eps_sorted[i] - eps_sorted[s]) > tol:
                blocks.append(order_idx[s:i].tolist())
                s = i
        blocks.append(order_idx[s:K].tolist())

    # Fill electrons one by one across spin orbitals
    fixed_blocks, boundary_block = [], []
    left = Nelec
    for blk in blocks:
        if left >= len(blk):
            fixed_blocks.append(blk)
            left -= len(blk)
        else:
            boundary_block = blk
            break

    if left > 0 and not boundary_block:
        raise RuntimeError(f"Cannot place {Nelec} electrons in {K} spin orbitals? This should not happen for Nelec <= 2M.")

    fixed_spin_occ = [i for blk in fixed_blocks for i in blk]

    occ_sets = []
    if left == 0:
        occ_sets.append(tuple(sorted(fixed_spin_occ)))
    else:
        for subset in combinations(boundary_block, left):
            occ_sets.append(tuple(sorted(fixed_spin_occ + list(subset))))

    # Map spin-orbital occupancies to (alpha_indices, beta_indices)
    dets = []
    for occ_spin in occ_sets:
        occ_a = []
        occ_b = []
        for so in occ_spin:
            spin, p = so_to_spin_spatial(so)
            if spin == 'a':
                occ_a.append(p)
            else:
                occ_b.append(p)
        dets.append(cc.SlaterDeterminant(M, sorted(occ_a), sorted(occ_b)))

    return sorted(dets)


def get_starting_basis_(h0, Nelec, order="AlphaFirst", tol=1e-12):
    """
    Build a starting CI basis by filling lowest spatial orbital energies for the
    GIVEN SUBSPACE h0. It creates SlaterDeterminant objects that are local to this subspace.
    """
    eps, M_subspace = _extract_spatial_energies(h0, order=order, tol=tol)
    # The M for the determinant is ALWAYS the subspace M. This function is self-contained.
    M_for_det = M_subspace

    print(f"DEBUG, here in get_starting_basis : M_for_det = {M_for_det}")

    if not (0 <= Nelec <= 2*M_subspace):
        raise ValueError(f"Nelec ({Nelec}) cannot be between 0 and {2*M_subspace} for a subspace of size {M_subspace}.")

    # sort orbitals by energy, stable to preserve degeneracies
    order_idx = np.argsort(eps, kind="mergesort")
    eps_sorted = eps[order_idx]

    # group into degeneracy blocks
    blocks = []
    if M_subspace > 0:
        s = 0
        for i in range(1, M_subspace):
            if abs(eps_sorted[i] - eps_sorted[s]) > tol:
                blocks.append(order_idx[s:i].tolist())
                s = i
        blocks.append(order_idx[s:M_subspace].tolist())

    # how many pairs (doubly occupied orbitals) and whether odd electron
    pairs = Nelec // 2
    has_single = (Nelec % 2 == 1)

    # collect fully filled blocks and detect boundary
    fixed_blocks, boundary_block = [], []
    pairs_left = pairs
    for blk in blocks:
        if pairs_left >= len(blk):
            fixed_blocks.append(blk)
            pairs_left -= len(blk)
        else:
            boundary_block = blk
            break

    # all fixed paired orbitals
    fixed_pairs = [i for blk in fixed_blocks for i in blk]

    # choose pairs out of the boundary block if needed
    pair_sets = []
    if M_subspace > 0 and pairs_left > 0 and not boundary_block:
        raise RuntimeError(f"Cannot place {pairs} pairs in {M_subspace} orbitals. Not enough low-energy states for Nelec={Nelec}.")
        
    if pairs_left == 0:
        pair_sets.append(tuple(sorted(fixed_pairs)))
    elif boundary_block:
        for subset in combinations(boundary_block, pairs_left):
            pair_sets.append(tuple(sorted(fixed_pairs + list(subset))))

    dets = []
    if not has_single:
        # even number: fill pairs only
        for P in pair_sets:
            occ_a = sorted(list(P))
            occ_b = sorted(list(P))
            dets.append(cc.SlaterDeterminant(M_for_det, occ_a, occ_b))
    else:
        # odd number: put unpaired electron in lowest-energy remaining orbitals
        for P in pair_sets:
            Pset = set(P)
            remaining = [i for i in order_idx if i not in Pset]
            if not remaining:
                continue
            # lowest energy among remaining
            e0 = eps[remaining[0]]
            singles_block = [i for i in remaining if abs(eps[i]-e0) <= tol]

            for s in singles_block:
                # unpaired alpha
                occ_a = sorted(list(Pset) + [s])
                occ_b = sorted(list(Pset))
                dets.append(cc.SlaterDeterminant(M_for_det, occ_a, occ_b))
                # unpaired beta (comment out if you only want Ms >= 0)
                occ_a2 = sorted(list(Pset))
                occ_b2 = sorted(list(Pset) + [s])
                dets.append(cc.SlaterDeterminant(M_for_det, occ_a2, occ_b2))

    return sorted(dets)

def get_starting_basis(h0, Nelec, order="AlphaFirst", tol=1e-12):
    """
    Build a starting CI basis by filling lowest-energy states for the
    GIVEN SUBSPACE h0.

    - If h0 is spatial (M×M) or spin-symmetric (2M×2M with eps_alpha == eps_beta),
      we use spatial energies and the "pairs + single" logic.

    - If h0 is spin-broken (2M×2M with eps_alpha != eps_beta),
      we fill individual spin orbitals by energy.
    """
    h0 = np.asarray(h0)
    K = h0.shape[0]

    # Try spatial / spin-symmetric path first
    try:
        eps, M_subspace = _extract_spatial_energies(h0, order=order, tol=tol)
        # Spatial or restricted spin case; keep your old logic here
        M_for_det = M_subspace

        if not (0 <= Nelec <= 2*M_subspace):
            raise ValueError(f"Nelec ({Nelec}) cannot be between 0 and {2*M_subspace} for a subspace of size {M_subspace}.")

        # --- BEGIN: your original spatial logic (unchanged) ---
        order_idx = np.argsort(eps, kind="mergesort")
        eps_sorted = eps[order_idx]

        blocks = []
        if M_subspace > 0:
            s = 0
            for i in range(1, M_subspace):
                if abs(eps_sorted[i] - eps_sorted[s]) > tol:
                    blocks.append(order_idx[s:i].tolist())
                    s = i
            blocks.append(order_idx[s:M_subspace].tolist())

        pairs = Nelec // 2
        has_single = (Nelec % 2 == 1)

        fixed_blocks, boundary_block = [], []
        pairs_left = pairs
        for blk in blocks:
            if pairs_left >= len(blk):
                fixed_blocks.append(blk)
                pairs_left -= len(blk)
            else:
                boundary_block = blk
                break

        fixed_pairs = [i for blk in fixed_blocks for i in blk]

        pair_sets = []
        if M_subspace > 0 and pairs_left > 0 and not boundary_block:
            raise RuntimeError(f"Cannot place {pairs} pairs in {M_subspace} orbitals. Not enough low-energy states for Nelec={Nelec}.")

        if pairs_left == 0:
            pair_sets.append(tuple(sorted(fixed_pairs)))
        elif boundary_block:
            for subset in combinations(boundary_block, pairs_left):
                pair_sets.append(tuple(sorted(fixed_pairs + list(subset))))

        dets = []
        if not has_single:
            for P in pair_sets:
                occ_a = sorted(list(P))
                occ_b = sorted(list(P))
                dets.append(cc.SlaterDeterminant(M_for_det, occ_a, occ_b))
        else:
            for P in pair_sets:
                Pset = set(P)
                remaining = [i for i in order_idx if i not in Pset]
                if not remaining:
                    continue
                e0 = eps[remaining[0]]
                singles_block = [i for i in remaining if abs(eps[i] - e0) <= tol]

                for s in singles_block:
                    occ_a = sorted(list(Pset) + [s])
                    occ_b = sorted(list(Pset))
                    dets.append(cc.SlaterDeterminant(M_for_det, occ_a, occ_b))

                    occ_a2 = sorted(list(Pset))
                    occ_b2 = sorted(list(Pset) + [s])
                    dets.append(cc.SlaterDeterminant(M_for_det, occ_a2, occ_b2))
        # --- END: original spatial logic ---

        return sorted(dets)

    except ValueError as e:
        # If the error is from spin-broken case, fall through to spin-orbital filling.
        if "Spin-broken case" not in str(e):
            raise

    # Spin-broken branch: use spin-orbital energies
    return _get_starting_basis_spin_orbital(h0, Nelec, order=order, tol=tol)

def get_imp_starting_basis(h0, Nelec, Nelec_imp, imp_indices, order="AlphaFirst", tol=0.01):
    """
    Generates starting determinants with a fixed number of electrons on the impurity.
    It partitions the system, solves for ground states in each subspace, and then
    combines them into valid determinants for the full system.
    """
    K = h0.shape[0]
    M_global = K // 2 # This is the total M for the full system

    if Nelec == 0:
        print("Generating basis for Nelec = 0 (vacuum state).")
        # The only determinant is the one with no occupied orbitals.
        vacuum_det = cc.SlaterDeterminant(M_global, [], [])
        return [vacuum_det]
    
    # --- 1. Partition the system ---
    imp_indices = sorted(list(set(imp_indices)))
    M_imp = len(imp_indices)
    
    all_spatial_indices = np.arange(M_global)
    bath_indices = np.setdiff1d(all_spatial_indices, imp_indices).tolist()
    M_bath = len(bath_indices)

    print(f"Partitioning system: {M_imp} impurity orbitals, {M_bath} bath orbitals.")

    imp_spin_indices = imp_indices + [i + M_global for i in imp_indices]
    bath_spin_indices = bath_indices + [i + M_global for i in bath_indices]

    h0_imp = h0[np.ix_(imp_spin_indices, imp_spin_indices)]
    h0_bath = h0[np.ix_(bath_spin_indices, bath_spin_indices)]

    #print(f"DEBUG: h0_bath.shape = {h0_bath.shape}")

    Nelec_bath = Nelec - Nelec_imp
    if Nelec_bath < 0:
        raise ValueError(f"Nelec_imp ({Nelec_imp}) cannot be greater than Nelec ({Nelec}).")

    print(f"Generating basis for {Nelec_imp} electrons on impurity and {Nelec_bath} on bath.")

    # --- 2. Call worker function on subspaces ---
    # The worker will return LOCAL determinants (e.g., M=7 for imp, M=21 for bath)
    imp_dets_local = get_starting_basis(h0_imp, Nelec_imp, order=order, tol=tol)
    bath_dets_local = get_starting_basis(h0_bath, Nelec_bath, order=order, tol=tol)
    
    if not imp_dets_local or not bath_dets_local:
        print("Warning: One subspace yielded zero determinants. Returning empty list.")
        return []

    print(f"Found {len(imp_dets_local)} impurity configurations and {len(bath_dets_local)} bath configurations.")
    
    # --- 3. Combine local results into global determinants ---
    final_dets = []
    for imp_det, bath_det in product(imp_dets_local, bath_dets_local):
        
        # Use the accessor methods to get the LOCAL occupations (indices from 0 to M_subspace-1)
        imp_alpha_local = imp_det.alpha_occupied_indices()
        imp_beta_local = imp_det.beta_occupied_indices()
        bath_alpha_local = bath_det.alpha_occupied_indices()
        bath_beta_local = bath_det.beta_occupied_indices()

        #print(f"imp_alpha_local = {imp_alpha_local}, imp_beta_local = {imp_beta_local}, \
        #      bath_alpha_local = {bath_alpha_local},bath_beta_local = {bath_beta_local}")
        
        #print(f"bath_indices = {bath_indices}")
        #print(f"imp_indices = {imp_indices}")


        # Map local indices back to GLOBAL spatial indices
        global_alpha_imp = [imp_indices[i] for i in imp_alpha_local]
        global_beta_imp = [imp_indices[i] for i in imp_beta_local]
        global_alpha_bath = [bath_indices[i] for i in bath_alpha_local]
        global_beta_bath = [bath_indices[i] for i in bath_beta_local]
        
        # Combine and sort for the final determinant
        final_alpha = sorted(global_alpha_imp + global_alpha_bath)
        final_beta = sorted(global_beta_imp + global_beta_bath)
        
        # Create the FINAL determinant in the GLOBAL context (using M_global)
        final_dets.append(cc.SlaterDeterminant(M_global, final_alpha, final_beta))

    unique_dets = sorted(list(set(final_dets)))

    is_spin_sym = test_spin_sym(h0)
    print(f"DEBUG: is_spin_sym : {is_spin_sym}")
    if is_spin_sym:
        if Nelec % 2 == 0:
            target_Sz = 0 
            print(f"Retaining only Sz={target_Sz} in starting basis")
            unique_dets,_ = subbasis_by_Sz(unique_dets, target_Sz)

    #for det in unique_dets:
    #    print(f"det : {det}")
    
    print(f"Generated a total of {len(unique_dets)} unique starting determinants.")
    return unique_dets

def get_rhf_determinant(Nelec, M):
    """
    Returns the RHF/ROHF Slater determinant in a spin-blocked MO basis.
    """
    if Nelec % 2 == 0:
    
        n_occ = Nelec // 2
        occ_hf = list(range(n_occ))

        return [cc.SlaterDeterminant(M, occ_hf, occ_hf)]
    else :
        n_occ = Nelec // 2
        occ_m = list(range(n_occ))
        occ_p = list(range(n_occ+1))

        b1 = cc.SlaterDeterminant(M, occ_m, occ_p)
        b2 = cc.SlaterDeterminant(M, occ_p, occ_m)

        return [b1,b2]


def expand_basis(current_basis,one_body_terms,two_body_terms):
    """Given a basis, return the basis accessible through the hamiltonian 
    (i.e. the unique set of all slater determinant generated by application of the hamiltonian onto 
    each elements of the basis)
    """

    connected_by_H1 = cc.get_connections_one_body(current_basis, one_body_terms)
    connected_by_H2 = cc.get_connections_two_body(current_basis, two_body_terms)
    
    new_basis_set = set(current_basis) | set(connected_by_H1) | set(connected_by_H2)
    return sorted(list(new_basis_set))



