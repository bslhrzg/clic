
# basis_Np.py
from . import clic_clib as cc
from itertools import combinations
import numpy as np
from . import mf 

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

from collections import defaultdict

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


def _extract_spatial_energies(h0, order="AlphaFirst", tol=1e-12):
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
        return diag, K

def get_starting_basis(h0, Nelec, order="AlphaFirst", tol=1e-12):
    """
    Build a starting CI basis by filling lowest spatial orbital energies.

    - Accepts spatial h0 (M×M) or spin-orbital h0 (2M×2M).
    - Fills 2e per spatial orbital (α and β).
    - If boundary falls in a degenerate block, return all consistent determinants.
    - If Nelec odd, put unpaired electron in lowest-energy block available,
      return both Ms = ±1/2 variants.

    Returns
    -------
    list[cc.SlaterDeterminant]
    """
    eps, M = _extract_spatial_energies(h0, order=order, tol=tol)
    if not (0 <= Nelec <= 2*M):
        raise ValueError("Nelec must be between 0 and 2*M")

    # sort orbitals by energy, stable to preserve degeneracies
    order_idx = np.argsort(eps, kind="mergesort")
    eps_sorted = eps[order_idx]

    # group into degeneracy blocks
    blocks = []
    s = 0
    for i in range(1, M):
        if abs(eps_sorted[i] - eps_sorted[s]) > tol:
            blocks.append(order_idx[s:i].tolist())
            s = i
    blocks.append(order_idx[s:M].tolist())

    # how many pairs (doubly occupied orbitals) and whether odd electron
    pairs = Nelec // 2
    has_single = (Nelec % 2 == 1)

    # collect fully filled blocks and detect boundary
    fixed_blocks = []
    boundary_block = []
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
    if pairs_left == 0:
        pair_sets.append(tuple(sorted(fixed_pairs)))
    else:
        for subset in combinations(boundary_block, pairs_left):
            pair_sets.append(tuple(sorted(fixed_pairs + list(subset))))

    dets = []
    if not has_single:
        # even number: fill pairs only
        for P in pair_sets:
            occ_a = sorted(P)
            occ_b = sorted(P)
            dets.append(cc.SlaterDeterminant(M, occ_a, occ_b))
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
                occ_b = sorted(Pset)
                dets.append(cc.SlaterDeterminant(M, occ_a, occ_b))
                # unpaired beta
                occ_a2 = sorted(Pset)
                occ_b2 = sorted(list(Pset) + [s])
                dets.append(cc.SlaterDeterminant(M, occ_a2, occ_b2))

    return sorted(dets)



def expand_basis(current_basis,one_body_terms,two_body_terms):
    """Given a basis, return the basis accessible through the hamiltonian 
    (i.e. the unique set of all slater determinant generated by application of the hamiltonian onto 
    each elements of the basis)
    """

    connected_by_H1 = cc.get_connections_one_body(current_basis, one_body_terms)
    connected_by_H2 = cc.get_connections_two_body(current_basis, two_body_terms)
    
    new_basis_set = set(current_basis) | set(connected_by_H1) | set(connected_by_H2)
    return sorted(list(new_basis_set))



