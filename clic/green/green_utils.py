# green_utils.py

import numpy as np
import clic_clib as cc
import scipy.sparse as sp
import numpy.linalg as npl
from numpy.linalg import norm


# -----------------------------
# Helpers: wavefunction <-> basis
# -----------------------------

def wavefunction_support(wf, coeff_thresh=1e-7):
    """
    Return the set of determinants in a Wavefunction with |coeff|>coeff_thresh.
    Assumes wf.data() is a mapping {SlaterDeterminant: complex}.
    """
    #data = wf.data()
    #return {det for det, c in data.items() if abs(c) > coeff_thresh}
    # 1. Make an independent copy in C++
    wf_copy = cc.Wavefunction(wf) 
    # 2. Prune the copy in C++
    wf_copy.prune(threshold=coeff_thresh)
    # 3. Get the remaining keys
    return set(wf_copy.data().keys())

def wf_to_vec(wf, basis_list):
    # Let's refine the idea:
    v = np.zeros(len(basis_list), dtype=np.complex128)
    for i, det in enumerate(basis_list):
        # wf.amplitude(det) is <det|wf>
        v[i] = wf.amplitude(det) 
    return v


# -----------------------------
# Build fixed Krylov basis by repeated H-applications at determinant level
# -----------------------------

def expand_basis_by_H(seed_dets, one_body_terms, two_body_terms, NappH):
    """
    Determinant-level expansion:
    B_0 = seed_dets
    B_{t+1} = B_t ∪ H*B_t connections (via one- and two-body graph expansion)
    Stop after NappH expansions.
    """
    current = set(seed_dets)
    for _ in range(NappH):
        conn1 = cc.get_connections_one_body(list(current), one_body_terms)
        conn2 = cc.get_connections_two_body(list(current), two_body_terms)
        current |= set(conn1)
        current |= set(conn2)
    return sorted(list(current))

def build_sector_basis_from_seeds(seeds_wf, one_body_terms, two_body_terms, NappH, coeff_thresh=1e-14):
    """
    seeds_wf: list of seed Wavefunctions for a given particle sector.
    1) collect support determinants from all seeds
    2) expand that set by NappH applications of H
    """
    if not seeds_wf:
        return []
    seed_support = set()
    for wf in seeds_wf:
        seed_support |= wavefunction_support(wf, coeff_thresh=coeff_thresh)
    return expand_basis_by_H(seed_support, one_body_terms, two_body_terms, NappH)

# -----------------------------
# Hamiltonian restricted to a fixed determinant basis
# -----------------------------

def build_H_in_basis(basis_dets, h0_clean, U_clean):
    """
    Use your fast OpenMP builder on the restricted basis.
    Returns a scipy.spmatrix (CSR).
    """
    if len(basis_dets) == 0:
        return sp.csr_matrix((0,0), dtype=np.complex128)
    H = cc.build_hamiltonian_openmp(basis_dets, h0_clean, U_clean)
    return H
