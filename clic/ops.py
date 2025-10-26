# ops.py
from . import clic_clib as cc
import numpy as np

def one_rdm(wf,M,block=None):
    """ 
    TO DO : currently too expensive. Should check first if occ
    Compute the one-body reduced density matrix given a wavefunction
    Args:
        wf : a wavefunction object
        M  : number of spatial orbitals 
        block: if not None, return the rdm only for the block indexes
    Returns:
        np.ndarray: the 1-rdm
    """

    if block == None : 
        K = 2*M
        block = list(range(K))
    else : 
        K = len(block)

    rdm = np.zeros((K, K), dtype=np.complex128)
    for i in block:
        for j in block:
            # Create the operator term c†_i c_j
            spin_i = cc.Spin.Alpha if i < M else cc.Spin.Beta
            spin_j = cc.Spin.Alpha if j < M else cc.Spin.Beta
            orb_i = i if i < M else i - M
            orb_j = j if j < M else j - M
            
            # The operator term is a list containing a single tuple (h_ij = 1.0)
            op_term = [(orb_i, orb_j, spin_i, spin_j, 1.0)]
            
            # Apply the operator c†_i c_j to the ground state
            # This creates the state |Φ⟩ = c†_i c_j |Ψ⟩
            phi_wf = cc.apply_one_body_operator(wf, op_term)
            
            # The RDM element is <Ψ|Φ>
            rdm[i, j] = wf.dot(phi_wf)

    return rdm


def get_ham(basis,h0,U,method="openmp"):
    """TO ADD : DETECT SPIN FLIPS TERMS"""
    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    U = np.ascontiguousarray(U, dtype=np.complex128)

    if method == "openmp":
        H = cc.build_hamiltonian_openmp(basis, h0, U)
    else : 
        print("method not implemented yet")
        assert 1==2
    return H 


def get_one_body_terms(h0, M, thr=1e-12):
    """
    The non-zeros (above threshold) elements of the one-body hamiltonian
    Args:
        h0 (np.ndarray): the one-body hamiltonian, A 2D array (matrix).
        M: the number of spatial orbitals.
        thr: optional, a threshold value for the returned elements

    Returns:
        list: A list containing the non zeros elements and the corresponding orbitals
    """
    terms = []
    for i in range(2*M):
        for j in range(2*M):
            if abs(h0[i, j]) > thr:
                spin_i = cc.Spin.Alpha if i < M else cc.Spin.Beta
                spin_j = cc.Spin.Alpha if j < M else cc.Spin.Beta
                orb_i = i if i < M else i - M
                orb_j = j if j < M else j - M
                terms.append((orb_i, orb_j, spin_i, spin_j, complex(h0[i, j])))
    return terms

def get_two_body_terms(U, M, thr=1e-12):
    """
    The non-zeros (above threshold) elements of the two-body hamiltonian
    Args:
        U (np.ndarray): the two-body hamiltonian, A 4D array (tensor).
        M: the number of spatial orbitals.
        thr: optional, a threshold value for the returned elements

    Returns:
        list: A list containing the non zeros elements and the corresponding orbitals
    """
    terms = []
    for i in range(2*M):
        for j in range(2*M):
            for k in range(2*M):
                for l in range(2*M):
                    if abs(U[i, j, k, l]) > thr:
                        spins = [cc.Spin.Alpha if idx < M else cc.Spin.Beta for idx in [i, j, k, l]]
                        orbs = [idx if idx < M else idx - M for idx in [i, j, k, l]]
                        terms.append((orbs[0], orbs[1], orbs[2], orbs[3],
                                      spins[0], spins[1], spins[2], spins[3],
                                      complex(U[i, j, k, l])))
    return terms