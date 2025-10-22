from . import clic_clib as cc
import numpy as np

def one_rdm(wf,M):
    """ 
    Compute the one-body reduced density matrix given a wavefunction
    Args:
        wf : a wavefunction object
        M  : number of spatial orbitals 
    Returns:
        np.ndarray: the 1-rdm
    """

    K = 2*M

    rdm = np.zeros((K, K), dtype=np.complex128)
    for i in range(K):
        for j in range(K):
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