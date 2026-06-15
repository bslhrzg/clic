# ops.py
import clic_clib as cc
import numpy as np
from time import time
from scipy.sparse import csr_matrix


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

    #wf.normalize()
    rdm = np.zeros((K, K), dtype=np.complex128)
    for (i,ib) in enumerate(block):
        for (j,jb) in enumerate(block):
            # Create the operator term c†_i c_j
            spin_i = cc.Spin.Alpha if ib < M else cc.Spin.Beta
            spin_j = cc.Spin.Alpha if jb < M else cc.Spin.Beta
            orb_i = ib if ib < M else ib - M
            orb_j = jb if jb < M else jb - M
            
            # The operator term is a list containing a single tuple (h_ij = 1.0)
            op_term = [(orb_i, orb_j, spin_i, spin_j, 1.0)]
            
            # Apply the operator c†_i c_j to the ground state
            # This creates the state |Φ⟩ = c†_i c_j |Ψ⟩
            phi_wf = cc.apply_one_body_operator(wf, op_term)
            #phi_wf.normalize()
            
            # The RDM element is <Ψ|Φ>
            rdm[i, j] = wf.dot(phi_wf)

    return rdm


def get_n_i_j(wf,M,i,j):
    """ 
    TO DO : currently too expensive. Should check first if occ
    Compute the one-body reduced density matrix given a wavefunction
    Args:
        wf : a wavefunction object
        M  : number of spatial orbitals 
        i,j: get
    Returns:
        < wf | c^+_i c_j | wf >
    """

    wf.normalize()

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
    #phi_wf.normalize()
    
    # The RDM element is <Ψ|Φ>
    nij = wf.dot(phi_wf)

    return nij

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

def get_two_body_terms_(U, M, thr=1e-12):
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

def get_one_body_terms(h0, M, thr=1e-12):
    nz = np.argwhere(np.abs(h0) > thr)

    terms = []
    for i, j in nz:
        spin_i = cc.Spin.Alpha if i < M else cc.Spin.Beta
        spin_j = cc.Spin.Alpha if j < M else cc.Spin.Beta

        terms.append((
            int(i % M),
            int(j % M),
            spin_i,
            spin_j,
            complex(h0[i, j]),
        ))

    return terms

def get_two_body_terms(U, M, thr=1e-12):
    idx = np.argwhere(np.abs(U) > thr)

    terms = []
    for i, j, k, l in idx:
        inds = np.array([i, j, k, l])

        spins = [
            cc.Spin.Alpha if x < M else cc.Spin.Beta
            for x in inds
        ]

        orbs = [
            int(x if x < M else x - M)
            for x in inds
        ]

        terms.append((
            orbs[0], orbs[1], orbs[2], orbs[3],
            spins[0], spins[1], spins[2], spins[3],
            complex(U[i, j, k, l])
        ))

    return terms

def expect_Sz_from_rdm(rdm, M, block):
    # uses the block mapping you built in one_rdm
    val = 0.0
    for i, ib in enumerate(block):
        if ib < M:
            val += 0.5 * rdm[i, i]
        else:
            val -= 0.5 * rdm[i, i]
    return np.real(val)

def apply_Sz(wf, M, block=None):
    if block is None:
        block = list(range(2*M))
    terms = []
    for ib in block:
        spin = cc.Spin.Alpha if ib < M else cc.Spin.Beta
        orb  = ib if ib < M else ib - M
        coeff = 0.5 if ib < M else -0.5
        terms.append((orb, orb, spin, spin, coeff))
    # try batched apply if available
    try:
        return cc.apply_one_body_operator(wf, terms)
    except Exception:
        acc = wf.zero_like()
        for t in terms:
            acc = acc + cc.apply_one_body_operator(wf, [t])
        return acc

def expect_Sz(wf, M, block=None):
    if block is None:
        block = list(range(2*M))
    rdm = one_rdm(wf, M, block)
    return expect_Sz_from_rdm(rdm, M, block)

# prebuild S± term lists once per M
def _terms_Sminus(M):
    # S- = sum_p c†_{pβ} c_{pα}
    return [(p, p, cc.Spin.Beta,  cc.Spin.Alpha, 1.0) for p in range(M)]

def _terms_Splus(M):
    # S+ = sum_p c†_{pα} c_{pβ}
    return [(p, p, cc.Spin.Alpha, cc.Spin.Beta,  1.0) for p in range(M)]

def _apply_sum_terms(wf, terms):
    # try one batched call; fallback to accumulation
    try:
        return cc.apply_one_body_operator(wf, terms)
    except Exception:
        acc = wf.zero_like()
        for t in terms:
            acc = acc + cc.apply_one_body_operator(wf, [t])
        return acc

def apply_one_body_matrix(wf, M, matrix, block=None, thr=1e-12):
    """Apply a one-body matrix defined on a spin-orbital block."""
    matrix = np.asarray(matrix)
    if block is None:
        block = list(range(2 * M))
    if matrix.shape != (len(block), len(block)):
        raise ValueError("matrix shape must match the selected spin-orbital block")

    terms = []
    for i, j in np.argwhere(np.abs(matrix) > thr):
        ib = block[i]
        jb = block[j]
        spin_i = cc.Spin.Alpha if ib < M else cc.Spin.Beta
        spin_j = cc.Spin.Alpha if jb < M else cc.Spin.Beta
        orb_i = int(ib if ib < M else ib - M)
        orb_j = int(jb if jb < M else jb - M)
        terms.append((orb_i, orb_j, spin_i, spin_j, complex(matrix[i, j])))
    if not terms:
        return wf.zero_like()
    return _apply_sum_terms(wf, terms)

def expect_one_body_matrix(wf, M, matrix, block=None):
    """Return <O> and <O^2> for a Hermitian one-body operator O."""
    phi = apply_one_body_matrix(wf, M, matrix, block=block)
    return np.real(wf.dot(phi)), np.real(phi.dot(phi))

def expect_Splus_Sminus(wf, M):
    # ⟨Ψ| S+ S- |Ψ⟩ = ⟨Ψ| S+ (S- |Ψ⟩)⟩
    psi1 = _apply_sum_terms(wf, _terms_Sminus(M))
    psi2 = _apply_sum_terms(psi1, _terms_Splus(M))
    return np.real(wf.dot(psi2))

def expect_Sminus_Splus(wf, M):
    psi1 = _apply_sum_terms(wf, _terms_Splus(M))
    psi2 = _apply_sum_terms(psi1, _terms_Sminus(M))
    return np.real(wf.dot(psi2))

def expect_S2(wf, M, block=None):
    if block is None:
        block = list(range(2*M))

    spatial = sorted({ib if ib < M else ib - M for ib in block})
    index = {ib: i for i, ib in enumerate(block)}
    sz = np.zeros((len(block), len(block)), dtype=np.complex128)
    sp = np.zeros_like(sz)
    sm = np.zeros_like(sz)
    for p in spatial:
        alpha = p
        beta = p + M
        if alpha in index:
            sz[index[alpha], index[alpha]] = 0.5
        if beta in index:
            sz[index[beta], index[beta]] = -0.5
        if alpha in index and beta in index:
            sp[index[alpha], index[beta]] = 1.0
            sm[index[beta], index[alpha]] = 1.0

    Sz, Sz2 = expect_one_body_matrix(wf, M, sz, block=block)
    psi_sp = apply_one_body_matrix(wf, M, sp, block=block)
    psi_sm = apply_one_body_matrix(wf, M, sm, block=block)
    S2 = Sz2 + 0.5 * (np.real(psi_sp.dot(psi_sp)) + np.real(psi_sm.dot(psi_sm)))
    return S2, Sz

