import numpy as np
from . import clic_clib as cc

def transform_integrals_interleaved_to_alphafirst(h0_int, U_int, M):
    """
    Transforms integrals from spin-interleaved to AlphaFirst ordering.
    """
    K = 2 * M
    af_map = np.zeros(K, dtype=int)
    for i in range(M):
        af_map[i] = 2 * i
        af_map[i + M] = 2 * i + 1

    h0_af = np.zeros_like(h0_int)
    for p_af in range(K):
        for q_af in range(K):
            h0_af[p_af, q_af] = h0_int[af_map[p_af], af_map[q_af]]
            
    U_af = np.zeros_like(U_int)
    for p_af in range(K):
        for q_af in range(K):
            for r_af in range(K):
                for s_af in range(K):
                    p_int, q_int = af_map[p_af], af_map[q_af]
                    r_int, s_int = af_map[r_af], af_map[s_af]
                    U_af[p_af, q_af, r_af, s_af] = U_int[p_int, q_int, r_int, s_int]
    
    h0_af = np.ascontiguousarray(h0_af, dtype=np.complex128)
    U_af = np.ascontiguousarray(U_af, dtype=np.complex128)
    return h0_af, U_af


def double_h(h_core, M):
    """Converts spatial one-electron integrals to spin-orbital form (AlphaFirst)."""
    K = 2 * M
    h0 = np.zeros((K, K))
    for p in range(M):
        for q in range(M):
            # Alpha-alpha block
            h0[p, q] = h_core[p, q]
            # Beta-beta block
            h0[p + M, q + M] = h_core[p, q]
    return h0

def umo2so(U_mo, M):
    """
    Converts spatial physicist's integrals <pq|V|rs> to spin-orbital
    physicist's integrals <ij|V|kl> in AlphaFirst ordering.
    """
    K = 2 * M
    U_so = np.zeros((K, K, K, K))
    # U_mo[p,q,r,s] = <pq|V|rs>
    for p in range(M):
        for q in range(M):
            for r in range(M):
                for s in range(M):
                    val = U_mo[p, q, r, s]
                    if abs(val) > 1e-12:
                        p_a, p_b = p, p + M
                        q_a, q_b = q, q + M
                        r_a, r_b = r, r + M
                        s_a, s_b = s, s + M
                        
                        # αααα
                        U_so[p_a, q_a, r_a, s_a] = val
                        # ββββ
                        U_so[p_b, q_b, r_b, s_b] = val
                        # αβαβ
                        U_so[p_a, q_b, r_a, s_b] = val
                        # βαβα
                        U_so[p_b, q_a, r_b, s_a] = val
    return U_so

def basis_change_h0_U(A, B, C):
    """
    Performs a basis change on a 2-electron operator (U) and a 1-electron operator (A).

    This version uses a single, optimized einsum call for the 4D tensor transformation.

    Args:
        A (np.ndarray): A 2D array (matrix).
        B (np.ndarray): A 4D array (tensor).
        C (np.ndarray): The 2D transformation matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the transformed Anew and Unew.
    """
    # Same calculation for the 2D matrix A
    Anew = C.T.conj() @ A @ C
    Unew = np.einsum('ai,bj,abgd,gk,dl->ijkl', C.conj(), C.conj(), B, C, C, optimize=True)
    
    return Anew, Unew


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