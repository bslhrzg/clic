
import numpy as np 
from . import clic_clib as cc
import scipy.sparse


# --- Integral Generation for Anderson Impurity Model ---
def get_impurity_integrals(M, u, e_bath, V_bath, mu):
    """Builds the one- and two-electron integrals for the Anderson Impurity Model.

        This function sets up the Hamiltonian terms in the spin-orbital basis,
        where alpha-spin orbitals are indexed first, followed by beta-spin orbitals.

        Args:
            M (int): Total number of spatial orbitals (impurity + bath).
            u (float): The on-site Hubbard interaction for the impurity orbital.
            e_bath (np.ndarray): Array of energies for the bath orbitals.
            V_bath (np.ndarray): Array of hybridization strengths between the
                                impurity and bath orbitals.
            mu (float): The chemical potential.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - **h0** (np.ndarray): The (2M, 2M) one-electron integral matrix.
                - **U** (np.ndarray): The (2M, 2M, 2M, 2M) two-electron integral tensor.
    """
    K = 2 * M
    h_spatial = np.zeros((M, M))
    diagonal_elements = np.concatenate(([-mu], e_bath))
    np.fill_diagonal(h_spatial, diagonal_elements)
    h_spatial[0, 1:] = V_bath
    h_spatial[1:, 0] = np.conj(V_bath)

    h0 = np.zeros((K, K))
    h0[0:M, 0:M] = h_spatial
    h0[M:K, M:K] = h_spatial
    
    U = np.zeros((K, K, K, K))
    imp_alpha_idx, imp_beta_idx = 0, M
    U[imp_alpha_idx, imp_beta_idx, imp_alpha_idx, imp_beta_idx] = u
    U[imp_beta_idx, imp_alpha_idx, imp_beta_idx, imp_alpha_idx] = u

    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    U = np.ascontiguousarray(U, dtype=np.complex128)
    return h0, U


def create_hubbard_V(M, U_val):
    """Builds the hubbard two-electron integrals in the spin-orbital basis,
        where alpha-spin orbitals are indexed first, followed by beta-spin orbitals.

        Args:
            M (int): Total number of spatial orbitals
            U_val (float): The on-site Hubbard interaction

        Returns:
            np.ndarray: The (2M, 2M, 2M, 2M) two-electron integral tensor.
    """
    K = 2 * M
    V = np.zeros((K, K, K, K), dtype=np.complex128)
    for i in range(M):
        alpha_i = i
        beta_i  = i + M
        V[alpha_i, beta_i, alpha_i, beta_i] = 2.0 * U_val
    V = np.ascontiguousarray(V, dtype=np.complex128)
    return V


def get_hubbard_dimer_ed_ref(t, U, M):
    K = 2 * M
    c_dag = [cc.get_creation_operator(K, i + 1) for i in range(K)]
    c = [cc.get_annihilation_operator(K, i + 1) for i in range(K)]
    H = scipy.sparse.csr_matrix((2**K, 2**K), dtype=np.complex128)
    if M == 2:
        H += -t * (c_dag[0] @ c[1] + c_dag[1] @ c[0])
        H += -t * (c_dag[0+M] @ c[1+M] + c_dag[1+M] @ c[0+M])
    for i in range(M):
        n_up = c_dag[i] @ c[i]
        n_down = c_dag[i+M] @ c[i+M]
        H += U * (n_up @ n_down)
    return H



# ---

def get_one_body_terms(h1, M, thr=1e-14):
    """Get a list of non zeros (above thr) one body operators"""
    terms = []
    for i in range(2*M):
        for j in range(2*M):
            if abs(h1[i, j]) > thr:
                si = cc.Spin.Alpha if i < M else cc.Spin.Beta
                sj = cc.Spin.Alpha if j < M else cc.Spin.Beta
                oi, oj = (i % M), (j % M)
                terms.append((oi, oj, si, sj, complex(h1[i, j])))
    return terms

def get_two_body_terms(U, M, thr=1e-14):
    """Get a list of non zeros (above thr) two body operators"""
    terms = []
    for i,j,k,l in np.argwhere(np.abs(U) > thr):
        spins = [cc.Spin.Alpha if idx < M else cc.Spin.Beta for idx in [i,j,k,l]]
        orbs  = [idx % M for idx in [i,j,k,l]]
        terms.append((orbs[0],orbs[1],orbs[2],orbs[3],
                      spins[0],spins[1],spins[2],spins[3], complex(0.5*U[i,j,k,l])))
    return terms

