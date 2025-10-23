# hamiltonians.py
import numpy as np 
from . import clic_clib as cc
import scipy.sparse
import h5py
from .basis_1p import double_h,umo2so,transform_integrals_interleaved_to_alphafirst

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

def load_spatial_integrals(filename):
    """Loads spatial integrals and metadata from an HDF5 file."""
    try:
        with h5py.File(filename, "r") as f:
            hcore = f["h0"][:]
            ee = f["U"][:]
    except FileNotFoundError:
        raise RuntimeError(f"Integral file not found at: {filename}")
    except KeyError as e:
        raise RuntimeError(f"Missing key {e} in integral file '{filename}'. Expecting 'h0' and 'U'.")

    M = hcore.shape[0]
    print(f"Loaded spatial integrals from {filename}: M = {M}")
    return hcore, ee, M

def get_integrals_from_file(filepath: str, spin_structure: str): # <-- Add new argument
    """
    Main orchestrator function to load integrals from a file
    and convert them to the required spin-orbital format (AlphaFirst).
    Handles both spatial and spin-orbital integral sources.
    """
    # 1. Load the raw integrals from the HDF5 file
    with h5py.File(filepath, "r") as f:
        h0_raw = f["h0"][:]
        U_raw = f["U"][:]
    
    M = h0_raw.shape[0] // 2 if spin_structure != "spatial" else h0_raw.shape[0]
    print(f"Loaded raw integrals from {filepath}: M_spatial = {M}, spin_structure = '{spin_structure}'")

    # 2. Perform necessary basis transformations
    if spin_structure == "spatial":
        print(" -> Converting spatial integrals to AlphaFirst spin-orbital basis...")
        # hcore (M, M) -> h0 (2M, 2M)
        h0_spin_orbital = double_h(h0_raw, M)
        # U_spatial (M,M,M,M) -> U (2M,2M,2M,2M)
        U_spin_orbital = umo2so(U_raw, M)
    
    elif spin_structure == "interleaved":
        print(" -> Converting interleaved spin-orbital integrals to AlphaFirst...")
        h0_spin_orbital, U_spin_orbital = transform_integrals_interleaved_to_alphafirst(h0_raw, U_raw, M)
        
    elif spin_structure == "alpha_first":
        print(" -> Integrals are already in AlphaFirst format. No transformation needed.")
        h0_spin_orbital = h0_raw
        U_spin_orbital = U_raw
    
    # 3. Ensure C-contiguous and correct dtype for the C++ backend
    h0 = np.ascontiguousarray(h0_spin_orbital, dtype=np.complex128)
    U = np.ascontiguousarray(U_spin_orbital, dtype=np.complex128)
    
    return h0, U, M