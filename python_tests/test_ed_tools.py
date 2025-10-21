import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigsh

try:
    from clic_clib import get_creation_operator, get_annihilation_operator
except ImportError:
    print("Could not import modules. Please run 'pip install -e .' first.")
    exit()

def test_hubbard_dimer():
    print("\n--- Testing ED tools by building a Hubbard Dimer ---")
    
    N_sites = 2
    N_so = N_sites * 2 # Number of spin-orbitals
    t = 1.0  # Hopping
    U = 1.0  # On-site interaction
    mu = 0#U/2 # Half-filling
    
    # Spin-orbital indices (1-based):
    # 1: site 1, up
    # 2: site 1, down
    # 3: site 2, up
    # 4: site 2, down

    # Get creation/annihilation operators from C++ core
    c_dag = [get_creation_operator(N_so, i) for i in range(1, N_so + 1)]
    c = [get_annihilation_operator(N_so, i) for i in range(1, N_so + 1)]
    
    # --- Build Hamiltonian ---
    dim = 2**N_so
    H = scipy.sparse.csc_matrix((dim, dim), dtype=np.complex128)

    # Hopping term: -t * (c_1up^† c_2up + c_2up^† c_1up + c_1dn^† c_2dn + c_2dn^† c_1dn)
    # Indices: 1up=0, 1dn=1, 2up=2, 2dn=3
    H += -t * (c_dag[0] @ c[2] + c_dag[2] @ c[0]) # up spins
    H += -t * (c_dag[1] @ c[3] + c_dag[3] @ c[1]) # down spins
    
    # Interaction term: U * (n_1up n_1dn + n_2up n_2dn)
    # n_i,s = c_i,s^† c_i,s
    n_1up = c_dag[0] @ c[0]
    n_1dn = c_dag[1] @ c[1]
    n_2up = c_dag[2] @ c[2]
    n_2dn = c_dag[3] @ c[3]
    H += U * (n_1up @ n_1dn + n_2up @ n_2dn)

    # Chemical potential
    for i in [0,1,2,3]:
        H += -mu * (c_dag[i] @ c[i])
                
    
    # --- Diagonalize ---
    # Find the ground state energy
    # We use eigsh which is for Hermitian matrices and finds a few eigenvalues
    # 'SA' means smallest algebraic value (ground state)
    eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
    gs_energy = eigenvalues[0]
    
    print(f"Hubbard Dimer (2 sites, 2 electrons): t={t}, U={U}")
    print(f"Calculated Ground State Energy: {gs_energy:.8f}")

    # --- Compare to known analytical result for 2 electrons ---
    # E = U/2 - sqrt( (U/2)^2 + 4t^2 )
    expected_gs_energy = U/2 - np.sqrt( (U/2)**2 + 4*(t**2) )
    print(f"Expected Analytical Energy:   {expected_gs_energy:.8f}")

    assert np.isclose(gs_energy, expected_gs_energy)
    
    print("ED tools test PASSED.")
    

if __name__ == "__main__":
    test_hubbard_dimer()