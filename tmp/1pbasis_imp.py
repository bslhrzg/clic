import numpy as np
from scipy.sparse.linalg import eigsh
import time
import matplotlib.pyplot as plt
from clic import * 

import numpy as np

def householder_tridiagonal(H):
    """
    Tridiagonalize a Hermitian matrix H using Householder reflectors.
    Returns T, Q with T = Q^† H Q, T tridiagonal.
    """
    H = np.array(H, dtype=complex)
    n = H.shape[0]
    Q = np.eye(n, dtype=complex)

    for k in range(n-2):
        x = H[k+1:, k]   # vector below the diagonal
        if np.allclose(x[1:], 0):
            continue

        # Build Householder vector
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        alpha = -np.sign(x[0].real) * np.linalg.norm(x)
        u = x - alpha * e1
        u = u / np.linalg.norm(u)

        # Form Householder reflector on the trailing block
        Hk = np.eye(n-k-1) - 2.0 * np.outer(u, u.conj())

        # Embed into full size
        Qk = np.eye(n, dtype=complex)
        Qk[k+1:, k+1:] = Hk

        # Apply similarity
        H = Qk.conj().T @ H @ Qk
        Q = Q @ Qk

    # Enforce strict tridiagonal (cleanup numerical noise)
    for i in range(n):
        for j in range(n):
            if abs(i-j) > 1:
                H[i,j] = 0.0

    return H, Q



# --- Main Test Execution ---
if __name__ == "__main__":
    # --- System Setup ---
    nb = 7
    M = 1 + nb
    u = 1
    mu = u/2
    Nelec = M
    e_bath = np.linspace(-2.0, 2.0, nb)
    if nb == 1 :
        e_bath = [0.0]
    print("e_bath = ",e_bath)
    V_bath = np.full(nb, 0.1)
    
    h0, U_mat = get_impurity_integrals(M, u, e_bath, V_bath, mu)

    h01spin = h0[:M,:M]
    print("h01spin = ")
    print(h01spin)

    imp_idx=0
    h0h,Q = householder_tridiagonal(h01spin)

    print("h0h = ")
    print(h0h)

    assert 1==2    

    print("h0 = ")
    print(h0)

    # --- Find Ground State ---
    basis = get_fci_basis(M, Nelec)
    print(f"basis size = {len(basis)}")
    H_sparse = build_hamiltonian_openmp(basis, h0, U_mat)
    eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
    e0 = eigvals[0]
    psi0_wf = Wavefunction(M, basis, eigvecs[:, 0])
    
    print(f"Impurity Model Ground State Energy: {e0:.6f}")

    # --- Matrix-Free Lanczos Calculation ---
    one_body_terms = get_one_body_terms(h0, M)
    two_body_terms = get_two_body_terms(U_mat, M)
    
    ws = np.linspace(-6, 6, 1001)

    NappH = 1   # for example; 0 means "just the seed support"
    L = 100
    eta = 0.02
    impurity_indices = [0, M]  # same as before

   
    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nScript finished. Check the plot for the spectral function.")