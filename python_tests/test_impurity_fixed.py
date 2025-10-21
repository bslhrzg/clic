import numpy as np
from scipy.sparse.linalg import eigsh
import time
import matplotlib.pyplot as plt
from clic import * 



def get_scalar_lanczos_wf(H_op, v_init, L):
    """
    Robust scalar Lanczos for a single Wavefunction vector.
    """
    alphas, betas = [], []
    v_list = []

    # Normalize initial vector
    norm_v = np.sqrt(abs(v_init.dot(v_init)))
    if norm_v < 1e-14:
        return np.array(alphas), np.array(betas)
    
    v_list.append((1.0 / norm_v) * v_init)

    for j in range(L):
        q = v_list[j]
        w = H_op(q)
        
        # Re-orthogonalize against previous two vectors (maintains orthogonality)
        if j > 0:
            w = w - betas[j-1] * v_list[j-1]
        
        alpha = q.dot(w)
        alphas.append(alpha)
        
        w = w - alpha * q
        
        beta = np.sqrt(abs(w.dot(w)))
        
        if beta < 1e-12:
            break # Breakdown
        
        betas.append(beta)
        v_list.append((1.0 / beta) * w)

    return np.array(alphas), np.array(betas)

def green_function_scalar_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta, impurity_indices):
    """
    Calculates DIAGONAL elements G_ii(ω) using a corrected scalar Lanczos.
    """
    Norb = 2 * M
    Nw = len(ws)
    G_all = np.zeros((Nw, Norb, Norb), dtype=np.complex128)

    for i in impurity_indices:
        # --- Greater Green's function (particle addition) ---
        seed_g = apply_creation(psi0_wf, i % M, Spin.Alpha if i < M else Spin.Beta)
        norm_g_sq = abs(seed_g.dot(seed_g))
        
        if norm_g_sq > 1e-12:
            a_g, b_g = get_scalar_lanczos_wf(H_op, seed_g, L)
            z = (ws + e0) + 1j * eta
            
            # CORRECTED Continued Fraction
            g_g = np.zeros_like(z)
            for k in range(len(a_g) - 1, -1, -1):
                b2 = b_g[k]**2 if k < len(b_g) else 0.0
                g_g = 1.0 / (z - a_g[k] - b2 * g_g)
            
            G_all[:, i, i] += norm_g_sq * g_g

        # --- Lesser Green's function (particle removal) ---
        seed_l = apply_annihilation(psi0_wf, i % M, Spin.Alpha if i < M else Spin.Beta)
        norm_l_sq = abs(seed_l.dot(seed_l))
        
        if norm_l_sq > 1e-12:
            a_l, b_l = get_scalar_lanczos_wf(H_op, seed_l, L)
            z = (-ws + e0) - 1j * eta
            
            # CORRECTED Continued Fraction
            g_l = np.zeros_like(z)
            for k in range(len(a_l) - 1, -1, -1):
                b2 = b_l[k]**2 if k < len(b_l) else 0.0
                g_l = 1.0 / (z - a_l[k] - b2 * g_l)
            
            G_all[:, i, i] -= norm_l_sq * g_l
            
    return G_all


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

    print(f"\nRunning MATRIX-FREE BLOCK Lanczos for impurity orbitals {impurity_indices}...")
    t_start = time.time()

    G_block, meta = green_function_block_lanczos_fixed_basis(
        M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
        h0, U_mat, one_body_terms, two_body_terms,
        coeff_thresh=1e-12, L=L, reorth=False
    )
    print("Fixed-basis sizes:", meta)

    
   
    t_end = time.time()
    print(f"  Calculation finished in {t_end-t_start:.3f}s")
    A_block = -(1/np.pi) * np.imag(G_block)
    
   
    
    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    for (i,ii) in enumerate(impurity_indices):
        plt.plot(ws, i*np.max(A_block[:,ii,ii])+(A_block[:, ii, ii]), label="A_bl_"+str(i)+"(ω)")
    plt.title("Impurity Spectral Function for Anderson Model (Matrix-Free Lanczos)")
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nScript finished. Check the plot for the spectral function.")