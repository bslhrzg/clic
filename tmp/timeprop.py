import numpy as np
import scipy.sparse as sp
import numpy.linalg as npl
import matplotlib.pyplot as plt
import time
from clic import * 



def _lanczos_tridiagonal(H, v0, L=200, reorth=False):
    """
    Perform Hermitian Lanczos iteration starting from vector v0.
    Constructs an orthonormal Krylov basis Q and the tridiagonal matrix T.

    Parameters
    ----------
    H : (N×N) linear operator (csr_matrix or LinearOperator)
        Hermitian Hamiltonian, must support `@` for matrix-vector product.
    v0 : ndarray (N,)
        Initial vector to start the Krylov subspace.
    L : int
        Maximum number of Lanczos steps.
    reorth : bool
        If True, apply a (single-pass) reorthogonalization against all previous vectors
        to improve numerical stability.

    Returns
    -------
    Q : ndarray (N×m)
        Orthonormal Lanczos basis vectors.
    T : ndarray (m×m)
        Tridiagonal Lanczos matrix.
    v0_norm : float
        Norm of the initial vector v0.
    """

    # Ensure v0 is a complex128 array
    v0 = np.asarray(v0, dtype=np.complex128)
    N = v0.size

    # Normalize the initial vector
    v0_norm = np.linalg.norm(v0)
    if v0_norm == 0:
        # If v0 is zero, return empty basis and matrix
        return np.zeros((N,0), dtype=np.complex128), np.zeros((0,0), dtype=np.complex128), 0.0

    # Initialize first basis vector q = v0 / ||v0||
    q_prev = np.zeros_like(v0)   # "ghost" previous vector, initially zero
    q = v0 / v0_norm
    Qcols = [q.copy()]           # list to store basis vectors
    alphas, betas = [], []       # diagonals and off-diagonals of T

    beta = 0.0
    for _ in range(L):
        # Apply Hamiltonian
        w = H @ q
        # Rayleigh quotient α = <q|H|q>
        alpha = np.vdot(q, w)
        # Remove components along current and previous q (three-term recurrence)
        w = w - alpha * q - beta * q_prev

        # Optional reorthogonalization against all previous Q vectors
        if reorth and Qcols:
            Qmat = np.column_stack(Qcols)
            w -= Qmat @ (Qmat.conj().T @ w)

        # Next off-diagonal β = ||w||
        beta = np.linalg.norm(w)
        alphas.append(alpha)

        # If β ≈ 0, Krylov subspace has closed (breakdown)
        if beta < 1e-14:
            break

        # Store β and normalize new q
        betas.append(beta)
        q_prev, q = q, w / beta
        Qcols.append(q.copy())

    # Number of steps actually performed
    m = len(alphas)
    if m == 0:
        # Should not normally happen, but guard against empty Krylov space
        return np.zeros((N,0), dtype=np.complex128), np.zeros((0,0), dtype=np.complex128), v0_norm

    # Construct Q matrix (columns are q0, q1, …, q_{m-1})
    Q = np.column_stack(Qcols[:m])

    # Construct tridiagonal T from α and β coefficients
    T = np.zeros((m, m), dtype=np.complex128)
    for k in range(m):
        T[k, k] = alphas[k]      # diagonal entries
        if k+1 < m:
            T[k, k+1] = betas[k] # upper diagonal
            T[k+1, k] = betas[k] # lower diagonal

    return Q, T, v0_norm


def green_function_from_time_propagation(
    i, j,
    M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
    h0_clean, U_clean, one_body_terms, two_body_terms,
    coeff_thresh=1e-12, L=100, reorth=False
):
    """
    Compute the retarded Green's function G_ij(ω) using time propagation + Lanczos.

    Strategy:
      - Build N+1 and N-1 sector bases starting from |a_j> = c_j†|ψ0> and |r_j> = c_j|ψ0>.
      - Restrict H to those bases and shift by E0 so ground state is stationary.
      - Run Lanczos on each sector to get tridiagonal T matrices.
      - Use T’s spectral decomposition to compute time overlaps S_add(t), S_rem(t).
      - Fourier-integrate with damping exp(-ηt) to get G_ij(ω).

    Parameters
    ----------
    i, j : int
        Indices of the Green's function G_ij(ω) to compute (0 ≤ i,j < 2M).
    M : int
        Number of spatial orbitals (so Norb = 2M including spin).
    psi0_wf : Wavefunction
        Ground state wavefunction.
    e0 : float
        Ground state energy.
    ws : ndarray
        Frequency grid.
    eta : float
        Broadening parameter (imaginary shift).
    impurity_indices : list[int]
        Indices of impurity orbitals (not directly needed here).
    NappH : int
        Number of H-applications to expand determinant basis.
    h0_clean, U_clean, one_body_terms, two_body_terms :
        Hamiltonian input data.
    coeff_thresh : float
        Threshold for determinant coefficients.
    L : int
        Maximum Lanczos steps.
    reorth : bool
        Reorthogonalize Lanczos vectors.

    Returns
    -------
    g : ndarray (len(ws),)
        Complex retarded Green's function values at given ω grid.
    """

    ws = np.asarray(ws, dtype=float)
    Norb = 2*M
    assert 0 <= i < Norb and 0 <= j < Norb

    # Helper: convert global index to (orbital, spin)
    def _index_to_spin_orb(i, M):
        si = cc.Spin.Alpha if i < M else cc.Spin.Beta
        oi = i % M
        return oi, si

    # Spin/orbital for indices i and j
    oj, sj = _index_to_spin_orb(j, M)
    oi, si = _index_to_spin_orb(i, M)

    # Seeds: |a_j> = c_j†|ψ0>, |r_j> = c_j|ψ0>
    wf_add_j = cc.apply_creation(psi0_wf, oj, sj)
    wf_rem_j = cc.apply_annihilation(psi0_wf, oj, sj)
    have_add = bool(wf_add_j.data()); have_rem = bool(wf_rem_j.data())
    if not have_add and not have_rem:
        # If both vanish, G_ij(ω) = 0
        return np.zeros_like(ws, dtype=np.complex128)

    # Bra seeds for overlap evaluation (|a_i>, |r_i>)
    wf_add_i = cc.apply_creation(psi0_wf, oi, si)
    wf_rem_i = cc.apply_annihilation(psi0_wf, oi, si)

    # Build determinant bases for N+1 and N-1 sectors by H-expansion
    basis_add = build_sector_basis_from_seeds(
        [wf_add_j] if have_add else [], one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh
    )
    basis_rem = build_sector_basis_from_seeds(
        [wf_rem_j] if have_rem else [], one_body_terms, two_body_terms, NappH, coeff_thresh=coeff_thresh
    )

    print(f"basis sizes: add ({len(basis_add)}), remove ({len(basis_rem)})")

    # Build restricted Hamiltonians
    H_add = build_H_in_basis(basis_add, h0_clean, U_clean) if have_add and len(basis_add) else sp.csr_matrix((0,0), dtype=np.complex128)
    H_rem = build_H_in_basis(basis_rem, h0_clean, U_clean) if have_rem and len(basis_rem) else sp.csr_matrix((0,0), dtype=np.complex128)

    # Shift by ground state energy e0 (makes |ψ0> stationary)
    if H_add.shape[0] > 0:
        H_add = H_add - e0 * sp.eye(H_add.shape[0], dtype=np.complex128, format='csr')
    if H_rem.shape[0] > 0:
        H_rem = H_rem - e0 * sp.eye(H_rem.shape[0], dtype=np.complex128, format='csr')

    # Project seeds into those bases
    a_j_vec = wf_to_vec(wf_add_j, basis_add) if have_add and H_add.shape[0] else np.zeros((0,), dtype=np.complex128)
    r_j_vec = wf_to_vec(wf_rem_j, basis_rem) if have_rem and H_rem.shape[0] else np.zeros((0,), dtype=np.complex128)
    a_i_vec = wf_to_vec(wf_add_i, basis_add) if have_add and H_add.shape[0] else np.zeros((0,), dtype=np.complex128)
    r_i_vec = wf_to_vec(wf_rem_i, basis_rem) if have_rem and H_rem.shape[0] else np.zeros((0,), dtype=np.complex128)

    # Lanczos on each sector starting from |a_j>, |r_j>
    Qp=Tp=n0p=None; Qm=Tm=n0m=None
    if have_add and a_j_vec.size:
        Qp, Tp, n0p = _lanczos_tridiagonal(H_add, a_j_vec, L=L, reorth=reorth)
    if have_rem and r_j_vec.size:
        Qm, Tm, n0m = _lanczos_tridiagonal(H_rem, r_j_vec, L=L, reorth=reorth)

    # Helper: pick time grid from ws and η
    def _compute_time_grid(ws, eta):
        ws = np.asarray(ws, dtype=float)
        wmin, wmax = ws.min(), ws.max()
        span = max(wmax - wmin, 1e-6)
        # target time scales: from η and from frequency spacing
        t_eta = 8.0 / max(eta, 1e-6)
        dw = np.median(np.diff(np.unique(np.round(ws, 12)))) if ws.size > 1 else span
        t_dw = 2.0*np.pi / max(dw, 1e-6)
        t_max = max(t_eta, t_dw)
        # dt to avoid aliasing: dt ≲ π/(4Ω)
        wabs = max(abs(wmin), abs(wmax), 1.0)
        dt_alias = np.pi / (4.0 * wabs)
        dt = min(dt_alias, t_max/4096.0)
        # clamp Nt between 512 and 8192
        Nt = int(min(max(np.ceil(t_max/dt), 512), 8192))
        ts = dt * np.arange(Nt, dtype=float)
        return ts, dt

    ts, dt = _compute_time_grid(ws, eta)

    # Helper: compute time overlaps S(t) = <bra| exp(i sign T t) e1 > * v0_norm
    def _time_overlaps_from_lanczos(T, Q, bra_vec, v0_norm, ts, sign):
        if T is None or T.size == 0:
            return np.zeros_like(ts, dtype=np.complex128)
        # Project bra onto Lanczos basis
        c = Q.conj().T @ bra_vec  # coefficients in Lanczos basis
        # Diagonalize small tridiagonal T
        evals, U = np.linalg.eigh(T)
        # e1 = (1,0,0,…)
        e1 = np.zeros((T.shape[0],), dtype=np.complex128); e1[0] = 1.0
        Udag_e1 = U.conj().T @ e1         # components of e1 in eigenbasis
        cdag_U  = np.conj(c) @ U          # bra projected in eigenbasis
        # Time evolution exp(i sign λ t) for each eigenvalue
        phases = np.exp(1j * sign * np.outer(evals, ts))   # (m, Nt)
        # Overlap sum
        return v0_norm * (cdag_U[:,None] * Udag_e1[:,None] * phases).sum(axis=0)

    # Time overlaps for addition (sign=-1) and removal (sign=+1)
    S_add = _time_overlaps_from_lanczos(Tp, Qp, a_i_vec, n0p, ts, sign=-1) if Qp is not None else np.zeros_like(ts, dtype=np.complex128)
    S_rem = _time_overlaps_from_lanczos(Tm, Qm, r_i_vec, n0m, ts, sign=+1) if Qm is not None else np.zeros_like(ts, dtype=np.complex128)

    # Fourier integration to G(ω): retarded integral with damping exp(-ηt)
    phase = np.exp(1j * np.outer(ws, ts)) * np.exp(-eta * ts)[None, :]
    g = -1j * dt * (phase @ (S_add + S_rem))  # -i factor for retarded GF

    return g

# --- Main Test Execution ---
if __name__ == "__main__":
    # --- System Setup ---
    nb = 9
    M = 1 + nb
    W = 2
    u = W
    mu = u/2
    Nelec = M
    e_bath = np.linspace(-W, W, nb)
    if nb == 1 :
        e_bath = [0.0]
    #print("e_bath = ",e_bath)
    cstV = 0.2
    V_bath = np.full(nb, cstV)
    
    h0, U_mat = get_impurity_integrals(M, u, e_bath, V_bath, mu)
    

    #print("h0 = ")
    #print(h0)

    # --- double chain 
    h0_spin = np.real(h0[:M,:M])
    Nelec_half = Nelec // 2
    final_hamiltonian_params = perform_natural_orbital_transform(h0_spin, u, Nelec_half)
    H_final_matrix = construct_final_hamiltonian_matrix(final_hamiltonian_params, M)
    H_final_matrix[0,0] = -u/2

    h0 = double_h(H_final_matrix,M)
    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    sb = get_starting_basis(h0, M)

    # --- Find Ground State ---
    #basis = get_fci_basis(M, Nelec)
    #print(f"basis size = {len(basis)}")
    #H_sparse = build_hamiltonian_openmp(basis, h0, U_mat)
    #eigvals, eigvecs = sp.linalg.eigsh(H_sparse, k=1, which='SA')
    #e0 = eigvals[0]
    #psi0_wf = Wavefunction(M, basis, eigvecs[:, 0])
    
    cipsi_max_iter = 12
    res = selective_ci(h0,U_mat,M,Nelec,hamiltonian_generator,cipsi_one_iter,
                       Nmul=1.0,max_iter=cipsi_max_iter,prune_thr=1e-6)

    e0 = res.ground_state_energy
    psi0_wf = res.ground_state_wavefunction
    
    print(f"Impurity Model Ground State Energy: {e0:.6f}")

    # --- Matrix-Free Lanczos Calculation ---
    one_body_terms = get_one_body_terms(h0, M)
    two_body_terms = get_two_body_terms(U_mat, M)
    
    ws = np.linspace(-6, 6, 1001)

    NappH = 3   # for example; 0 means "just the seed support"
    eta = 0.2
    L = 150
    impurity_indices = [0, M]  # same as before

    t_start = time.time()

    """
    print(f"\nRunning MATRIX-FREE BLOCK Lanczos for impurity orbitals {impurity_indices}...")

    G_block, meta = green_function_block_lanczos_fixed_basis(
        M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
        h0, U_mat, one_body_terms, two_body_terms,
        coeff_thresh=1e-12, L=L, reorth=False
    )
    print("Fixed-basis sizes:", meta)
    """
    g_time =  green_function_from_time_propagation(
    0, 0,
    M, psi0_wf, e0, ws, eta, impurity_indices, NappH,
    h0, U_mat, one_body_terms, two_body_terms,
    coeff_thresh=1e-7, L=L, reorth=False
)
   
    t_end = time.time()
    print(f"  Calculation finished in {t_end-t_start:.3f}s")
    #A_block = -(1/np.pi) * np.imag(G_block)
    
    A_time = -(1/np.pi) * np.imag(g_time)
   
    
    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    #for (i,ii) in enumerate(impurity_indices):
    #    plt.plot(ws, i*np.max(A_block[:,ii,ii])+(A_block[:, ii, ii]), label="A_bl_"+str(i)+"(ω)")
    plt.plot(ws,A_time, label="g from time")
    plt.title("Impurity Spectral Function for Anderson Model (Matrix-Free Lanczos)")
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dos.png")
    plt.show()
    
    print("\nScript finished. Check the plot for the spectral function.")