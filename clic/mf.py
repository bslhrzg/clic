# mf.py
import numpy as np
from . import symmetries

def mfscf_(h0_0, U_0, Ne, maxiter=100):
    """
    Mean-field self-consistent field (MF-SCF) loop with simple linear mixing,
    in a basis where spins are BLOCKED (↑ then ↓):
        h0_0[:M,:M]   = spin-↑ block
        h0_0[M:,M:]   = spin-↓ block

    Args:
        h0_0 : (NF, NF) ndarray (complex or real)
            One-particle Hamiltonian in spin-block basis.
        U_0  : (NF, NF, NF, NF) ndarray (real or complex)
            Two-body interaction tensor U[i, j, k, l].
        Ne   : int
            Target total particle number.
        maxiter : int
            Maximum SCF iterations.

    Returns:
        hmf : (M, M) ndarray
            Mean-field spin-↑ block Hamiltonian (the one diagonalized).
        es  : (2M,) ndarray (complex)
            Spin-doubled eigenvalues (degenerate pairs).
        Vs  : (2M, 2M) ndarray (complex)
            Spin-doubled eigenvectors (columns).
        rho : (2M, 2M) ndarray (complex)
            Final density matrix in the full (↑⊕↓) basis.

    Notes:
        We diagonalize only the ↑ block (assuming spin symmetry at start)
        and then duplicate to build the full (↑,↓) structure. The MF potential
        is built on the full NF×NF space.
    """
    NF = h0_0.shape[0]
    M = NF // 2


    # Spin-up block (top-left M×M)
    h0_up = h0_0[:M, :M]

    print("max U :")
    print(np.max(U_0))

    print("starting from ρ(h0) in spin-up block")
    es_up, Vs_up = solve_h0(h0_up)
    es, Vs = double_es_Vs_blocked(es_up, Vs_up)  # build (↑,↓) structure with contiguous blocks
    print(f"iter 0, E = {np.real(es[:Ne]).sum()}")#, es = {es}")
    rho = get_rho(es, Vs, Ne)

    print(f"Ne = {Ne}")
    print(f"tr rho = {np.trace(rho)}")

    alpha = 0.2
    print(f"mixing parameter: α = {alpha}")

    hmf = None
    Vmf = None
    E0 = 1_000.0
    threshold = 1e-6

    # Report sparsity info (optional)
    nz_count = int(np.count_nonzero(U_0))
    print(f"number of non-zero U elements : {nz_count}")

    DE = -1.0
    print(f"{'Iter':>4s} {'E_total':>12s} {'EC':>12s} {'ΔE[n-1]':>15s}")# {'es[1:Ne]':>20s}

    it = 0
    for it in range(1, maxiter + 1):
        # Vectorized mean-field build on the FULL space
        Vmf, ec = get_mean_field(U_0, rho, use_einsum=True)

        # Work in spin-up block for diagonalization
        hmf = h0_up + Vmf[:M, :M]

        es_up, Vs_up = solve_h0(hmf)
        E = 2 * np.real(es[:Ne//2]).sum()
        es, Vs = double_es_Vs_blocked(es_up, Vs_up)
        

        if it % 10 == 0:
            #es_occ = np.real(es[:Ne])
            #es_occ_str = "[" + ", ".join(f"{x:.3f}" for x in es_occ) + "]"
            print(f"{it:4d} {np.real(E + ec):12.8f} {np.real(ec):12.8f} {DE:15.4e}") #{es_occ_str:>20s}

        DE = abs(E - E0)
        if DE < threshold:
            print(f"converged in {it} iterations")
            break
        else:
            E0 = E

        rho_new = get_rho(es, Vs, Ne)
        rho = alpha * rho_new + (1.0 - alpha) * rho

    if it == maxiter:
        print(f"NOT CONVERGED IN {maxiter} ITERATIONS")


    #print(f"HF energies : {es}")
    print(f"tr rho = {np.trace(rho)}")
    return hmf, es, Vs, rho


def get_rho(es, Vs, Ne, beta=1e3):
    """
    rho = sum_n f(ε_n; μ,β) |v_n><v_n|
    μ via bisection so that sum_n f(ε_n) = Ne.
    """
    es = np.real(np.asarray(es))
    M = Vs.shape[0]

    def total_occ(mu):
        x = beta * (es - mu)
        x = np.clip(x, -700, 700)
        return (1.0 / (1.0 + np.exp(x))).sum()

    lo = es.min() - 10.0
    hi = es.max() + 10.0
    mu = _bisection(lambda m: total_occ(m) - Ne, lo, hi, tol=1e-12, maxiter=200)

    x = beta * (es - mu)
    x = np.clip(x, -700, 700)
    f = 1.0 / (1.0 + np.exp(x))
    rho = (Vs * f[np.newaxis, :]) @ Vs.conj().T
    return rho


def solve_h0(h0):
    """
    Hermitian eigenproblem for h0.
    Returns real-sorted eigenvalues and eigenvectors (columns).
    """
    es, Vs = np.linalg.eigh(h0)
    return np.real(es), Vs


def double_es_Vs_blocked(es_up, Vs_up):
    """
    Spin-doubling for BLOCKED basis with BLOCK-DIAGONAL rotation:
      rows 0..M-1  = ↑, rows M..2M-1 = ↓
      columns 0..M-1   are ↑ orbitals
      columns M..2M-1  are ↓ orbitals (identical rotation)

    Returns:
        es_ : (2M,)
        Vs_ : (2M,2M)
    """
    M = len(es_up)
    # concatenate eigenvalues: [all spin-up, then all spin-down]
    es_ = np.zeros(2 * M, dtype=np.complex128)
    es_[:M] = es_up
    es_[M:] = es_up

    Vs_ = np.zeros((2 * M, 2 * M), dtype=np.complex128)
    # block-diagonal: diag(Vs_up, Vs_up)
    Vs_[:M, :M] = Vs_up          # spin-up block columns
    Vs_[M:, M:] = Vs_up          # spin-down block columns
    return es_, Vs_


def get_mean_field(U, rho, use_einsum=True, nz=None):
    
    # Coulomb term with U_ikjl
    J = np.einsum('ijkl,jl->ik', U, rho, optimize=True)
    # Exchange term with U_iljk
    K = np.einsum('ijlk,jl->ik', U, rho, optimize=True)
    
    V = J - K
    
    EC = -0.5 * np.trace(rho @ V)
    return V, float(np.real(EC))

def _bisection(f, a, b, tol=1e-12, maxiter=200):
    """
    Simple bisection for root of f on [a,b] with sign change (or expanded bracket).
    """
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        for scale in [2, 4, 8, 16]:
            aa = a - scale * (b - a)
            bb = b + scale * (b - a)
            fa = f(aa)
            fb = f(bb)
            if fa * fb <= 0:
                a, b = aa, bb
                break
        else:
            return 0.5 * (a + b)

    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(b - a) < tol or fm == 0.0:
            return m
        if np.sign(fa) * np.sign(fm) <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# -------------------------------------------------

def _reorder_to_spin_blocks(es_sorted, Vs_sorted, M):
    """
    Reorders globally sorted eigenvalues and eigenvectors back into a
    spin-blocked (AlphaFirst) convention. This is necessary when degenerate
    eigenvalues (like in RHF) cause sorting to interleave spin channels.

    Args:
        es_sorted (np.ndarray): Globally sorted eigenvalues.
        Vs_sorted (np.ndarray): Corresponding eigenvectors (columns).
        M (int): Number of spatial orbitals (size of one spin block).

    Returns:
        es_reordered (np.ndarray): Eigenvalues in [alpha..., beta...] order.
        Vs_reordered (np.ndarray): Eigenvectors in the same spin-blocked order.
    """
    NF = 2 * M
    alpha_vectors = []
    beta_vectors = []
    alpha_energies = []
    beta_energies = []

    # Classify each MO as alpha or beta
    for i in range(NF):
        vec = Vs_sorted[:, i]
        # An MO is alpha if its weight is predominantly in the first M components
        is_alpha = np.sum(np.abs(vec[:M])**2) > 0.5
        if is_alpha:
            alpha_vectors.append(vec)
            alpha_energies.append(es_sorted[i])
        else:
            beta_vectors.append(vec)
            beta_energies.append(es_sorted[i])
            
    # Reassemble into the desired block structure
    es_reordered = np.array(alpha_energies + beta_energies)
    Vs_reordered = np.zeros_like(Vs_sorted)
    
    if alpha_vectors:
        Vs_reordered[:, :M] = np.column_stack(alpha_vectors)
    if beta_vectors:
        Vs_reordered[:, M:] = np.column_stack(beta_vectors)
        
    return es_reordered, Vs_reordered

def _solve_h_symmetric(h0: np.ndarray):
    """
    (This function is correct as it was, the problem is handled after its use)
    Diagonalizes a hermitian matrix by exploiting its block-diagonal symmetry.
    Returns globally sorted eigenvalues and eigenvectors.
    """
    sym_dict = symmetries.analyze_symmetries(h0)
    blocks = sym_dict['blocks']
    identical_groups = sym_dict['identical_groups']

    unique_Vs_blocks, unique_es_blocks = {}, {}
    for group in identical_groups:
        leader_idx = group[0]
        leader_block_indices = blocks[leader_idx]
        h_block = h0[np.ix_(leader_block_indices, leader_block_indices)]
        es_block, Vs_block = np.linalg.eigh(h_block)
        unique_es_blocks[leader_idx] = es_block
        unique_Vs_blocks[leader_idx] = Vs_block
        
    Vs = symmetries.assemble_symmetric_hamiltonian(sym_dict, unique_matrices=unique_Vs_blocks)

    all_es = []
    for i in range(len(blocks)):
        found_leader = -1
        for group in identical_groups:
            if i in group:
                found_leader = group[0]
                break
        all_es.extend(unique_es_blocks[found_leader])
    all_es = np.array(all_es)

    sort_indices = np.argsort(np.real(all_es))
    es_sorted = all_es[sort_indices]
    Vs_sorted = Vs[:, sort_indices]
    
    return np.real(es_sorted), Vs_sorted, sym_dict


def mfscf(h0_0, U_0, Ne, maxiter=100, alpha=0.2, threshold=1e-8):
    """
    Symmetry-aware mean-field self-consistent field (MF-SCF) loop.
    This version now returns eigenvectors and eigenvalues in a spin-blocked
    (AlphaFirst) convention, even when degeneracies are present.
    """
    NF = h0_0.shape[0]
    M = NF // 2

    print("--- Initializing SCF from h0 ---")
    es, Vs, sym_dict = _solve_h_symmetric(h0_0)
    print(f"Symmetry analysis found {len(sym_dict['blocks'])} blocks.")
    print(f"Found {len(sym_dict['identical_groups'])} unique groups of blocks.")
    
    rho = get_rho(es, Vs, Ne)
    # The rest of the initial prints...
    print(f"iter 0, Tr(rho*h0) = {np.real(np.trace(rho @ h0_0)):.8f}")
    print(f"Ne = {Ne}, initial tr(rho) = {np.trace(rho):.8f}")
    print(f"mixing parameter: α = {alpha}")

    hmf, E0_total = None, 1_000.0
    
    nz_count = int(np.count_nonzero(U_0))
    print(f"\nNumber of non-zero U elements: {nz_count}")
    print(f"{'Iter':>4s} {'E_total':>15s} {'E_corr':>15s} {'ΔE_total':>15s}")
    
    it = 0
    for it in range(1, maxiter + 1):
        Vmf, ec = get_mean_field(U_0, rho)
        hmf = h0_0 + Vmf
        es, Vs, _ = _solve_h_symmetric(hmf) # es and Vs are globally sorted here

        E_total = 0.5 * np.real(np.trace(rho @ (h0_0+hmf)))
        DE = abs(E_total - E0_total)
        
        if it % 10 == 0 or it == 1:
            print(f"{it:4d} {E_total:15.8f} {np.real(ec):15.8f} {DE:15.4e}")

        if DE < threshold:
            print(f"\nConverged in {it} iterations.")
            break
        
        E0_total = E_total

        rho_new = get_rho(es, Vs, Ne)
        rho = alpha * rho_new + (1.0 - alpha) * rho

    if it == maxiter:
        print(f"\nNOT CONVERGED IN {maxiter} ITERATIONS")

    print(f"\nFinal tr(rho) = {np.trace(rho):.8f}")
    
    # --- FINAL FIX-UP STEP ---
    # Reorder the final results to conform to the spin-blocked convention
    print("Reordering final eigenvectors to spin-blocked convention...")
    es_final, Vs_final = _reorder_to_spin_blocks(es, Vs, M)
    
    # Return the reordered, user-friendly results
    return hmf, es_final, Vs_final, rho


