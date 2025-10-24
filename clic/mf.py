# mf.py
import numpy as np

def mfscf(h0_0, U_0, Ne, maxiter=100):
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

def get_mean_field_(U, rho, use_einsum=True, nz=None):
    """
    V_{ik} = sum_{j,l} rho_{jl} * ( U_{i j k l} - U_{i j l k} )
    EC     = -1/2 * Tr( rho V )

    Args:
        U  : (NF,NF,NF,NF) ndarray
        rho: (NF,NF) ndarray
        use_einsum : bool
            If True, build V with a fully vectorized contraction via np.einsum.
        nz : optional list of (i,j,k,l) tuples (fallback loop if provided and use_einsum=False)

    Returns:
        V  : (NF,NF) complex ndarray
        EC : float
    """
    NF = U.shape[0]

    if use_einsum:
        # A_{ik} = Σ_{j,l} U_{i j k l} rho_{j l}
        # B_{ik} = Σ_{j,l} U_{i j l k} rho_{j l}
        A = np.einsum('ijkl,jl->ik', U, rho, optimize=True)
        B = np.einsum('ijlk,jl->ik', U, rho, optimize=True)
        V = A - B
    else:
        # Fallback: Python loop (kept for completeness)
        if nz is None:
            nz = [tuple(idx) for idx in np.argwhere(U != 0)]
        V = np.zeros((NF, NF), dtype=np.complex128)
        for (i, j, k, l) in nz:
            V[i, k] += rho[j, l] * (U[i, j, k, l] - U[i, j, l, k])

    EC = -0.5 * np.trace(rho @ V)
    return V, float(np.real(EC))

def get_mean_field(U, rho, use_einsum=True, nz=None):
    # Standard physicist's notation V_{ik} = Σ_{jl} (U_{ijlk} - U_{iljk}) ρ_{lj}
    # Let's write it with rho_{jl}
    # V_{ik} = Σ_{jl} U_{ilkj} ρ_{lj} - Σ_{jl} U_{ijlk} ρ_{lj}
    # V_{ik} = Σ_{jl} U_{ikjl} ρ_{lj} - Σ_{jl} U_{ijlk} ρ_{jl} # Renaming dummy indices l->k, k->j, j->l
    
    # Original Python code:
    # A = np.einsum('ijkl,jl->ik', U, rho) 
    # B = np.einsum('ijlk,jl->ik', U, rho)
    # V = A - B

    # Let's try the other common convention.
    # U[i,k,j,l] corresponds to c_i^+ c_j^+ c_l c_k
    # V_ik = Sum_jl <ik|v|jl> rho_lj - <il|v|jk> rho_lj
    # V_ik = Sum_jl U_ikjl rho_lj - U_iljk rho_lj
    # NOTE: rho is symmetric, so rho_lj = rho_jl
    
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