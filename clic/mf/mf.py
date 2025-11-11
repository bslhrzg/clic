# mf.py
import numpy as np
from scipy.linalg import block_diag
from clic.symmetries import symmetries
from clic.io_clic.io_utils import *

# -----------------------------------------------------
# legacy function
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

    vprint(3,"max U :")
    vprint(3,np.max(U_0))

    vprint(1,"starting from ρ(h0) in spin-up block")
    es_up, Vs_up = solve_h0(h0_up)
    es, Vs = double_es_Vs_blocked(es_up, Vs_up)  # build (↑,↓) structure with contiguous blocks
    vprint(1,f"iter 0, E = {np.real(es[:Ne]).sum()}")#, es = {es}")
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

# -----------------------------------------------------

# -----------------------------------------------------
# DENSITY AND BLOCKED DENSITY
# -----------------------------------------------------

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

def get_rho(es, Vs, Ne, beta=1e3, deg_tol=1e-8):
    """
    Constructs a density matrix that respects degeneracies, guaranteeing
    both symmetry and the correct particle number.

    All states within a degenerate energy shell are assigned the same
    (average) occupation number, ensuring that the resulting rho has the

    same symmetry as the Hamiltonian that generated it.
    """
    es = np.real(np.asarray(es))
    NF = Vs.shape[0]

    # --- 1. Find the chemical potential mu so that sum(f) = Ne ---
    def total_occ(mu):
        x = beta * (es - mu)
        # Clip to avoid overflow in np.exp
        x = np.clip(x, -700, 700)
        return (1.0 / (1.0 + np.exp(x))).sum()

    lo = es.min() - 20.0 / beta
    hi = es.max() + 20.0 / beta
    mu = _bisection(lambda m: total_occ(m) - Ne, lo, hi, tol=1e-12, maxiter=200)

    # --- 2. Calculate initial occupations ---
    x = beta * (es - mu)
    x = np.clip(x, -700, 700)
    f = 1.0 / (1.0 + np.exp(x))
    
    # --- 3. Symmetrize occupations across degenerate energy shells ---
    # This step is the key to preserving symmetry, and it also preserves
    # the total particle number, since sum(f) = sum(f_sym).
    f_sym = np.copy(f)
    i = 0
    while i < NF:
        # Find all indices j>=i that are degenerate with es[i]
        j = i
        while j + 1 < NF and abs(es[j+1] - es[i]) < deg_tol:
            j += 1
        
        # If a degenerate group is found (size > 1), average their occupations
        if j > i:
            group_indices = slice(i, j + 1)
            avg_occ = np.mean(f[group_indices])
            f_sym[group_indices] = avg_occ
            
        # Move to the next unprocessed state
        i = j + 1
        
    # --- 4. Build rho directly from the definition ---
    # This mathematically guarantees Tr(rho) = sum(f_sym) = Ne
    rho = (Vs * f_sym[np.newaxis, :]) @ Vs.conj().T
    
    # Add a final sanity check
    final_trace = np.trace(rho)
    if not np.isclose(final_trace, Ne, atol=1e-6):
        vprint(0, f"FATAL WARNING in get_rho: Final trace is {np.real(final_trace):.8f}, expected {Ne}")
        vprint(0, f"Sum of occupations before symmetrizing: {np.sum(f):.8f}")
        vprint(0, f"Sum of occupations after symmetrizing: {np.sum(f_sym):.8f}")

    return rho

def symmetrize_rho_to_blocks(rho, sym_dict):
    """Project ρ onto the block structure and average within identical groups.
       Preserves Tr ρ exactly."""
    rho = rho.copy()
    blocks = sym_dict['blocks']
    groups = sym_dict.get('identical_groups', [])
    NF = rho.shape[0]

    # zero out off-block entries
    keep = np.zeros((NF, NF), dtype=bool)
    for B in blocks:
        keep[np.ix_(B, B)] = True
    rho[~keep] = 0.0

    # average each identical group’s block submatrices
    for g in groups:
        submats = [rho[np.ix_(blocks[b], blocks[b])] for b in g]
        if not submats:
            continue
        A = sum(submats) / len(submats)
        for b in g:
            I = blocks[b]
            rho[np.ix_(I, I)] = A
    return rho

def get_rho_symmetric_blockwise(es, Vs, Ne, sym_dict, col_block_id, beta=1e3):
    rho = get_rho(es, Vs, Ne)                    # guarantees Tr ρ = Ne
    rho = symmetrize_rho_to_blocks(rho, sym_dict)   # guarantees exact block pattern
    return rho

# Helper to get electron counts for blocked density
def block_electron_counts(rho: np.ndarray, sym_dict: dict):
    """
    Return electrons per block: n_b = Tr[ρ_{bb}]
    where ρ_{bb} is the submatrix on rows 'blocks[b]'.

    Returns
    -------
    counts : np.ndarray of shape (n_blocks,)
        Electron count per block.
    total  : float
        Sum over all blocks (should equal Tr ρ).
    offblock_max : float
        Max absolute value of ρ outside the declared blocks, useful as a sanity check.
    """
    rho = np.asarray(rho)
    assert rho.ndim == 2 and rho.shape[0] == rho.shape[1], "rho must be square"
    blocks = sym_dict["blocks"]
    nB = len(blocks)

    counts = np.zeros(nB, dtype=float)

    # electrons per block
    for b, idx in enumerate(blocks):
        sub = rho[np.ix_(idx, idx)]
        counts[b] = float(np.real_if_close(np.trace(sub)))

    # off-block max magnitude
    mask = np.ones_like(rho, dtype=bool)
    for idx in blocks:
        mask[np.ix_(idx, idx)] = False
    offblock_max = 0.0 if not mask.any() else float(np.max(np.abs(rho[mask])))

    total = float(np.real_if_close(np.trace(rho)))
    return counts, total, offblock_max

def group_electron_counts(rho: np.ndarray, sym_dict: dict):
    """
    Aggregate electrons per identical group.

    Returns
    -------
    group_info : list of dict
        One entry per group with:
          - 'group_blocks': list of block indices in the group
          - 'sum': total electrons in the group
          - 'per_block': list of per-block electrons
          - 'avg_per_block': mean of per-block electrons
          - 'spread': max minus min over per-block electrons
    """
    counts, _, _ = block_electron_counts(rho, sym_dict)
    groups = sym_dict.get("identical_groups", [])
    info = []
    for g in groups:
        per_block = [float(counts[b]) for b in g]
        info.append({
            "group_blocks": list(g),
            "sum": float(np.sum(per_block)),
            "per_block": per_block,
            "avg_per_block": float(np.mean(per_block)) if per_block else 0.0,
            "spread": float(np.max(per_block) - np.min(per_block)) if len(per_block) > 1 else 0.0,
        })
    return info

# -----------------------------------------------------
# MEAN FIELD POTENTIAL
# -----------------------------------------------------

def get_mean_field(U, rho, use_einsum=True):
    
    # Coulomb term with U_ikjl
    J = np.einsum('ijkl,jl->ik', U, rho, optimize=True)
    # Exchange term with U_iljk
    K = np.einsum('ijlk,jl->ik', U, rho, optimize=True)
    
    V = J - K
    
    EC = -0.5 * np.trace(rho @ V)
    return V, float(np.real(EC))


def get_mean_field_nz(Uvals, rho, nzidx):
    # Get indices of non-zero elements in U
    
    i_idx, j_idx, k_idx, l_idx = nzidx

    # Allocate results
    V = np.zeros_like(rho)

    # Contract only over nonzero terms
    # Equivalent to V[i,k] += U[i,j,k,l] * rho[j,l]
    np.add.at(V, (i_idx, k_idx), Uvals * rho[j_idx, l_idx])

    # Exchange part: V[i,k] -= U[i,l,k,j] * rho[j,l]
    # One way is to reuse the same structure with swapped indices
    np.add.at(V, (i_idx, l_idx), -Uvals * rho[j_idx, k_idx])



    # Compute Coulomb energy
    EC = -0.5 * np.trace(rho @ V)
    return V, float(np.real(EC))


# -----------------------------------------------------
# SOLVER 
# -----------------------------------------------------

def solve_h0(h0, tol=1e-12):
    """
    Hermitian eigenproblem for h0.
    If the matrix is diagonal within a given tolerance, skip diagonalization.
    Returns sorted eigenvalues and eigenvectors (columns).
    """
    off_diag_norm = np.linalg.norm(h0 - np.diag(np.diagonal(h0)))
    if off_diag_norm < tol:
        es = np.real(np.diagonal(h0))
        Vs = np.eye(len(es))
    else:
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

def _solve_h_symmetric(h0: np.ndarray, sym_dict=None, tol=1e-12):
    if sym_dict is None:
        sym_dict = symmetries.analyze_symmetries(h0)
    blocks = sym_dict['blocks']
    identical_groups = sym_dict['identical_groups']

    unique_Vs_blocks, unique_es_blocks = {}, {}
    for group in identical_groups:
        leader = group[0]
        I = blocks[leader]
        h_block = h0[np.ix_(I, I)]
        es_b, Vs_b = np.linalg.eigh(h_block)
        unique_es_blocks[leader] = es_b
        unique_Vs_blocks[leader] = Vs_b

    # Assemble block-diagonal eigenvectors (same as you do now)
    Vs = symmetries.assemble_symmetric_hamiltonian(sym_dict, unique_matrices=unique_Vs_blocks)

    # Build the unsorted spectrum and a parallel array of block ids
    all_es = []
    col_block_id = []
    for b in range(len(blocks)):
        # find the leader of b’s identical group
        leader = next(g[0] for g in identical_groups if b in g)
        nb = len(unique_es_blocks[leader])
        all_es.extend(unique_es_blocks[leader])
        col_block_id.extend([b]*nb)

    all_es = np.asarray(all_es, dtype=float)
    sort_idx = np.argsort(all_es)
    es_sorted = all_es[sort_idx]
    Vs_sorted = Vs[:, sort_idx]
    col_block_id = np.asarray(col_block_id, dtype=int)[sort_idx]

    # return the mapping so get_rho_* can average by identical_groups
    return es_sorted, Vs_sorted, sym_dict, col_block_id

# -----------------------------------------------------
# MAIN FUNCTION 
# -----------------------------------------------------


def mfscf(h0_0, U_0, Ne, maxiter=100, alpha=0.2, threshold=1e-7, spinsym_only=False):
    """
    Symmetry-aware mean-field self-consistent field (MF-SCF) loop.
    This version now returns eigenvectors and eigenvalues in a spin-blocked
    (AlphaFirst) convention, even when degeneracies are present.

    spinsym_only: Bool
        If True, use a spin only symmetry block structure (this is regular Restricted Hartree-Fock)
    """
    
    print_subheader("Mean-Field Self-Consistent Procedure with block handling")
    vprint(1,f"mfscf proc with Ne={Ne}, maxiter={maxiter}, alpha={alpha}, threshold={threshold}, spinsym_only = {spinsym_only}")


    NF = h0_0.shape[0]
    M = NF // 2

    es, Vs, sym_dict, col_block_id = _solve_h_symmetric(h0_0)
    rho = get_rho_symmetric_blockwise(es, Vs, Ne, sym_dict, col_block_id)


    is_spin_sym=False
    if spinsym_only :
        h0_a = h0_0[:M,:M]
        h0_b = h0_0[M:,M:]
        is_spin_sym = np.allclose(h0_a,h0_b)
        assert is_spin_sym, "h0_0 is not spin symmetric, I could continue but I don't want to"
        vprint(1,f"Constructing spin only symmetry")
        sym_dict = symmetries.construct_spin_only_symdict(NF)
        
    vprint(1,f"Symmetry analysis found {len(sym_dict['blocks'])} blocks.")
    vprint(1,f"Found {len(sym_dict['identical_groups'])} unique groups of blocks.")
    vprint(1,f"Blocks: {sym_dict["blocks"]}")
    vprint(1,f"Identical Groups (by block index): {sym_dict["identical_groups"]}")



    vprint(1,f"iter 0, Tr(rho*h0) = {np.real(np.trace(rho @ h0_0)):.8f}")
    vprint(1,f"Ne = {Ne}, initial tr(rho) = {np.trace(rho):.8f}")
    vprint(1,f"mixing parameter: α = {alpha}")

    hmf, E0_total = None, 1_000.0
    
    nz_count = int(np.count_nonzero(U_0))
    vprint(2,f"Number of non-zero U elements: {nz_count}")
    vprint(1,f"{'Iter':>4s} {'E_total':>15s} {'E_corr':>15s} {'ΔE_total':>15s}")

    # If U is sparse, we precompute non zero elements 
    use_nz=False
    if nz_count < 1e-1 * NF**4 :
        vprint(2,f"U is sparse, using nzU")
        nzidx = np.nonzero(np.abs(U_0) > 1e-12)
        Uvals = U_0[nzidx]
        use_nz = True 



    it = 0
    for it in range(1, maxiter + 1):

        if use_nz :
            Vmf, ec = get_mean_field_nz(Uvals, rho, nzidx)
        else:
            Vmf, ec = get_mean_field(U_0, rho)

        hmf = h0_0 + Vmf
    
        if it == 1 : 
            if not spinsym_only :
                sym_dict_test = symmetries.analyze_symmetries(hmf,verbose=False)
                if symmetries.compare_symmetries(sym_dict,sym_dict_test) == False: 
                    vprint(1,"WARNING : Vmf breaks block symmetry, using blocks of hmf")
                    sym_dict = sym_dict_test
         
        es, Vs, _, col_block_id = _solve_h_symmetric(hmf, sym_dict=sym_dict)  # reuse same blocks


        E_total = 0.5 * np.real(np.trace(rho @ (h0_0+hmf)))
        DE = abs(E_total - E0_total)
        
        if it % 10 == 0 or it == 1:
            vprint(1,f"{it:4d} {E_total:15.8f} {np.real(ec):15.8f} {DE:15.4e}")

        if DE < threshold:
            vprint(1,f"Converged in {it} iterations.")
            break
        
        E0_total = E_total

        rho_new = get_rho_symmetric_blockwise(es, Vs, Ne, sym_dict, col_block_id)
        rho = alpha * rho_new + (1.0 - alpha) * rho

    if it == maxiter:
        vprint(1,f"NOT CONVERGED IN {maxiter} ITERATIONS")

    vprint(1,f"Final tr(rho) = {np.trace(rho):.8f}")
    
    # Reorder the final results to conform to the spin-blocked convention
    vprint(2,"Reordering final eigenvectors to spin-blocked convention...")
    es_final, Vs_final = _reorder_to_spin_blocks(es, Vs, M)
    
    # Return the reordered, user-friendly results
    return hmf, es_final, Vs_final, rho


