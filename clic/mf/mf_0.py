# mf.py
import numpy as np
from scipy.linalg import block_diag
from clic.symmetries import symmetries
from clic.io_clic.io_utils import *

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

def get_rho_symmetric(es, Vs, Ne, beta=1e3, deg_tol=1e-8):
    """
    Constructs a density matrix that respects degeneracies.

    All states within a degenerate energy shell are assigned the same
    (average) occupation number, ensuring that the resulting rho has the
    same symmetry as the Hamiltonian that generated it.

    Args:
        es (np.ndarray): Sorted eigenvalues.
        Vs (np.ndarray): Corresponding eigenvectors.
        Ne (int): Target number of electrons.
        beta (float): Inverse temperature.
        deg_tol (float): Tolerance for considering eigenvalues as degenerate.

    Returns:
        np.ndarray: The symmetrized density matrix.
    """
    es = np.real(np.asarray(es))
    NF = Vs.shape[0]

    # --- 1. Find the chemical potential as before ---
    def total_occ(mu):
        x = beta * (es - mu)
        # Clip to avoid overflow in np.exp
        x = np.clip(x, -700, 700)
        return (1.0 / (1.0 + np.exp(x))).sum()

    # Broaden search range in case mu is far from eigenvalues
    lo = es.min() - 20.0 / beta
    hi = es.max() + 20.0 / beta
    mu = _bisection(lambda m: total_occ(m) - Ne, lo, hi, tol=1e-12, maxiter=200)

    # --- 2. Calculate initial occupations ---
    x = beta * (es - mu)
    x = np.clip(x, -700, 700)
    f = 1.0 / (1.0 + np.exp(x))
    
    # --- 3. Symmetrize occupations across degenerate shells ---
    f_sym = np.copy(f)
    i = 0
    while i < NF:
        # Find all indices j>=i that are degenerate with es[i]
        j = i
        while j + 1 < NF and abs(es[j+1] - es[i]) < deg_tol:
            j += 1
        
        # If a degenerate group is found (size > 1)
        if j > i:
            group_indices = slice(i, j + 1)
            # Calculate the average occupation for this group
            avg_occ = np.mean(f[group_indices])
            # Assign this average occupation to all states in the group
            f_sym[group_indices] = avg_occ
            
        # Move to the next unprocessed state
        i = j + 1
        
    # --- 4. Build rho with the symmetrized occupations ---
    # Vs is (NF, NF), f_sym is (NF,). We want to scale each column of Vs
    # by the corresponding f_sym value.
    rho = (Vs * f_sym[np.newaxis, :]) @ Vs.conj().T
    
    return rho

def get_rho_symmetric_blockwise(es, Vs, Ne, sym_dict, col_block_id, beta=1e3):
    es = np.real(np.asarray(es))

    def total_occ(mu):
        x = np.clip(beta*(es - mu), -700, 700)
        return (1.0/(1.0 + np.exp(x))).sum()

    lo = es.min() - 10.0
    hi = es.max() + 10.0
    mu = _bisection(lambda m: total_occ(m) - Ne, lo, hi, tol=1e-12, maxiter=200)

    f = 1.0/(1.0 + np.exp(np.clip(beta*(es - mu), -700, 700)))
    f_sym = f.copy()

    # average over columns that correspond to blocks in the same identical group
    for group in sym_dict['identical_groups']:
        mask = np.isin(col_block_id, group)
        if np.any(mask):
            avg = f[mask].mean()
            f_sym[mask] = avg

    rho = (Vs * f_sym[np.newaxis, :]) @ Vs.conj().T
    return rho

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

def _solve_h_symmetric_(h0: np.ndarray, sym_dict=None):
    """
    Diagonalizes a hermitian matrix by exploiting its block-diagonal symmetry.
    Returns globally sorted eigenvalues and eigenvectors.
    """
    if sym_dict is None :
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

def mfscf(h0_0, U_0, Ne, maxiter=100, alpha=0.2, threshold=1e-7):
    """
    Symmetry-aware mean-field self-consistent field (MF-SCF) loop.
    This version now returns eigenvectors and eigenvalues in a spin-blocked
    (AlphaFirst) convention, even when degeneracies are present.
    """
    
    print_subheader("Mean-Field Self-Consistent Procedure with block handling")

    h0_0_copy = np.copy(h0_0)

    NF = h0_0.shape[0]
    M = NF // 2

    # DEBUG
    nimp_spatial = 7 
    # Correct impurity indices (spin-blocked)
    iimp = list(range(nimp_spatial)) + list(range(M, M + nimp_spatial))

    # Correct bath indices are all indices NOT in iimp
    all_indices = set(range(NF))
    iimp_set = set(iimp)
    ib = sorted(list(all_indices - iimp_set))

    print(f"iimp = {iimp}")
    print(f"ib = {ib}")

    h0imp = h0_0[np.ix_(iimp,iimp)]
    #_,_,sym_imp = _solve_h_symmetric(h0imp)
    print("h0imp = ")
    print(h0imp)
    # DEBUG

    #es, Vs, sym_dict = _solve_h_symmetric(h0_0)
    #rho = get_rho(es, Vs, Ne)

    es, Vs, sym_dict, col_block_id = _solve_h_symmetric(h0_0)
    rho = get_rho_symmetric_blockwise(es, Vs, Ne, sym_dict, col_block_id)

    # DEBUG 
    # Using only spin sym : 
    #dummy_ham = np.random.random((M,M))
    #dummy_ham = dummy_ham + dummy_ham.T
    #dummy_spin_sym_ham = block_diag(dummy_ham,dummy_ham)
    #sym_dict = symmetries.analyze_symmetries(dummy_spin_sym_ham)
    # DEBUG

    vprint(1,f"Symmetry analysis found {len(sym_dict['blocks'])} blocks.")
    vprint(1,f"Found {len(sym_dict['identical_groups'])} unique groups of blocks.")
    vprint(1,f"Blocks: {sym_dict["blocks"]}")
    vprint(1,f"Identical Groups (by block index): {sym_dict["identical_groups"]}")

    print(f"es = {es}")

    #rho = np.diag(np.ones(NF))
    # DEBUG
    print("*"*42)
    rho_imp = rho[np.ix_(iimp, iimp)]
    comm = rho_imp @ h0imp - h0imp @ rho_imp
    print("debug rho = ",np.max(np.abs(comm)))     # should be ≈ 0 if ρ respects the cubic basis
    print(f"is rho_imp diagonal : off diag norm ={ np.linalg.norm(rho_imp - np.diag(np.diagonal(rho_imp)))}")
    print(f"Nelec in imp = {np.trace(rho_imp)}")
    print(f"diag rho_imp = {np.diag(rho_imp)}")
    symrho = symmetries.analyze_symmetries(rho,verbose=True)
    if symmetries.compare_symmetries(sym_dict,symrho) == False: 
                vprint(1,"WARNING : rho breaks block symmetry")
    else : 
        print("RHO SYM IS OK !!!!")
    print("*"*42)
    # DEBUG


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


        # DEBUG
        h0bath_nz = np.sum(np.abs(Vmf[np.ix_(ib,ib)]))
        if h0bath_nz > 1e-12:
            print(f"h0bath_nz = {h0bath_nz}")
        # DEBUG
    
        if it == 1 : 
            sym_dict_test = symmetries.analyze_symmetries(hmf,verbose=False)
            if symmetries.compare_symmetries(sym_dict,sym_dict_test) == False: 
                vprint(1,"WARNING : Vmf breaks block symmetry, using blocks of hmf")
                #sym_dict = sym_dict_test
                hmfimp = hmf[np.ix_(iimp,iimp)]
                print("hmfimp =")
                print(hmfimp)

                Vmfimp = Vmf[np.ix_(iimp, iimp)]
                comm = Vmfimp @ h0imp - h0imp @ Vmfimp
                print("debug Vmf = ",np.max(np.abs(comm))) 


                
            #assert 1 == 0

        #es, Vs, _ = _solve_h_symmetric(hmf,sym_dict) # es and Vs are globally sorted here
        es, Vs, _, col_block_id = _solve_h_symmetric(hmf, sym_dict=sym_dict)  # reuse same blocks


        E_total = 0.5 * np.real(np.trace(rho @ (h0_0+hmf)))
        DE = abs(E_total - E0_total)
        
        if it % 10 == 0 or it == 1:
            vprint(1,f"{it:4d} {E_total:15.8f} {np.real(ec):15.8f} {DE:15.4e}")

        if DE < threshold:
            vprint(1,f"Converged in {it} iterations.")
            break
        
        E0_total = E_total

        #rho_new = get_rho_symmetric(es, Vs, Ne)
        rho_new = get_rho_symmetric_blockwise(es, Vs, Ne, sym_dict, col_block_id)

        rho = alpha * rho_new + (1.0 - alpha) * rho

    if it == maxiter:
        vprint(1,f"NOT CONVERGED IN {maxiter} ITERATIONS")

    vprint(1,f"Final tr(rho) = {np.trace(rho):.8f}")
    
    # Reorder the final results to conform to the spin-blocked convention
    vprint(2,"Reordering final eigenvectors to spin-blocked convention...")
    es_final, Vs_final = _reorder_to_spin_blocks(es, Vs, M)

    #DEBUG 
    rho_imp = rho[np.ix_(iimp,iimp)]
    print(f"Nelec in imp = {np.trace(rho_imp)}")
    assert np.allclose(h0_0, h0_0_copy), "FATAL: h0_0 was modified inside mfscf!"

    #DEBUG
    
    # Return the reordered, user-friendly results
    return hmf, es_final, Vs_final, rho


