import numpy as np
from numpy.linalg import eigh, svd
from scipy.linalg import block_diag, qr
import clic

# ------------------------------------------------------------
# Scalar 
# ------------------------------------------------------------



def get_double_chain_transform(h_spin, Nelec):
    """
    Performs the 5-step Natural Orbital transformation, yielding a Hamiltonian
    with a double-chain structure and the corresponding unitary transformation matrix C.

    Args:
        h_spin (np.ndarray): The initial single-particle Hamiltonian (M x M).
        u (float): The on-site interaction strength.
        Nelec (int): The number of electrons in the spin sector (Nelec_total / 2).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - h_final_matrix (np.ndarray): The final Hamiltonian in the double-chain basis.
            - C_total (np.ndarray): The total unitary transformation matrix C.
    """
    M = h_spin.shape[0]

    # --- (i) Mean-Field ---
    h_mf = h_spin.copy()
    #h_mf[0, 0] += u * 0.5

    toto = h_spin.copy() 
    #print("original toto = ")
    #print(np.real(toto))
    # --- (ii) Natural Orbital Basis for the Bath ---
    e_mf, C_mf = eigh(h_mf)
    rho_mf = C_mf[:, :Nelec] @ C_mf[:, :Nelec].T
    rho_bath = rho_mf[1:, 1:]
    n_no, W = eigh(rho_bath)
    print(f"n_no = {n_no}")
    
    occupations_dist_from_integer = np.minimum(n_no, 1 - n_no)
    b_idx = np.argmax(occupations_dist_from_integer)
    
    other_indices = [i for i in range(M - 1) if i != b_idx]
    # Sorting ensures a deterministic basis ordering
    filled_indices = sorted([i for i in other_indices if n_no[i] > 0.5])
    empty_indices = sorted([i for i in other_indices if n_no[i] <= 0.5])
    
    ordered_bath_indices = [b_idx] + filled_indices + empty_indices
    W_ordered = W[:, ordered_bath_indices]
    C1 = block_diag(1, W_ordered)

    toto = C1.conj().T @ toto @ C1 
    print("after bath no toto = ")
    print(np.real(toto))
    # --- (iii) Bonding/Anti-bonding Transformation ---
    rho_no_basis = C1.T @ rho_mf @ C1
    rho_ib = rho_no_basis[:2, :2]
    e_bond, U_bond = eigh(rho_ib)
    print(f"e_bond = {e_bond}")
    C2 = np.identity(M)
    C2[:2, :2] = U_bond
    
    C_upto_decoupled = C1 @ C2

    toto = C2.conj().T @ toto @ C2 
    print("after bond/antibond toto = ")
    print(np.real(toto))

    # --- (iv) Lanczos Tridiagonalization (Chain Transformation) ---
    h_decoupled = C_upto_decoupled.T @ h_mf @ C_upto_decoupled

    num_filled = len(filled_indices)
    conduction_indices = [0] + list(range(2 + num_filled, M))
    valence_indices = [1] + list(range(2, 2 + num_filled))

    h_conduction = h_decoupled[np.ix_(conduction_indices, conduction_indices)]
    v0_c = np.zeros(h_conduction.shape[0]); v0_c[0] = 1.0
    T_c, Q_c = clic.scalar_lanczos(h_conduction, v0_c)

    h_valence = h_decoupled[np.ix_(valence_indices, valence_indices)]
    v0_v = np.zeros(h_valence.shape[0]); v0_v[0] = 1.0
    T_v, Q_v = clic.scalar_lanczos(h_valence, v0_v)
    
    C_lanczos = np.identity(M)
    if Q_c.shape[1] > 0:
      C_lanczos[np.ix_(conduction_indices, conduction_indices)] = Q_c
    if Q_v.shape[1] > 0:
      C_lanczos[np.ix_(valence_indices, valence_indices)] = Q_v
      
    # C_to_chains transforms from original basis to the basis of Lanczos vectors
    # (permuted).
    C_to_chains = C_upto_decoupled @ C_lanczos

    toto = C_lanczos.conj().T @ toto @ C_lanczos 
    print("afterlanczos toto = ")
    print(np.real(toto))

    # --- (v) Construct Final Basis and Transformation Matrix C_total ---
    # The final basis is {|i>, |b>, |c1>, |c2>, ..., |v1>, |v2>, ...}
    # where |i> and |b> are the impurity and special NO in the C1 basis.
    # The other vectors are the Lanczos chain vectors (excluding chain heads).
    
    # Get the vector representations of the chain states in the original basis
    # These are the columns of the C_to_chains matrix
    len_c = Q_c.shape[1]
    len_v = Q_v.shape[1]
    
    # The vector for the head of the conduction chain, |c0> = |A>, in the original basis
    c0_vec = C_to_chains[:, conduction_indices[0]] if len_c > 0 else np.zeros(M)
    # The vector for the head of the valence chain, |v0> = |B>, in the original basis
    v0_vec = C_to_chains[:, valence_indices[0]] if len_v > 0 else np.zeros(M)

    C_total = np.zeros((M, M))
    
    # The first two columns of C_total are the impurity |i> and special NO |b>
    # We recover them by rotating |c0> and |v0> back with U_bond.T
    # |i> = U_bond[0,0]|c0> + U_bond[0,1]|v0>
    # |b> = U_bond[1,0]|c0> + U_bond[1,1]|v0>
    C_total[:, 0] = U_bond[0, 0] * c0_vec + U_bond[0, 1] * v0_vec
    C_total[:, 1] = U_bond[1, 0] * c0_vec + U_bond[1, 1] * v0_vec

    # The remaining columns are the rest of the Lanczos chain vectors
    c_chain_start_idx = 2
    if len_c > 1:
        c_rest_indices_in_decoupled_basis = [conduction_indices[i] for i in range(1, len_c)]
        C_total[:, c_chain_start_idx : c_chain_start_idx + len_c - 1] = C_to_chains[:, c_rest_indices_in_decoupled_basis]

    v_chain_start_idx = c_chain_start_idx + (len_c - 1 if len_c > 0 else 0)
    if len_v > 1:
        v_rest_indices_in_decoupled_basis = [valence_indices[i] for i in range(1, len_v)]
        C_total[:, v_chain_start_idx : v_chain_start_idx + len_v - 1] = C_to_chains[:, v_rest_indices_in_decoupled_basis]

    # --- Transform the Hamiltonian and correct for the mean-field shift ---
    h_mf_final_basis = C_total.T @ h_mf @ C_total
    
    #mean_field_correction_term = np.zeros_like(h_spin)
    #mean_field_correction_term[0, 0] = u * 0.5
    #transformed_correction = C_total.T @ mean_field_correction_term @ C_total
    
    h_final_matrix = h_mf_final_basis #- transformed_correction
    
    return h_final_matrix, C_total


# ------------------------------------------------------------
# Multi-orbital 
# ------------------------------------------------------------

def restore_original_impurity(C_total, Nimp, na, nf, ne, r_svd,
                              conduction_seeds, valence_seeds,
                              cond_indices, val_indices,
                              Ub_pairs, Vimp):
    """
    Right-multiply C_total by a unitary so that columns 0..Nimp-1 are exactly
    the original impurity basis vectors. Does:
      1) Undo the 2x2 bonding/antibonding per active pair on the chain heads
      2) Undo the impurity Schmidt rotation Vimp
      3) Permute columns to put impurity columns first
    """
    M = C_total.shape[0]
    R = np.eye(M, dtype=C_total.dtype)

    # 1) Undo Ub on the chain heads for each paired channel
    for alpha in range(r_svd):
        jv = val_indices[alpha]   # valence head position
        jc = cond_indices[alpha]  # conduction head position
        U2 = Ub_pairs[alpha]      # 2x2 used earlier
        R_pair = np.eye(M, dtype=C_total.dtype)
        # multiply the two columns [jv, jc] on the right by U2 to return to [imp_Schmidt, act_Schmidt]
        R_pair[np.ix_([jv, jc], [jv, jc])] = U2
        R = R @ R_pair

    # 2) Identify impurity Schmidt columns then apply Vimp^† on those columns
    imp_cols = [val_indices[alpha] for alpha in range(r_svd)]
    for alpha in range(r_svd, Nimp):
        if alpha in valence_seeds:
            pos = valence_seeds.index(alpha)
            imp_cols.append(val_indices[pos])
        else:
            pos = conduction_seeds.index(alpha)
            imp_cols.append(cond_indices[pos])

    R_imp = np.eye(M, dtype=C_total.dtype)
    R_imp[np.ix_(imp_cols, imp_cols)] = Vimp.conj().T
    R = R @ R_imp

    # 3) Permute columns so that impurity columns come first, preserving order
    remaining = [j for j in range(M) if j not in imp_cols]
    target_order = imp_cols + remaining
    P = np.eye(M, dtype=C_total.dtype)[:, target_order]
    R = R @ P

    C_fixed = C_total @ R
    return C_fixed


def get_double_chain_transform_multi(h, Nimp, Nelec, tol_occ=1e-8):
    """
    Multiorbital double-chain transform.

    Steps
      1) MF density from h
      2) Rotate bath to NOs, split into active, filled, empty
      3) Schmidt pairing between impurity and active bath (SVD)
      4) Per-pair 2x2 rotation to get conduction and valence heads
      5) Block Lanczos on conduction and valence halves
      6) Restore the original impurity exactly

    Returns
      h_final, C_total, meta
    """

    M = h.shape[0]                                              # total one-body dimension
    assert h.shape == (M, M) and 1 <= Nimp < M                  # sanity checks

    # ---- small utilities ----
    def classify_bath(rho_bb):
        occ, W = eigh(rho_bb)                                   # bath density eigenbasis (NOs)
        # split bath NOs by occupation: fractional = "active", ~1 = filled, ~0 = empty
        filled = [i for i, n in enumerate(occ) if n > 1 - tol_occ]
        empty  = [i for i, n in enumerate(occ) if n < tol_occ]
        active = [i for i, n in enumerate(occ) if tol_occ <= n <= 1 - tol_occ]
        return occ, W, active, filled, empty

    def pad_with_identity(Q, n):
        m = Q.shape[1]                                          # Q is n×m with orthonormal columns
        if m == n:                                              # already square/unitary in subspace
            return Q
        out = np.eye(n, dtype=Q.dtype)                          # extend to a square unitary by appending basis vectors
        out[:, :m] = Q
        return out
    
    # At the starting point, h can be in general dense
    #   IMP   BATH 
    #   himp  Vib
    #   .     hb

    # ---- (1) MF density ----
    es, vs = eigh(h)                                            # single-particle eigenpairs of h (Hermitian)
    rho = vs[:, :Nelec] @ vs[:, :Nelec].conj().T                # ρ = Σ_{occ} |ψ⟩⟨ψ| (projector onto lowest Nelec)

    # ---- (2) Bath NOs and deterministic order ----
    occ_b, W, active_idx, filled_idx, empty_idx = classify_bath(rho[Nimp:, Nimp:])
    na, nf, ne = len(active_idx), len(filled_idx), len(empty_idx)
    assert na + nf + ne == M - Nimp                             # partition covers the bath

    order_bath = active_idx + filled_idx + empty_idx            # fix an ordering: active first (to pair), then filled, empty
    C1 = block_diag(np.eye(Nimp, dtype=h.dtype), W[:, order_bath])  # rotate only the bath by W (impurity untouched)
    rho1 = C1.conj().T @ rho @ C1                               # transform density to bath-NO basis
    h1   = C1.conj().T @ h   @ C1                               # transform Hamiltonian similarly
    # Here, h1 has the form 
    #  IMP  ACTIVE  FILLED EMPTY
    #  himp hia     hif     hie
    #       ha      haf     hae  
    #               hf      0
    #                       he
    # himp is untouched, all other are changed

    if na == 0:                                                 # no fractional bath states ⇒ nothing to pair or chain
        return h1, C1, dict(r=0, na=0, nf=nf, ne=ne, order_bath=order_bath)

    # ---- (3) Schmidt pairing on active block ----
    a0 = Nimp                                                   # start index of bath in current basis
    C = rho1[a0:a0+na, :Nimp]                                   # impurity–active-bath cross-block of ρ (na × Nimp)

    # SVD gives matched impurity/bath Schmidt directions: C = Wa Σ Vimp†
    Wa, svals, Vimp_dag = svd(C, full_matrices=False)           # thin SVD (min(na, Nimp) columns)
    Vimp = Vimp_dag.conj().T                                    # right singular vectors (impurity rotation)

    r_svd = int(min(np.sum(svals > 1e-12), min(na, Nimp)))      # numerical rank capped by available pairs

    # rotate impurity by Vimp and active bath by Wa; filled/empty bath unchanged
    C2 = block_diag(Vimp, block_diag(Wa, np.eye(nf + ne, dtype=h.dtype)))
    rho2 = C2.conj().T @ rho1 @ C2
    h2   = C2.conj().T @ h1   @ C2

    # The goal here is to make the coupling between the impurity and the active space diagonal in the 
    # density matrix. So, after the svd, rho2 has the block rho2_ia diagonal
    # This defines natural pairs of orbitals to further diagonalize the IMP-ACTIVE rho subspace by blocks of pairs
    # Now h2 has the form 
    #  Imp-Act-mixture  FILLED EMPTY
    #  hiam            hiam-f  hiam-e
    #                   hf      0
    #                           he
    # The blocks filled and empty are untouched 

    # ---- (4) Per-pair 2×2 bonding/antibonding on the r_svd heads ----
    Ub = np.eye(M, dtype=h.dtype)                               # accumulate pairwise 2×2 rotations
    Ub_pairs = []                                                # keep the 2×2s for the later exact restore
    for alpha in range(r_svd):
        i_idx, a_idx = alpha, Nimp + alpha                      # indices of the impurity/bath Schmidt pair in current basis
        rho2_pair = rho2[np.ix_([i_idx, a_idx], [i_idx, a_idx])]# 2×2 density restricted to that pair
        evals, U2 = eigh(rho2_pair)                             # diagonalize ⇒ more/less occupied directions
        U2 = U2[:, np.argsort(evals)[::-1]]                     # order so col 0 is the more filled ("valence-like")
        Ub[np.ix_([i_idx, a_idx], [i_idx, a_idx])] = U2         # embed 2×2 into the big unitary
        Ub_pairs.append(U2)

    C3  = Ub                                                     # per-pair rotation unitary
    rho3 = C3.conj().T @ rho2 @ C3                               # apply to ρ
    h3   = C3.conj().T @ h2   @ C3                               # and to h

    # Now, h3 has the form 
    #  IA-FILLED IA-EMPTY  FILLED   EMPTY
    #  hiaf         0       hiaf-f  0
    #               hiae    0       hiae-e
    #                       hf      0
    #                               he
    # hf and he still untouched, the hamiltonian is now fully block diagonal 
    # with a filled and an empty block
    # The two blocks can be made tridiagonal separately

    # ---- (5) Build halves and run block Lanczos ----
    # remaining bath indices are bath NOs: first 'filled' then 'empty' per order_bath
    filled_block = list(range(Nimp + na, Nimp + na + nf))        # filled-bath tail of the valence half
    empty_block  = list(range(Nimp + na + nf, M))                # empty-bath tail of the conduction half

    valence_seeds    = [alpha for alpha in range(r_svd)]         # the "more filled" heads (col 0 of each pair)
    conduction_seeds = [Nimp + alpha for alpha in range(r_svd)]  # the "less filled" heads

    # leftover impurity directions (if Nimp > r_svd): send by diagonal occupation
    for alpha in range(r_svd, Nimp):
        if float(np.real(rho3[alpha, alpha])) > 0.5:
            valence_seeds.append(alpha)                          # > 0.5 ⇒ valence half
        else:
            conduction_seeds.append(alpha)                       # ≤ 0.5 ⇒ conduction half

    val_indices  = valence_seeds + filled_block                  # full index list of valence half
    cond_indices = conduction_seeds + empty_block                # full index list of conduction half

    r_v, r_c = len(valence_seeds), len(conduction_seeds)         # initial block sizes for block Lanczos

    Hv = h3[np.ix_(val_indices,  val_indices)]                   # project h to valence half
    Hc = h3[np.ix_(cond_indices, cond_indices)]                  # project h to conduction half

    # block Lanczos: returns block-tridiagonal factors (A,B) and the basis Q (orthonormal columns)
    _, _, Qv, _ = clic.block_lanczos_matrix(Hv, r_v)                # Qv: valence chain basis (tall)
    _, _, Qc, _ = clic.block_lanczos_matrix(Hc, r_c)                # Qc: conduction chain basis (tall)

    Qv_embed = pad_with_identity(Qv, len(val_indices))           # embed each Q as a square unitary on its subspace
    Qc_embed = pad_with_identity(Qc, len(cond_indices))

    C4 = np.eye(M, dtype=h.dtype)                                # assemble subspace unitaries
    C4[np.ix_(val_indices,  val_indices )] = Qv_embed
    C4[np.ix_(cond_indices, cond_indices)] = Qc_embed

    # compose all steps before the final restore
    C_total = C1 @ C2 @ C3 @ C4

    # ---- (6) Exact restore of original impurity ----
    # invert per-pair rotations on the heads, undo Vimp on the impurity-Schmidt block,
    # then permute columns to place the original impurity first. This is delegated.
    C_total = restore_original_impurity(
        C_total, Nimp, na, nf, ne, r_svd,
        conduction_seeds, valence_seeds,
        cond_indices, val_indices,
        Ub_pairs, Vimp
    )

    h_final = C_total.conj().T @ h @ C_total                    # final Hamiltonian in double-chain basis (impurity restored)

    meta = dict(                                                # bookkeeping for diagnostics/restarts
        r=r_svd, na=na, nf=nf, ne=ne,
        order_bath=order_bath,
        conduction_seeds=conduction_seeds,
        valence_seeds=valence_seeds,
        cond_indices=cond_indices,
        val_indices=val_indices
    )
    return h_final, C_total, meta


# ------------------------------------------------------------
# Block - Sym 
# ------------------------------------------------------------

def double_chain_by_blocks(
    h: np.ndarray,
    rho: np.ndarray,
    Nimp: int,
    Nelec: int,
    analyze_symmetries_fn,
    transform_fn,                 # e.g. get_double_chain_transform_multi
    tol_occ: float = 1e-8,
    verbose: bool = False,
):
    """
    Apply the multiorbital double-chain transform blockwise, preserving and mirroring
    exact block symmetries. For identical blocks, reuse the leader's transform.

    Inputs
      h:        M x M Hermitian one-body Hamiltonian whose Nimp first index will be treated as the impurity
      rho:      M x M 1-rdm for h with Nelec
      Nimp:     number of impurity orbitals at global indices [0..Nimp-1]
      Nelec:    electron count in this spin sector (global)
      analyze_symmetries_fn: callable(h) -> {"blocks": [...], "identical_groups": [...], ...}
      transform_fn: callable(h_block, Nimp_block, Nelec_block, tol_occ=...) -> (h_final_block, C_block, meta)
      tol_occ:  occupancy threshold forwarded to transform_fn
      verbose:  optional prints

    Returns
      h_final:  M x M Hamiltonian after blockwise double-chain transforms
      C_total:  M x M global unitary; h_final = C_total^† h C_total
      meta:     dict with per-block metadata
    """
    M = h.shape[0]
    assert h.shape == (M, M)
    assert 1 <= Nimp <= M

    # 0) Analyze block structure once
    sym = analyze_symmetries_fn(h, verbose=verbose)
    blocks = sym["blocks"]                  # list[list[int]]
    identical_groups = sym["identical_groups"]  # list[list[block_index]]
    if verbose:
        print(f"[blocks] {len(blocks)} blocks; identical groups: {identical_groups}")


    # 2) Prepare outputs
    C_total = np.eye(M, dtype=h.dtype)      # global unitary, block-diagonal fill
    h_final = np.zeros_like(h)              # we will place each transformed block
    block_results = []                      # per-block bookkeeping
    leader_unitaries = {}                   # map leader block idx -> (C_block, h_block_final, meta)

    # Helper: extract submatrix by index list
    def submat(A, idx):
        return A[np.ix_(idx, idx)]

    # Helper: place a submatrix B into A at idx positions (Hermitian block)
    def place_block(A, idx, B):
        A[np.ix_(idx, idx)] = B

    # Helper: count impurity orbitals in a block
    imp_set = set(range(Nimp))
    def count_block_impurity(idx_list):
        return sum(1 for j in idx_list if j in imp_set)

    # Helper: compute integer electron count in a block from rho trace
    def electrons_in_block(idx_list):
        tr = np.real_if_close(np.trace(submat(rho, idx_list)))
        # round to nearest integer for stability
        return int(np.rint(float(tr)))
    
    # electron counts per block from the provided rho
    electrons_per_block = np.array([electrons_in_block(idx) for idx in blocks], dtype=float)
    if verbose:
        print(f"[check] Tr rho = {np.trace(rho).real:.12f}, sum blocks = {electrons_per_block.sum():.12f}, Nelec = {Nelec}")

    # 3) Iterate identical groups; compute once per leader
    for group in identical_groups:
        leader = group[0]                           # choose first block in the group as leader
        leader_idx = blocks[leader]
        # Per-block impurity and electrons
        Nimp_b = count_block_impurity(leader_idx)
        Nelec_b = electrons_in_block(leader_idx)

        if verbose:
            print(f"[leader block {leader}] size={len(leader_idx)} Nimp_b={Nimp_b} Nelec_b={Nelec_b}")

        if Nimp_b == 0:
            # No impurity content in this block. Do nothing: C_block = identity, h_block_final = h_block.
            h_block = submat(h, leader_idx)
            C_block = np.eye(len(leader_idx), dtype=h.dtype)
            h_block_final = h_block.copy()
            meta = dict(note="no_impurity_in_block", size=len(leader_idx))
        else:
            # Run the double-chain transform on the leader block
            h_block = submat(h, leader_idx)
            h_block_final, C_block, meta = transform_fn(h_block, Nimp_b, Nelec_b, tol_occ=tol_occ)

        # Cache leader result
        leader_unitaries[leader] = (C_block, h_block_final, meta)

        # Install into global matrices for the leader
        place_block(C_total, leader_idx, C_block)
        place_block(h_final, leader_idx, h_block_final)

        block_results.append(dict(block_id=leader, leader=True, indices=leader_idx,
                                  Nimp_b=Nimp_b, Nelec_b=Nelec_b, meta=meta))

        # 4) Mirror to the followers in the identical group
        for follower in group[1:]:
            idx = blocks[follower]
            # Safety: blocks in identical group must have same size and same Nimp_b
            assert len(idx) == len(leader_idx), "Identical blocks must have equal size"
            Nimp_f = count_block_impurity(idx)
            assert Nimp_f == Nimp_b, "Identical blocks must contain equal impurity size"
            Nelec_f = electrons_in_block(idx)
            # We reuse the exact same unitary and h_block_final to preserve symmetry
            C_f, h_f, meta_f = C_block, h_block_final, dict(meta)  # shallow copy ok

            place_block(C_total, idx, C_f)
            place_block(h_final, idx, h_f)

            block_results.append(dict(block_id=follower, leader=False, indices=idx,
                                      Nimp_b=Nimp_f, Nelec_b=Nelec_f, meta=meta_f))

    # 5) Sanity: h_final should equal C_total^† h C_total to numerical precision
    if verbose:
        check = C_total.conj().T @ h @ C_total
        err = np.linalg.norm(h_final - check) / max(1.0, np.linalg.norm(h))
        print(f"[sanity] ||h_final - C^† h C|| / ||h|| = {err:.3e}")

    meta_global = dict(
        blocks=blocks,
        identical_groups=identical_groups,
        per_block=block_results
    )
    return h_final, C_total, meta_global
