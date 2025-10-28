# bath_transform.py

import numpy as np
from scipy.linalg import eigh, block_diag
from . import symmetries, mf
# --------------------------------------------------------------------------
# Double chain for a single impurity
# --------------------------------------------------------------------------

def lanczos_tridiagonalization(H, v0):
    """
    Performs the Lanczos algorithm to tridiagonalize a Hermitian matrix H.
    Returns the tridiagonal matrix T and the transformation matrix Q such that H Q = Q T.
    """
    N = H.shape[0]
    if N == 0:
        return np.array([[]]), np.array([[]])
        
    Q = np.zeros((N, N), dtype=float)
    norm_v0 = np.linalg.norm(v0)
    if norm_v0 < 1e-15:
        # If v0 is zero (e.g., empty subspace), return empty matrices
        return np.array([[]]), np.array([[]])
    v = v0 / norm_v0
    Q[:, 0] = v

    alphas = np.zeros(N)
    betas = np.zeros(N - 1)

    for j in range(N):
        w = H @ v
        alpha = np.dot(v.conj(), w)
        alphas[j] = alpha

        if j < N - 1:
            w = w - alpha * v
            if j > 0:
                w = w - betas[j-1] * Q[:, j-1]
            
            beta = np.linalg.norm(w)
            
            if beta < 1e-12: # Invariant subspace found
                N_effective = j + 1
                T = np.diag(alphas[:N_effective]) + np.diag(betas[:j], k=1) + np.diag(betas[:j], k=-1)
                return T, Q[:, :N_effective]
            
            betas[j] = beta
            v = w / beta
            Q[:, j + 1] = v
    
    T = np.diag(alphas) + np.diag(betas, k=1) + np.diag(betas, k=-1)
    return T, Q


def get_double_chain_transform(h_spin, u, Nelec):
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
    h_mf[0, 0] += u * 0.5

    # --- (ii) Natural Orbital Basis for the Bath ---
    e_mf, C_mf = eigh(h_mf)
    rho_mf = C_mf[:, :Nelec] @ C_mf[:, :Nelec].T
    rho_bath = rho_mf[1:, 1:]
    n_no, W = eigh(rho_bath)
    
    occupations_dist_from_integer = np.minimum(n_no, 1 - n_no)
    b_idx = np.argmax(occupations_dist_from_integer)
    
    other_indices = [i for i in range(M - 1) if i != b_idx]
    # Sorting ensures a deterministic basis ordering
    filled_indices = sorted([i for i in other_indices if n_no[i] > 0.5])
    empty_indices = sorted([i for i in other_indices if n_no[i] <= 0.5])
    
    ordered_bath_indices = [b_idx] + filled_indices + empty_indices
    W_ordered = W[:, ordered_bath_indices]
    C1 = block_diag(1, W_ordered)

    # --- (iii) Bonding/Anti-bonding Transformation ---
    rho_no_basis = C1.T @ rho_mf @ C1
    rho_ib = rho_no_basis[:2, :2]
    _, U_bond = eigh(rho_ib)
    C2 = np.identity(M)
    C2[:2, :2] = U_bond
    
    C_upto_decoupled = C1 @ C2

    # --- (iv) Lanczos Tridiagonalization (Chain Transformation) ---
    h_decoupled = C_upto_decoupled.T @ h_mf @ C_upto_decoupled

    num_filled = len(filled_indices)
    conduction_indices = [0] + list(range(2 + num_filled, M))
    valence_indices = [1] + list(range(2, 2 + num_filled))

    h_conduction = h_decoupled[np.ix_(conduction_indices, conduction_indices)]
    v0_c = np.zeros(h_conduction.shape[0]); v0_c[0] = 1.0
    T_c, Q_c = lanczos_tridiagonalization(h_conduction, v0_c)

    h_valence = h_decoupled[np.ix_(valence_indices, valence_indices)]
    v0_v = np.zeros(h_valence.shape[0]); v0_v[0] = 1.0
    T_v, Q_v = lanczos_tridiagonalization(h_valence, v0_v)
    
    C_lanczos = np.identity(M)
    if Q_c.shape[1] > 0:
      C_lanczos[np.ix_(conduction_indices, conduction_indices)] = Q_c
    if Q_v.shape[1] > 0:
      C_lanczos[np.ix_(valence_indices, valence_indices)] = Q_v
      
    # C_to_chains transforms from original basis to the basis of Lanczos vectors
    # (permuted).
    C_to_chains = C_upto_decoupled @ C_lanczos

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
    
    mean_field_correction_term = np.zeros_like(h_spin)
    mean_field_correction_term[0, 0] = u * 0.5
    transformed_correction = C_total.T @ mean_field_correction_term @ C_total
    
    h_final_matrix = h_mf_final_basis - transformed_correction
    
    return h_final_matrix, C_total

# --------------------------------------------------------------------------
# Double chain for a multi-orbital case 
# --------------------------------------------------------------------------
from .mf import mfscf 


def perform_multi_orbital_no_transform(h0, U, block_dict, impurity_indices, Ne_per_block):
    """
    Performs the natural orbital to double-chain transformation for a multi-orbital
    problem that is already block-diagonalized by symmetry.

    Args:
        h0 (np.ndarray): Full one-particle Hamiltonian (e.g., 2M x 2M), assumed block-diagonal.
        U (np.ndarray): Full two-particle interaction tensor, assumed block-diagonal.
        block_dict (dict): Maps symmetry block name to list of indices for one spin channel.
                           Example: {"a2u": [0, 5, 8], "t1u": [1, 2, 6, 7]}
        impurity_indices (list): List of global indices for all impurity orbitals.
        Ne_per_block (dict): Maps block name to the number of electrons for that block
                             (for a single spin channel).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - h_final (np.ndarray): The final Hamiltonian in the double-chain basis.
            - C_total (np.ndarray): The total unitary transformation matrix.
    """
    M = h0.shape[0] // 2 # Size of one spin channel
    h_final = np.zeros_like(h0, dtype=np.complex128)
    C_total = np.identity(h0.shape[0], dtype=np.complex128)

    # Process spin-up and spin-down blocks separately but identically
    for spin_offset in [0, M]: # 0 for spin-up, M for spin-down
        
        for block_name, global_indices in block_dict.items():
            
            current_indices = [i + spin_offset for i in global_indices]
            if not current_indices: continue

            h0_block = h0[np.ix_(current_indices, current_indices)]
            U_block = U[np.ix_(current_indices, current_indices, current_indices, current_indices)]
            Ne_block = Ne_per_block[block_name]

            block_imp_indices_global = [i for i in current_indices if i in impurity_indices]
            imp_indices_local = [global_indices.index(i - spin_offset) for i in block_imp_indices_global]

            print(f"\n--- Processing block '{block_name}' (Spin {'down' if spin_offset else 'up'}) ---")
            
            h_final_b, C_b = _transform_one_block(h0_block, U_block, Ne_block, imp_indices_local)

            h_final[np.ix_(current_indices, current_indices)] = h_final_b
            C_total[np.ix_(current_indices, current_indices)] = C_b
            
    return h_final, C_total


def _transform_one_block(h0_block, U_block, Ne_block, imp_indices_local):
    """
    Worker function to perform the 5-step transformation on a single symmetry block.
    """
    N_block = h0_block.shape[0]
    N_imp = len(imp_indices_local)

    if N_block == 0:
        return np.array([[]]), np.array([[]])
    if N_imp == 0 or N_block <= N_imp: # No bath to transform
        return h0_block, np.identity(N_block)

    # --- Step i: Mean-Field SCF on the block ---
    h0_doubled = block_diag(h0_block, h0_block)
    U_doubled = np.zeros((2*N_block, 2*N_block, 2*N_block, 2*N_block), dtype=U_block.dtype)
    U_doubled[np.ix_(*[np.arange(N_block)]*4)] = U_block
    U_doubled[np.ix_(*[np.arange(N_block, 2*N_block)]*4)] = U_block

    hmf_up, _, _, rho_full = mfscf(h0_doubled, U_doubled, 2 * Ne_block, maxiter=50)
    rho_up = rho_full[:N_block, :N_block]
    
    # --- Step ii: Natural Orbitals ---
    bath_indices_local = [i for i in range(N_block) if i not in imp_indices_local]
    rho_bath = rho_up[np.ix_(bath_indices_local, bath_indices_local)]
    n_no, W = eigh(rho_bath)
    
    occupations_dist = np.minimum(n_no, 1.0 - n_no)
    special_b_indices_in_bath = np.argsort(occupations_dist)[::-1][:N_imp]
    
    other_indices = np.setdiff1d(np.arange(len(bath_indices_local)), special_b_indices_in_bath)
    filled_indices = sorted([i for i in other_indices if n_no[i] > 0.5])
    empty_indices = sorted([i for i in other_indices if n_no[i] <= 0.5])

    ordered_bath_indices = list(special_b_indices_in_bath) + filled_indices + empty_indices
    W_ordered = W[:, ordered_bath_indices]

    C1 = np.identity(N_block, dtype=h0_block.dtype)
    C1[np.ix_(bath_indices_local, bath_indices_local)] = W_ordered
    
    # --- Step iii: Generalized Bonding/Anti-bonding ---
    rho_no_basis = C1.conj().T @ rho_up @ C1
    ib_subspace_indices = list(imp_indices_local) + list(range(N_imp, 2 * N_imp))
    rho_ib = rho_no_basis[np.ix_(ib_subspace_indices, ib_subspace_indices)]
    
    # Sort by occupation to identify bonding/anti-bonding
    n_ab, U_bond_unordered = eigh(rho_ib)
    ab_order = np.argsort(n_ab)
    U_bond = U_bond_unordered[:, ab_order] # Columns are anti-bonding then bonding
    
    C2 = np.identity(N_block, dtype=h0_block.dtype)
    C2[np.ix_(ib_subspace_indices, ib_subspace_indices)] = U_bond
    
    C_upto_decoupled = C1 @ C2

    # --- Step iv: Lanczos Chains for each channel ---
    h_decoupled = C_upto_decoupled.conj().T @ hmf_up @ C_upto_decoupled

    # We will create N_imp pairs of chains
    Q_c_list, Q_v_list = [], []
    conduction_spaces, valence_spaces = [], []

    num_empty_per_chain = len(empty_indices) // N_imp
    num_filled_per_chain = len(filled_indices) // N_imp

    C_lanczos = np.identity(N_block, dtype=h0_block.dtype)

    for k in range(N_imp):
        # Conduction chain for channel k
        anti_bond_idx = k
        empty_start = 2 * N_imp + len(filled_indices) + k * num_empty_per_chain
        empty_end = empty_start + num_empty_per_chain
        # Add remainder to last chain
        if k == N_imp - 1: empty_end = 2 * N_imp + len(filled_indices) + len(empty_indices)
        
        c_indices = [anti_bond_idx] + list(range(empty_start, empty_end))
        conduction_spaces.append(c_indices)
        
        h_c = h_decoupled[np.ix_(c_indices, c_indices)]
        v0_c = np.zeros(h_c.shape[0]); v0_c[0] = 1.0
        _, Q_c = lanczos_tridiagonalization(np.real(h_c), v0_c)
        Q_c_list.append(Q_c)
        if Q_c.shape[1] > 0:
            C_lanczos[np.ix_(c_indices, c_indices)] = Q_c

        # Valence chain for channel k
        bond_idx = N_imp + k
        filled_start = 2 * N_imp + k * num_filled_per_chain
        filled_end = filled_start + num_filled_per_chain
        if k == N_imp - 1: filled_end = 2 * N_imp + len(filled_indices)
        
        v_indices = [bond_idx] + list(range(filled_start, filled_end))
        valence_spaces.append(v_indices)

        h_v = h_decoupled[np.ix_(v_indices, v_indices)]
        v0_v = np.zeros(h_v.shape[0]); v0_v[0] = 1.0
        _, Q_v = lanczos_tridiagonalization(np.real(h_v), v0_v)
        Q_v_list.append(Q_v)
        if Q_v.shape[1] > 0:
            C_lanczos[np.ix_(v_indices, v_indices)] = Q_v

    C_to_chains = C_upto_decoupled @ C_lanczos

    # --- Step v: Final Assembly ---
    C_block_final = np.zeros((N_block, N_block), dtype=h0_block.dtype)
    
    # Recover original impurity and b orbitals by inverting the bonding transform
    # The heads of the chains are the A and B orbitals.
    c0_vectors = C_to_chains[:, [space[0] for space in conduction_spaces]] # Columns are |A_k>
    v0_vectors = C_to_chains[:, [space[0] for space in valence_spaces]]   # Columns are |B_k>
    
    # This matrix holds |A_1>...|A_N>, |B_1>...|B_N> as columns
    AB_vectors = np.hstack([c0_vectors, v0_vectors])
    # Rotate back to get |i_1>...|i_N>, |b_1>...|b_N>
    ib_vectors = AB_vectors @ U_bond.conj().T
    
    final_idx = 0
    # Impurity orbitals
    C_block_final[:, final_idx : final_idx + N_imp] = ib_vectors[:, :N_imp]
    final_idx += N_imp
    # Special b orbitals
    C_block_final[:, final_idx : final_idx + N_imp] = ib_vectors[:, N_imp:]
    final_idx += N_imp
    
    # Add the rest of the chain vectors
    for k in range(N_imp):
        space_indices = conduction_spaces[k]
        Q_c = Q_c_list[k]
        if Q_c.shape[1] > 1:
            num_chain_sites = Q_c.shape[1] - 1
            original_basis_indices = [space_indices[i] for i in range(1, Q_c.shape[1])]
            C_block_final[:, final_idx : final_idx + num_chain_sites] = C_to_chains[:, original_basis_indices]
            final_idx += num_chain_sites

    for k in range(N_imp):
        space_indices = valence_spaces[k]
        Q_v = Q_v_list[k]
        if Q_v.shape[1] > 1:
            num_chain_sites = Q_v.shape[1] - 1
            original_basis_indices = [space_indices[i] for i in range(1, Q_v.shape[1])]
            C_block_final[:, final_idx : final_idx + num_chain_sites] = C_to_chains[:, original_basis_indices]
            final_idx += num_chain_sites

    # Transform hmf and correct for the mean-field potential to get the transformed h0
    Vmf_up = hmf_up - h0_block
    h_final_block = (C_block_final.conj().T @ hmf_up @ C_block_final) - \
                    (C_block_final.conj().T @ Vmf_up @ C_block_final)

    return h_final_block, C_block_final

# --------------------------------------------------------------------------
# Natural orbitals 
# --------------------------------------------------------------------------

def get_natural_orbital_transform(h_spin, u, Nelec):
    """
    Performs a simple Natural Orbital transformation
    This corresponds to steps (i) and (ii) of the more complex procedure.
    """
    M = h_spin.shape[0]

    # Step 1: Mean-field Hamiltonian and its density matrix
    h_mf = h_spin.copy()
    h_mf[0, 0] += u * 0.5
    
    e_mf, C_mf = eigh(h_mf)
    rho_mf = C_mf[:, :Nelec] @ C_mf[:, :Nelec].T

    # Step 2: Diagonalize the bath part of the density matrix
    rho_bath = rho_mf[1:, 1:]
    # W contains the eigenvectors of rho_bath, which define the new bath basis
    _, W = eigh(rho_bath)
    
    # The transformation matrix C_no leaves the impurity alone and transforms the bath
    C_no = block_diag(1, W)
    
    # Apply the transformation to the original non-interacting Hamiltonian
    h_new_basis = C_no.T @ h_spin @ C_no
    
    return h_new_basis, C_no

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Natural Orbitals for a multi-orbital case
# --------------------------------------------------------------------------

def get_multi_orbital_natural_orbital_transform(
    h0: np.ndarray, 
    U: np.ndarray, 
    Nelec: int,
    impurity_indices: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a symmetry-preserving natural orbital transformation for a
    multi-orbital impurity problem.

    The transformation finds the natural orbitals for the bath *within each
    symmetry block* of the Hamiltonian, leaving the impurity orbitals untouched.
    This is the multi-orbital generalization of `get_natural_orbital_transform`.

    Args:
        h0 (np.ndarray): Full one-particle Hamiltonian (2M x 2M, AlphaFirst).
        U (np.ndarray): Full two-particle interaction tensor (2M x 2M x 2M x 2M).
        Nelec (int): The total number of electrons in the system.
        impurity_indices (list[int]): List of SPATIAL indices for all impurity
                                      orbitals (e.g., [0, 1] for a 2-orbital impurity).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - h_final (np.ndarray): The final Hamiltonian in the natural orbital basis.
            - C_total (np.ndarray): The total unitary transformation matrix (2M x 2M).
    """
    M = h0.shape[0] // 2
    h0_spin = h0[:M, :M]
    impurity_indices = sorted(list(set(impurity_indices)))
    
    print("--- Starting Symmetry-Aware Natural Orbital Transformation ---")
    
    # 1. Run SCF to get the converged density matrix
    print("Step 1: Running mean-field SCF to get the density matrix...")
    _, _, _, rho = mf.mfscf(h0, U, Nelec, maxiter=50)
    rho_spin = rho[:M, :M] # Work with the spatial (spin-up) block
    
    # 2. Analyze the symmetry of the original h0
    print("Step 2: Analyzing symmetries of the initial Hamiltonian...")
    sym_dict = symmetries.analyze_symmetries(h0_spin)
    blocks = sym_dict['blocks']
    print(f"Found {len(blocks)} symmetry blocks.")

    # 3. Build the transformation matrix C block-by-block
    print("Step 3: Calculating natural orbitals for each symmetry block...")
    C_spin = np.identity(M, dtype=h0.dtype)
    
    for i, block_indices in enumerate(blocks):
        # Partition the orbitals within this block into impurity and bath
        imp_in_block = [idx for idx in block_indices if idx in impurity_indices]
        bath_in_block = [idx for idx in block_indices if idx not in impurity_indices]
        
        print(f"  - Processing Block {i} (size {len(block_indices)}): "
              f"{len(imp_in_block)} impurity, {len(bath_in_block)} bath orbitals.")

        if not bath_in_block:
            # If no bath orbitals in this block, no transformation is needed.
            continue
            
        # Extract the bath-bath submatrix of the density matrix for this block
        rho_bath_block = rho_spin[np.ix_(bath_in_block, bath_in_block)]
        
        # Diagonalize it to get the eigenvectors 'W', which define the new bath basis
        _, W_block = eigh(rho_bath_block)
        
        # Place the transformation 'W_block' into the correct slice of C_spin.
        # This transforms the bath orbitals of this block, leaving others untouched.
        C_spin[np.ix_(bath_in_block, bath_in_block)] = W_block
        
    # 4. Assemble the full 2M x 2M transformation matrix and apply it
    print("Step 4: Assembling final transformation matrix and transforming h0...")
    C_total = block_diag(C_spin, C_spin)
    h_final = C_total.conj().T @ h0 @ C_total
    
    print("--- Transformation complete. ---")
    
    return h_final, C_total