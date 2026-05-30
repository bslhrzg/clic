#clic/green/green_sym.py

import numpy as np

from clic.symmetries import symmetries
from clic.green import gfs

def get_green_block(
        M, # number of spatial orbitals
        psi_n, # Wavefunction
        e_n,   # Energy of psi_n 
        NappH, # number of application of H to produce fixed basis
        eta,    # distance to imaginary axis
        h0_n, U_n, # one and two particle hamiltonians
        ws, # frequency mesh
        iws,
        target_indices, # indexes for which we want G_{ij}
        one_bh_n,two_bh_n, # one and two body tables
        tables_n, 
        coeff_thresh,  # threshold to prune states 
        L,   # number of lanczos iterations
        green_diag_only = False,
    ): 
    


    # --- Symmetry Analysis (done once per thermal state) ---
    sub_h0 = h0_n[np.ix_(target_indices, target_indices)]
    symdict = symmetries.analyze_symmetries(sub_h0)
    
    blocks = symdict["blocks"] 
    identical_groups = symdict["identical_groups"]
    is_diagonal = symdict["is_diagonal"]

    if is_diagonal or green_diag_only:
        gfmeth = "scalar_continued_fraction"
    else : 
        gfmeth = "block"

    #if not is_diagonal:
    #    symdict = symmetries.construct_spin_only_symdict(np.shape(sub_h0)[0])
    #    blocks = symdict["blocks"] 
    #    identical_groups = symdict["identical_groups"]
    #    is_diagonal = symdict["is_diagonal"]
    
    print(f" Founds blocks : {blocks}")
    print(f"  Symmetry: is_diagonal={is_diagonal}. Found {len(identical_groups)} unique group(s) of blocks.")
    print(f"  identical groups: {identical_groups}")

    # This will store the full computed block for the current thermal state, using LOCAL indices.
    num_target = len(target_indices)
    G_sub_block_n = np.zeros((len(ws), num_target, num_target), dtype=np.complex128)

    print(f"DEBUG: here in green_block : iws is not None = {iws is not None}")
    if iws is not None: 
        G_sub_block_n_iw = np.zeros((len(iws), num_target, num_target), dtype=np.complex128)
    else : 
        G_sub_block_n_iw = None

    # --- Method 1: Time Propagation (calculates G_ij element by element) ---
    if gfmeth == "time_prop":
        # We compute one full representative block from each identical group, element by element.
        for group in identical_groups:
            # Ensure indices are flattened from the start to avoid any nesting issues.
            local_indices_in_rep_block = np.array(blocks[group[0]]).flatten()
            
            print(f"  Computing representative block (local indices {list(local_indices_in_rep_block)}) via Time Propagation...")

            # This part is correct: Compute all G_ij for i,j in the representative block
            for i_local, i_idx_in_block in enumerate(local_indices_in_rep_block):
                for j_local, j_idx_in_block in enumerate(local_indices_in_rep_block):
                    global_i = target_indices[i_idx_in_block]
                    global_j = target_indices[j_idx_in_block]
                    
                    g_ij_n = gfs.green_function_from_time_propagation(
                        global_i, global_j, M, psi_n, e_n, ws, eta,
                        target_indices, NappH, h0_n, U_n,
                        one_bh_n, two_bh_n, coeff_thresh, L
                    )
                    G_sub_block_n[:, i_idx_in_block, j_idx_in_block] = g_ij_n
            
            
            # 1. Extract the data block we just computed, using robust slicing.
            rep_block_data = G_sub_block_n[:, local_indices_in_rep_block, :][:, :, local_indices_in_rep_block]

            # 2. Loop through the other equivalent blocks and copy the data element by element.
            for equiv_block_local_idx in group[1:]: # Skip the one we just computed
                local_indices_in_equiv_block = np.array(blocks[equiv_block_local_idx]).flatten()
                
                print(f"    -> Copying result to equivalent local indices: {list(local_indices_in_equiv_block)}")
                
                for i_source, i_dest in enumerate(local_indices_in_equiv_block):
                    for j_source, j_dest in enumerate(local_indices_in_equiv_block):
                        G_sub_block_n[:, i_dest, j_dest] = rep_block_data[:, i_source, j_source]

    # --- Method 2: Scalar Continued Fraction (DIAGONAL ONLY) ---
    elif gfmeth == "scalar_continued_fraction":
        if not is_diagonal:
            print("*" * 42)
            print(
                "WARNING: scalar_continued_fraction computes only diagonal Green's "
                "function elements, but h0 has non-diagonal blocks."
            )
            print("Diagonal elements inside one block are computed separately.")
            print("*" * 42)

        for group in identical_groups:
            rep_block_idx = group[0]
            rep_block = blocks[rep_block_idx]

            for pos_in_block, rep_local_idx in enumerate(rep_block):
                rep_global_idx = target_indices[rep_local_idx]

                print(
                    f"  Computing representative g_({rep_global_idx},{rep_global_idx}) "
                    "via Scalar CF..."
                )

                g_ii_n, g_ii_iw, _ = gfs.green_function_scalar_fixed_basis(
                    M, psi_n, e_n, ws, eta, rep_global_idx, NappH,
                    h0_n, U_n, one_bh_n, two_bh_n,
                    tables_n,
                    iws=iws,
                    coeff_thresh=coeff_thresh, L=L
                )

                # Copy only to the same relative position in equivalent blocks.
                for equiv_block_idx in group:
                    equiv_block = blocks[equiv_block_idx]

                    if len(equiv_block) != len(rep_block):
                        raise RuntimeError(
                            "Internal symmetry error: identical blocks have different sizes."
                        )

                    equiv_local_idx = equiv_block[pos_in_block]

                    G_sub_block_n[:, equiv_local_idx, equiv_local_idx] = g_ii_n

                    if iws is not None:
                        G_sub_block_n_iw[:, equiv_local_idx, equiv_local_idx] = g_ii_iw      

    # --- Method 3: Block Continued Fraction (most efficient for blocks) ---
    elif gfmeth == "block":
        for group in identical_groups:
            rep_block_local_idx = group[0]
            local_indices_in_rep_block = np.array(blocks[rep_block_local_idx]).flatten()
            global_indices_to_compute = [target_indices[i] for i in local_indices_in_rep_block]
            
            print(f"  Computing representative block (global indices {global_indices_to_compute}) via Block CF...")
            
            # This function now correctly returns a dense block, e.g., of shape (1001, 1, 1)
            G_dense_computed, G_dense_computed_iw,_ = gfs.green_function_block_lanczos_fixed_basis(
                M, psi_n, e_n, ws, eta, global_indices_to_compute, NappH,
                h0_n, U_n, one_bh_n, two_bh_n,
                tables_n, 
                iws, coeff_thresh=coeff_thresh, L=L
            )

            # Copy the computed block to all symmetric equivalents
            for equiv_block_local_idx in group:
                local_indices_in_equiv_block = np.array(blocks[equiv_block_local_idx]).flatten()
                
               
                print(f"    -> Copying result to equivalent local indices: {list(local_indices_in_equiv_block)}")

                for i_source, i_dest in enumerate(local_indices_in_equiv_block):
                    for j_source, j_dest in enumerate(local_indices_in_equiv_block):
                        G_sub_block_n[:, i_dest, j_dest] = G_dense_computed[:, i_source, j_source]
                        if iws is not None: 
                            G_sub_block_n_iw[:, i_dest, j_dest] = G_dense_computed_iw[:, i_source, j_source]


    return G_sub_block_n, G_sub_block_n_iw
