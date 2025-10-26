import numpy as np
import clic

# --- Step 1: Define the unique building blocks for our example ---
# A 2x2 matrix for a pair of symmetric blocks
block_A = np.array([
    [1.0, 0.5],
    [0.5, 1.0]
], dtype=np.complex128)

# A 1x1 matrix for a decoupled orbital
block_B = np.array([[2.0]], dtype=np.complex128)

# --- Step 2: Manually create a target symmetric Hamiltonian ---
# This is what we want our function to be able to generate.
# Structure: block_A, block_B, block_A
h0_target = np.zeros((5, 5), dtype=np.complex128)
h0_target[np.ix_([0, 1], [0, 1])] = block_A
h0_target[np.ix_([2], [2])] = block_B
h0_target[np.ix_([3, 4], [3, 4])] = block_A
print("--- Target Hamiltonian ---")
print(h0_target)

# --- Step 3: Use our analysis tools to get the structure ---
# In a real problem, you'd start with h0 and get this structure.
sym_info = clic.analyze_symmetries(h0_target)
blocks = sym_info['blocks']
identical_groups = sym_info['identical_groups']

print("\n--- Analyzed Structure ---")
print(f"Blocks: {blocks}")
print(f"Identical Groups (by block index): {identical_groups}")

# --- Step 4: Use the high-level assembler function ---
# We only need to provide the matrices for the "leader" of each identical group.
# From the output, group 0 is [0, 2], its leader is 0. Group 1 is [1], leader is 1.
unique_matrices = {
    0: block_A,  # Matrix for the group led by block index 0 ([0, 2])
    1: block_B,  # Matrix for the group led by block index 1 ([1])
}

h0_reconstructed = clic.assemble_symmetric_hamiltonian(
    sym_info,
    unique_matrices
)

print("\n--- Reconstructed with assemble_symmetric_hamiltonian ---")
print(h0_reconstructed)
print(f"Is reconstruction successful? {np.allclose(h0_target, h0_reconstructed)}")

# --- Step 5 (Bonus): Using the low-level function directly ---
# This shows how the wrapper works underneath. You have to provide all matrices
# in the correct order.
full_list_of_matrices = [block_A, block_B, block_A]
h0_reconstructed_low_level = clic.reconstruct_matrix_from_blocks(
    blocks, 
    full_list_of_matrices
)

print("\n--- Reconstructed with reconstruct_matrix_from_blocks ---")
print(h0_reconstructed_low_level)
print(f"Is reconstruction successful? {np.allclose(h0_target, h0_reconstructed_low_level)}")