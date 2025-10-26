import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import List, Dict

def get_block_structure(h0: np.ndarray, tol: float = 1e-6) -> List[List[int]]:
    """
    Finds the block-diagonal structure of a Hamiltonian.

    This function treats the orbitals as nodes in a graph where an edge exists
    if the corresponding off-diagonal element in the Hamiltonian `h0` is
    non-zero (larger than `tol`). It then finds the connected components
    of this graph, which correspond to the decoupled blocks of the Hamiltonian.

    Args:
        h0 (np.ndarray): The one-body Hamiltonian matrix (M x M or 2M x 2M).
        tol (float): The tolerance for considering an off-diagonal element
                     to be non-zero.

    Returns:
        List[List[int]]: A list of blocks, where each block is a list of
                         the orbital indices that belong to it.
    """
    if h0.ndim != 2 or h0.shape[0] != h0.shape[1]:
        raise ValueError("h0 must be a square 2D array.")

    n_orb = h0.shape[0]
    
    # Create a boolean adjacency matrix: True where orbitals are coupled
    mask = np.abs(h0) > tol
    
    # The diagonal doesn't determine connections between different nodes
    np.fill_diagonal(mask, False)

    # Use scipy's graph tools to find connected components
    graph = csr_matrix(mask)
    n_components, labels = connected_components(graph, directed=False, connection='weak')

    # Group indices by their component label
    blocks = [[] for _ in range(n_components)]
    for i, label in enumerate(labels):
        blocks[label].append(i)
        
    return [sorted(block) for block in blocks]


def get_identical_blocks(blocks: List[List[int]], h0: np.ndarray, tol: float = 1e-6) -> List[List[int]]:
    """
    Finds groups of identical blocks within a block-diagonal Hamiltonian.

    For identifying symmetries, such as spin degeneracy if the
    spin-up and spin-down blocks are identical.

    Args:
        blocks (List[List[int]]): The block structure, as returned by
                                  `get_block_structure`.
        h0 (np.ndarray): The full one-body Hamiltonian matrix.
        tol (float): The tolerance for comparing submatrices of the blocks.

    Returns:
        List[List[int]]: A list of groups. Each group is a list of the
                         original indices (from the `blocks` list) of the
                         blocks that were found to be identical.
    """
    num_blocks = len(blocks)
    identical_groups = []
    visited_block_indices = set()

    for i in range(num_blocks):
        if i in visited_block_indices:
            continue

        current_group = [i]
        block_i = blocks[i]
        h_i = h0[np.ix_(block_i, block_i)]

        for j in range(i + 1, num_blocks):
            if j in visited_block_indices:
                continue
            
            block_j = blocks[j]

            # A necessary condition for being identical is having the same size
            if len(block_i) != len(block_j):
                continue
            
            h_j = h0[np.ix_(block_j, block_j)]

            # Compare the submatrices within the specified tolerance
            if np.allclose(h_i, h_j, atol=tol):
                current_group.append(j)
                visited_block_indices.add(j)
        
        identical_groups.append(current_group)
        visited_block_indices.add(i)
        
    return identical_groups


def is_diagonal(blocks: List[List[int]]) -> bool:
    """
    Checks if the Hamiltonian is purely diagonal based on its block structure.

    Args:
        blocks (List[List[int]]): The block structure from `get_block_structure`.

    Returns:
        bool: True if every block contains only a single orbital, False otherwise.
    """
    return all(len(block) == 1 for block in blocks)


def analyze_symmetries(h0: np.ndarray, tol: float = 1e-6, verbose=False) -> Dict:
    """
    A convenient wrapper function to perform a full symmetry analysis.

    Args:
        h0 (np.ndarray): The one-body Hamiltonian matrix.
        tol (float): The tolerance for all comparisons.

    Returns:
        Dict: A dictionary containing the analysis results:
              - 'blocks': The decoupled blocks of indices.
              - 'identical_groups': Groups of indices of identical blocks.
              - 'is_diagonal': A boolean indicating if h0 is diagonal.
    """
    blocks = get_block_structure(h0, tol)
    identical = get_identical_blocks(blocks, h0, tol)
    diag = is_diagonal(blocks)

    if verbose:
        print("\n--- Analyzed Structure ---")
        print(f"Blocks: {blocks}")
        print(f"Identical Groups (by block index): {identical}")
    
    return {
        "blocks": blocks,
        "identical_groups": identical,
        "is_diagonal": diag
    }


# ---------------------------------------------------------

def reconstruct_matrix_from_blocks(
    blocks: List[List[int]], 
    block_matrices: List[np.ndarray],
    dtype=np.complex128
) -> np.ndarray:
    """
    Reconstructs a full matrix from its block-diagonal components.

    This is the inverse operation of `get_block_structure`. It places each
    matrix from `block_matrices` onto the diagonal of a larger zero matrix
    at the locations specified by `blocks`.

    Args:
        blocks (List[List[int]]): The block structure, where each inner list
                                  contains the indices for that block.
        block_matrices (List[np.ndarray]): A list of square matrices, where
                                          `block_matrices[i]` corresponds
                                          to the block defined by `blocks[i]`.
        dtype: The numpy data type for the output matrix.

    Returns:
        np.ndarray: The assembled full matrix.
        
    Example:
        >>> blocks = [[0, 2], [1]]
        >>> block_mats = [np.array([[1,1],[1,1]]), np.array([[5]])]
        >>> reconstruct_matrix_from_blocks(blocks, block_mats)
        array([[1., 0., 1.],
               [0., 5., 0.],
               [1., 0., 1.]])
    """
    if len(blocks) != len(block_matrices):
        raise ValueError(
            f"Mismatch: {len(blocks)} blocks were provided, but "
            f"{len(block_matrices)} block matrices were given."
        )

    # Determine the size of the final matrix
    if not blocks:
        return np.array([[]], dtype=dtype)
    
    # Check for overlapping indices and validate matrix sizes
    all_indices = []
    for i, b_indices in enumerate(blocks):
        mat = block_matrices[i]
        if mat.shape != (len(b_indices), len(b_indices)):
            raise ValueError(
                f"Shape mismatch for block {i}: block indices have size "
                f"{len(b_indices)}, but provided matrix has shape {mat.shape}."
            )
        all_indices.extend(b_indices)
    
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Overlapping indices detected in the block structure.")

    max_index = max(all_indices) if all_indices else -1
    full_size = max_index + 1

    # Create the full matrix and fill it
    full_matrix = np.zeros((full_size, full_size), dtype=dtype)
    
    for block_indices, matrix in zip(blocks, block_matrices):
        # Use np.ix_ for elegant and efficient submatrix assignment
        full_matrix[np.ix_(block_indices, block_indices)] = matrix
        
    return full_matrix


def assemble_symmetric_hamiltonian(
    sym_dict,
    unique_matrices: Dict[int, np.ndarray],
    dtype=np.complex128
) -> np.ndarray:
    """
    A high-level wrapper to reconstruct a Hamiltonian using symmetry information.

    This function simplifies the creation of a full Hamiltonian when you know
    which blocks are identical and only need to define the matrix for one
    member of each identical group.

    Args:
        sym_dict : Dictionnary holding the block structure
        unique_matrices (Dict[int, np.ndarray]): A dictionary mapping the first index
                                                 of an identical group to its
                                                 corresponding matrix.
        dtype: The numpy data type for the output matrix.

    Returns:
        np.ndarray: The assembled full symmetric Hamiltonian.
    """

 
    blocks = sym_dict["blocks"]
    identical_groups = sym_dict["identical_groups"]

    num_blocks = len(blocks)
    full_block_matrices = [None] * num_blocks

    for group in identical_groups:
        leader_idx = group[0]
        if leader_idx not in unique_matrices:
            raise KeyError(f"Matrix for representative block index {leader_idx} was not provided.")
        
        matrix_for_group = unique_matrices[leader_idx]
        
        for member_idx in group:
            full_block_matrices[member_idx] = matrix_for_group
            
    if any(m is None for m in full_block_matrices):
        raise ValueError("Some blocks were not assigned a matrix. Check `identical_groups`.")

    return reconstruct_matrix_from_blocks(blocks, full_block_matrices, dtype=dtype)