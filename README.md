# clic


## Installation and Build

1.  **Requirements**:
    - to do 

2.  **Build**:
    
    on Mac with llvm : 

    ```bash
    CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ pip install -e .
    ```

## Quick Start: 


```python
import numpy as np
from scipy.sparse.linalg import eigsh
import clic as qc

# --- System Definition ---
M = 2  # Spatial orbitals
Nelec = 2
t = 1.0
U = 4.0

# --- 1. Generate the FCI Basis ---
basis = [
    qc.SlaterDeterminant(M, [], [0,1]),     
    qc.SlaterDeterminant(M, [0], [0]),   
    qc.SlaterDeterminant(M, [0], [1]), 
    qc.SlaterDeterminant(M, [1], [0]),    
    qc.SlaterDeterminant(M, [1], [1]),   
    qc.SlaterDeterminant(M, [0,1], [])
]
basis.sort()

# --- 2. Define Integrals (AlphaFirst basis) ---
K = 2 * M
h0 = np.zeros((K, K), dtype=np.complex128)
h0[0, 1] = h0[1, 0] = -t  
h0[2, 3] = h0[3, 2] = -t 

# expects V[p,q,r,s] = <pq|V|rs>
U_mat = np.zeros((K, K, K, K), dtype=np.complex128)
U_mat[0, 2, 0, 2] = U  
U_mat[1, 3, 1, 3] = U  

# --- 3. Build and Diagonalize Hamiltonian ---
H_mat = qc.build_hamiltonian_openmp(basis, h0, U_mat)
eigvals, _ = eigsh(H_mat, k=1, which='SA')

print(f"Ground State Energy: {eigvals[0]:.8f}")

