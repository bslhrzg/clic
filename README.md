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


## Solving a model with a manually specified bath

`from clic import aimsolve` runs the many-body solver and computes impurity
Green functions without fitting a hybridization or calculating self-energy.
See [examples/aimsolve_model.py](examples/aimsolve_model.py) for a complete
one-shot model script, including a simple Hubbard interaction tensor. Run it
in your calculation directory with `python /path/to/clic/examples/aimsolve_model.py`
(optionally followed by a new HDF5 output filename).

The bath dictionary has two entries:

- `energies`: a real vector of bath spin-orbital energies, ordered as all up
  followed by all down; provide paired up/down slots, which need not be equal.
- `V`: a complex array of shape `(N_imp_spin_orbitals, N_bath_spin_orbitals)`.
  `V[i,b]` is the coefficient of `d_i† a_b`; its conjugate supplies the reverse
  hopping. Rows use the same AlphaFirst ordering as `h_imp` and `U_imp`.
  A column can couple to several impurity orbitals.

`bath=None` gives the isolated impurity. Bath size is inferred from these arrays;
`nb`, hybridization fitting/windowing, frozen-bath files, and `spin_avg_sigma`
do not modify the model. The impurity interaction tensor uses the same convention
as `dmft_step()`. Energies, frequency meshes and `eim` are in Ry, temperature in K;
all one-body energies must already be relative to a common chemical potential.

Settings follow the existing defaults/RSPT/`input.toml` precedence, with the new
`solver_params` dictionary taking final precedence over those sources. Set
`input_path=None` to ignore the TOML file. Explicit model/mesh arguments and
`eim` override the corresponding settings. For small models, start with
`ci_type="fci"` and `basis_prep_method="none"`. `nelec_range` can be a fixed total
electron number, a list of sectors, or `"auto"` as in the DMFT solver. Finite
temperature averages use the computed roots/sectors and existing DMFT pruning
threshold; they are not necessarily a complete thermal trace. Green-function
accuracy also depends on `NappH`, `L_lanczos`, and truncation settings.
The `rhf` basis preparation is rejected in this entry point because it mixes
impurity and bath orbitals without mapping Green-function targets back to the
input impurity basis.

The result dictionary contains `ws`, `iws`, `G_imp`, `G_imp_iw`, `A_imp`, and
`thermal_avgs`. Green matrices have frequency first and are in the input impurity
basis; `A_imp` is the diagonal spectral density. Optional `iws` contains real
Matsubara frequencies, not imaginary numbers; omitting it returns empty
Matsubara arrays. All matrix elements are computed by default, without inferring
bath symmetries from `h_imp` alone; `green_diag_only=True` explicitly requests
only the diagonal.

An optional `archive_path` creates a **new** HDF5 file with `model/`, `settings/`,
`data/`, and `thermal_avgs/` groups. It refuses to replace existing files.
The existing solver density-matrix and Green-function text outputs are retained;
`plot_sf=True` also writes the spectral plot. Run separate calculations in
separate working directories to keep these outputs distinct.
