# Developer's Guide: Extending the `clic` Package

This guide outlines the standard workflow for adding new features to the `clic` package. The project is designed with a clear separation between the core scientific logic (the API) and the user-facing configuration (the CLI), making it robust and extensible.

### The Golden Rule: API First, CLI Second

For any new feature, follow this two-step thought process:
1.  **API:** How would this feature work as a pure Python function or class method, operating on in-memory NumPy arrays and Python objects? This logic belongs in the core `clic` package.
2.  **CLI:** How do I expose this feature as a simple, user-friendly option in the `.toml` configuration files? This logic belongs in the CLI script and Pydantic validation models.

---

### The Development Cycle (4 Steps)

Adding a new feature almost always follows these four steps:

| Step                  | File(s) to Modify                                | Purpose                                                                 |
| --------------------- | ------------------------------------------------ | ----------------------------------------------------------------------- |
| **1. Implement Logic**| `clic/*.py` (e.g., `sci.py`, `basis_transforms.py`) | Write the new scientific algorithm (e.g., a new selector function).     |
| **2. Integrate API**  | `clic/api.py`                                    | Connect the new logic to the `GroundStateSolver` or `Model` class.      |
| **3. Expose Option**  | `clic/config_models.py`                          | Add the new user-facing option string to the Pydantic validation models.|
| **4. Connect CLI**    | `clic/scripts/cli.py`                            | Map the new TOML option to the corresponding API call.                  |

You only need to re-run `pip install -e .` if you add a new third-party dependency (e.g., `h5py`) to `pyproject.toml`. Changes to `.py` files are picked up automatically.

---

### Example Task 1: Adding a New Basis Transformation

**Goal:** Add a new method called `"rhf_no"` that runs RHF, then a small CI to compute and use Natural Orbitals.

#### Step 1: Implement Logic
In `clic/basis_transforms.py`, create the function that performs the calculation.
```python
# clic/basis_transforms.py
def rhf_natural_orbital_transform(h0, U, M, Nelec):
    # ... Your scientific code to:
    # 1. Run RHF
    # 2. Run a small CI (e.g., CISD)
    # 3. Compute the 1-RDM
    # 4. Diagonalize the 1-RDM to get Natural Orbitals
    # 5. Transform h0 and U to the NO basis
    return h0_new, U_new
```

#### Step 2: Integrate into API
In `clic/api.py`, add an elif block in GroundStateSolver._prepare_basis to call your new function.

```python
# clic/api.py
class GroundStateSolver:
    def _prepare_basis(self):
        method = self.settings.basis_prep_method
        # ...
        elif method == "rhf_no":
            print("...performing RHF -> Natural Orbital transformation.")
            h0_new, U_new = basis_transforms.rhf_natural_orbital_transform(
                self.model.h0, self.model.U, self.model.M, self.model.Nelec
            )
            self.model.h0, self.model.U = h0_new, U_new
```

#### Step 3: Expose User Option
In `clic/config_models.py`, add "rhf_no" to the list of allowed Literal strings. This enables automatic validation.

```Python
# clic/config_models.py
class SolverParameters(BaseModel):
    basis_prep_method: Literal["none", "rhf", "rhf_no", "dbl_chain"]
    ci_method: CiMethodConfig
```

#### Step 4: Connect CLI
No action needed! Because the cli.py script passes the SolverParameters object to the GroundStateSolver, the mapping is already complete. The _prepare_basis method will receive the validated string "rhf_no".

### Example Task 2: Adding a New Model Source (from HDF5)
**Goal**: Allow the model (integrals h0, U) to be read from an HDF5 file instead of generated parametrically.

#### Step 1: Implement Logic
The logic is simple file I/O. We will place it directly in the CLI script, as its primary role is to load data from files and prepare it for the API. You will need h5py.
Add "h5py" to the dependencies list in pyproject.toml.
Run pip install -e . once to install the new dependency.
#### Step 2: Integrate into API
No action needed! The clic.api.Model class is already perfect; it accepts h0 and U as NumPy arrays during initialization.
#### Step 3: Expose User Option
In clic/config_models.py, update ModelConfig to understand the new HDF5 source type.
```Python
# clic/config_models.py
class ModelConfig(BaseModel):
    model_name: str
    source_type: Literal["parametric", "hdf5"]  # Add "hdf5"
    parameters: ModelParameters | None = None      # Make optional
    hdf5_file: str | None = None                   # Add field for HDF5 path
```

#### Step 4: Connect CLI
In clic/scripts/cli.py, add logic to handle the hdf5 source type when creating the Model object.
```Python
# clic/scripts/cli.py
def main():
    # ... (after loading configs)
    
    # --- 2. Create the Model Object ---
    model = None
    if model_config.source_type == 'parametric':
        # ... (existing logic for parametric AIM) ...
    
    elif model_config.source_type == 'hdf5':
        import h5py
        print(f"Loading model from HDF5 file: {model_config.hdf5_file}")
        with h5py.File(model_config.hdf5_file, 'r') as f:
            h0 = f['h0'][:]
            U = f['U'][:]
            M_spatial = f.attrs['M_spatial']
            Nelec = f.attrs['Nelec']
        model = Model(h0=h0, U=U, M_spatial=M_spatial, Nelec=Nelec)

    if model is None:
        # ... (handle error) ...
    
    # --- 3. Run the Solver (this part is unchanged) ---
    solver = GroundStateSolver(model, solver_config.solver)
    # ...
```

By following this pattern, you can systematically add more generators, selectors, models, and methods while keeping the codebase clean, robust, and easy to maintain.