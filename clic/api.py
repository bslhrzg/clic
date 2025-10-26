# clic/api.py
import numpy as np
from . import bath_transform, clic_clib as cc # Import for Wavefunction, SlaterDeterminant
from . import hamiltonians, basis_1p, basis_Np, ops, sci, mf, gfs, plotting
from . import results
from .config_models import SolverParameters, GfConfig # Use Pydantic for validated settings
from scipy.linalg import eigh,block_diag
import copy
from . import symmetries 
from tqdm import tqdm    # For the progress bar

class Model:
    """Represents the physical system via its Hamiltonian integrals."""
    def __init__(self, h0: np.ndarray, U: np.ndarray, M_spatial: int, Nelec: int):
        self.h0 = np.ascontiguousarray(h0, dtype=np.complex128)
        self.U = np.ascontiguousarray(U, dtype=np.complex128)
        self.M = M_spatial
        self.Nelec = Nelec
        # We can add u, mu etc. if basis_transforms needs them
        self.u = U[0, M_spatial, 0, M_spatial].real if U.ndim == 4 and U.shape[0] > M_spatial else 0.0

        self.is_impurity_model: bool = False
        self.imp_indices: list[int] = []
        self.Nelec_imp: int | None = None

class GroundStateSolver:
    """The main API endpoint for running a ground state calculation."""
    def __init__(self, model: Model, settings: dict | SolverParameters):
        self.model = model
        # If raw dict is passed, validate it with our Pydantic model
        if isinstance(settings, dict):
            self.settings = SolverParameters(**settings)
        else:
            self.settings = settings
        
        self.result: results.NelecLowEnergySubspace | None = None # Type hint for the result
        self.transformation_matrix = None # Store the transform here

    def _prepare_basis(self):
        method = self.settings.basis_prep_method
        print(f"Preparing one-particle basis using method: '{method}'")
        if method == "none":
            return
        
        elif method == 'rhf':
            hmf, es, Vs, rho = mf.mfscf_(self.model.h0,self.model.U,self.model.Nelec)
            h0,U_mat = basis_1p.basis_change_h0_U(self.model.h0,self.model.U,Vs)
            self.model.h0 = h0
            self.model.U = U_mat
            self.transformation_matrix = Vs

        elif method in ["dbl_chain","bath_no"]:

            if len(self.model.imp_indices) == 1:

                # self.model.h0 is the full 2M x 2M matrix for both spins
                # We operate on one spin sector (M x M)
                h0_spin = np.real(self.model.h0[:self.model.M, :self.model.M])
                Nelec_half = self.model.Nelec // 2 

                if method == "dbl_chain":
                    h_final_matrix, C_spin = bath_transform.get_double_chain_transform(
                        h0_spin, self.model.u, Nelec_half
                    )
                else :
                    h_final_matrix, C_spin = bath_transform.get_natural_orbital_transform(
                        h0_spin, self.model.u, Nelec_half
                    )


                h_final_matrix[0, 0] = -self.model.u / 2

                #print("Final Hamiltonian in double-chain basis:")
                #print(h_final_matrix)

                #print("Is unitary ? :")
                #print(np.linalg.norm(h_final_matrix - C_spin.T @ h0_spin @ C_spin))

                # Construct the full 2M x 2M Hamiltonian for both spins
                # This assumes basis_1p.double_h correctly creates the block diagonal matrix.
                h0_new = basis_1p.double_h(h_final_matrix, self.model.M)
                C = block_diag(C_spin, C_spin)

                self.model.h0 = h0_new
                self.transformation_matrix = C 

            # --- MULTI-ORBITAL IMPURITY CASE ---
            else: 
                print(f"Applying multi-orbital transformation for {len(self.model.imp_indices)} impurity orbitals...")

                # Create a dummy block dictionary for the spin channels
                block_dict = {'full_spin_block': list(range(self.model.M))}
                
                # Define impurity indices (now expecting global SPATIAL indices)
                imp_indices_spatial = self.model.imp_indices 
                
                # Create the Ne_per_block dictionary
                Ne_per_block = {'full_spin_block': self.model.Nelec // 2}
                
                # Choose which transformation to run based on the method
                if method == "dbl_chain":
                    print("Multi-orbital double chain not yet implemented. Stopping.")
                    # Or call the previous `perform_multi_orbital_no_transform` here if you fix it
                    return # Stop for now
                
                elif method == "bath_no":
                    # Call the new, simpler function
                    h_final, C_total = bath_transform.get_multi_orbital_natural_orbital_transform(
                        self.model.h0, 
                        self.model.U, 
                        self.model.Nelec, 
                        imp_indices_spatial, # Pass the SPATIAL indices
                    )
                
                # Transform the U tensor using the final transformation matrix
                _ , U_final = basis_1p.basis_change_h0_U(self.model.h0, self.model.U, C_total)
                
                # Update the model
                self.model.h0 = h_final
                self.model.U = U_final
                self.transformation_matrix = C_total
                
        else:
            raise NotImplementedError(f"Basis prep method '{method}' not implemented.")


    def _run_sci(self, seed: list[cc.SlaterDeterminant], max_iter_override: int | None = None) -> results.NelecLowEnergySubspace:
        """A helper to run the SCI calculation with a given starting seed."""
        ci_settings = self.settings.ci_method
        max_iter = max_iter_override if max_iter_override is not None else ci_settings.max_iter
        
        print(f"Running SCI with max_iter={max_iter} and seed size {len(seed)}...")
        
        result_obj = sci.selective_ci(
            h0=self.model.h0, 
            U=self.model.U,
            C=self.transformation_matrix,
            M=self.model.M, 
            Nelec=self.model.Nelec,
            seed=seed,  # Use the provided seed
            generator=sci.hamiltonian_generator, 
            selector=sci.cipsi_one_iter,
            max_iter=max_iter, 
            conv_tol=ci_settings.conv_tol,
            prune_thr=ci_settings.prune_thr,
            Nmul=ci_settings.Nmul, 
            verbose=True
        )
        return result_obj

    def solve(self) -> results.NelecLowEnergySubspace:
        """Runs the full workflow, including the optional NO step."""
        self._prepare_basis()

        ci_settings = self.settings.ci_method
        if ci_settings.type == "fci":
            # FCI logic doesn't use NOs, handle it separately and early.
            print("Running FCI calculation...")
            result_obj = sci.do_fci(
                h0=self.model.h0, U=self.model.U, M=self.model.M, Nelec=self.model.Nelec, Sz=0, verbose=True)

        elif ci_settings.type == "sci":
            # --- Determine the initial seed for the first (or only) SCI run ---
            if self.settings.basis_prep_method == 'rhf':
                initial_seed = basis_Np.get_rhf_determinant(self.model.Nelec, self.model.M)
            else:
                if self.model.is_impurity_model:
                    initial_seed = basis_Np.get_imp_starting_basis(
                        np.real(self.model.h0), self.model.Nelec, self.model.Nelec_imp, self.model.imp_indices)
                else: 
                    initial_seed = basis_Np.get_starting_basis(np.real(self.model.h0), self.model.Nelec)
            
            # --- "no0" Workflow ---
            if self.settings.use_no == 'no0':
                print("\n--- Starting 'no0' procedure: calculating Natural Orbitals ---")
                
                # 1. Run a preliminary SCI calculation
                pre_sci_result = self._run_sci(seed=initial_seed, max_iter_override=1)
                psi_approx = pre_sci_result.ground_state_wavefunction
                
                # 2. Compute 1-RDM and get correctly sorted NO transformation matrix
                print("Calculating 1-RDM from approximate wavefunction...")
                rdm1_spatial = self.get_one_rdm(wavefunction=psi_approx, spatial=True)
                occ_numbers, C_no_spin_unsorted = eigh(rdm1_spatial)
                sort_indices = np.argsort(occ_numbers)[::-1]
                C_no_spin = C_no_spin_unsorted[:, sort_indices]
                C_no = block_diag(C_no_spin, C_no_spin)
                
                # 3. Transform the Hamiltonian into the NO basis
                print("Transforming Hamiltonian to Natural Orbital basis...")
                h0_no, U_no = basis_1p.basis_change_h0_U(self.model.h0, self.model.U, C_no)
                self.model.h0 = h0_no
                self.model.U = U_no
                
                # 4. Compose the transformation matrices
                if self.transformation_matrix is not None:
                    self.transformation_matrix = self.transformation_matrix @ C_no
                else:
                    self.transformation_matrix = C_no
                
                print("--- 'no0' procedure finished. Starting final calculation. ---\n")
                
                # The seed for the final calculation is the basis from the pre-SCI run.
                # The determinants themselves don't change, they are just labels. We are now simply
                # re-interpreting them in the new NO basis where the Hamiltonian has changed.
                final_seed = pre_sci_result.basis

                # 6. Run the final SCI 
                result_obj = self._run_sci(seed=final_seed)

                #print("Calculating 1-RDM from approximate wavefunction...")
                #psi_approx = result_obj.ground_state_wavefunction
                #rdm1_spatial = self.get_one_rdm(wavefunction=psi_approx, spatial=True)
                #print("rdm1 = ")
                #print(rdm1_spatial)

                #print("diag el")
                #print(np.diag(rdm1_spatial))
              

            else: # --- Standard SCI Workflow (use_no == 'none') ---
                result_obj = self._run_sci(seed=initial_seed)

        # Finalize and store results
        result_obj.transformation_matrix = self.transformation_matrix
        self.result = result_obj
        
        print(f"\nCalculation finished. Ground state energy = {self.result.ground_state_energy:.12f}")
        return self.result

    def get_one_rdm(self, wavefunction: cc.Wavefunction | None = None, spatial: bool = False) -> np.ndarray:
        """Computes the 1-RDM. Can compute for a provided wavefunction."""
        if wavefunction is None:
            if not self.result:
                raise RuntimeError("Solver has not been run and no wavefunction was provided.")
            wavefunction = self.result.ground_state_wavefunction

        rdm = ops.one_rdm(wavefunction, self.model.M)
        
        if spatial:
            return rdm[:self.model.M, :self.model.M]
        return rdm
    


    def save_result(self, filename: str):
        if not self.result:
            raise RuntimeError("Solver has not been run yet.")
        self.result.save(filename)



class FockSpaceSolver:
    """
    API endpoint for finding the low-energy subspace across a range of particle
    numbers (Nelec). It orchestrates multiple fixed-Nelec calculations and
    combines them into a ThermalGroundState object for thermodynamic analysis.
    """
    def __init__(self, model: Model, settings: dict | SolverParameters, nelec_range: tuple[int, int]):
        self.base_model = model
        
        if isinstance(settings, dict):
            self.settings = SolverParameters(**settings)
        else:
            self.settings = settings
            
        self.nelec_range = range(nelec_range[0], nelec_range[1] + 1)
        self.result: results.ThermalGroundState | None = None

    def solve(self, initial_temperature: float = 300.0) -> results.ThermalGroundState:
        """
        Runs the full workflow for each Nelec in the specified range and
        returns a combined ThermalGroundState result object.

        Args:
            initial_temperature (float): The initial temperature in Kelvin to
                                         use for the resulting ThermalGroundState object.
        """
        all_results = {}
        
        print(f"\n--- Starting Fock Space Calculation for Nelec in {list(self.nelec_range)} ---")
        
        for nelec in self.nelec_range:
            print(f"\n--- Solving for Nelec = {nelec} ---")
            
            # Use a deepcopy to ensure basis preparations or other mutations
            # do not leak between calculations for different Nelec.
            current_model = copy.deepcopy(self.base_model)
            current_model.Nelec = nelec
            
            # Use the existing solver as the engine for this specific Nelec
            solver = GroundStateSolver(current_model, self.settings)
            
            # Run the calculation and store the result
            nelec_result = solver.solve()
            all_results[nelec] = nelec_result
        
        # Instantiate the final results object with all the computed subspaces
        self.result = results.ThermalGroundState(all_results, temperature=initial_temperature)
        
        # Report the absolute ground state found
        gs_nelec, gs_energy, _ = self.result.find_absolute_ground_state()
        print("\n--- Fock Space Calculation Finished ---")
        print(f"Absolute ground state found at Nelec = {gs_nelec} with E = {gs_energy:.12f}")
        
        return self.result

    def save_result(self, filename: str):
        if not self.result:
            raise RuntimeError("Solver has not been run yet.")
        self.result.save(filename)

# ----------------------------------------------------------------------------------

class GreenFunctionCalculator_:
    """The main API endpoint for calculating Green's functions."""
    def __init__(self, settings: dict | GfConfig):
        # Validate settings if a raw dict is passed
        if isinstance(settings, dict):
            self.settings = GfConfig(**settings)
        else:
            self.settings = settings
        
        # Attributes to be loaded from the ground state file
        self.M = None
        self.psi0_wf = None
        self.e0 = None
        self.h0 = None
        self.U = None

    def load_ground_state(self):
        """Loads data from the ground state .npz file."""
        filepath = self.settings.ground_state_file
        print(f"Loading ground state from '{filepath}'...")
        try:
            data = np.load(filepath, allow_pickle=True)
            self.M = int(data['M_spatial'])
            self.e0 = float(data['energy'])
            self.h0 = data['final_h0']
            self.U = data['final_U']
            
            # Reconstruct the wavefunction
            coeffs = data['wf_coeffs']
            alpha_list = data['basis_alpha_list']
            beta_list = data['basis_beta_list']
            basis = [cc.SlaterDeterminant(self.M, a, b) for a, b in zip(alpha_list, beta_list)]
            self.psi0_wf = cc.Wavefunction(self.M, basis, coeffs)
            print("Ground state loaded successfully.")


        except FileNotFoundError:
            raise RuntimeError(f"Ground state file not found at: {filepath}")
        except KeyError as e:
            raise RuntimeError(f"Missing key {e} in ground state file. The file may be invalid or incomplete.")

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """Runs the Green's function calculation and returns the results."""
        self.load_ground_state()

        p_gf = self.settings.green_function
        p_lanczos = self.settings.lanczos
        
        ws = np.linspace(p_gf.omega_mesh[0], p_gf.omega_mesh[1], int(p_gf.omega_mesh[2]))
        
        if p_gf.block_indices == "impurity":
            impurity_indices = [0, self.M]
        else:
            impurity_indices = p_gf.block_indices

        # Get the one- and two-body terms from the loaded Hamiltonian
        one_bh = ops.get_one_body_terms(self.h0, self.M)
        two_bh = ops.get_two_body_terms(self.U, self.M)

        print("\nStarting Green's function calculation...")
        G_block, meta = gfs.green_function_block_lanczos_fixed_basis(
            M=self.M, psi0_wf=self.psi0_wf, e0=self.e0, ws=ws, eta=p_gf.eta,
            impurity_indices=impurity_indices, NappH=p_lanczos.NappH,
            h0_clean=self.h0, U_clean=self.U,
            one_body_terms=one_bh, two_body_terms=two_bh,
            coeff_thresh=p_lanczos.coeff_thresh, L=p_lanczos.L
        )
        print("Calculation finished. Details:", meta)

        # Compute spectral function A(w) = -1/pi * Im[G(w)]
        A_w = -(1 / np.pi) * np.imag(G_block)
        
        # Save and plot if requested
        if self.settings.output.gf_data_file:
            np.savez_compressed(
                self.settings.output.gf_data_file,
                G_w=G_block, A_w=A_w, omega=ws
            )
            print(f"GF data saved to '{self.settings.output.gf_data_file}'")

        if self.settings.output.plot_file:
            plotting.plot_spectral_function(
                ws, A_w, impurity_indices,
                "Impurity Spectral Function",
                self.settings.output.plot_file
            )
        
        return ws, G_block, A_w
    


class GreenFunctionCalculator:
    """
    The main API endpoint for calculating Green's functions using a symmetry-aware,
    time-propagation-based Lanczos method.
    """
    def __init__(self, settings: dict | GfConfig):
        if isinstance(settings, dict):
            self.settings = GfConfig(**settings)
        else:
            self.settings = settings
        
        # Attributes to be loaded from the ground state file
        self.M: int | None = None
        self.psi0_wf: cc.Wavefunction | None = None
        self.e0: float | None = None
        self.h0: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.imp_indices: list[int] | None = None

    def load_ground_state(self):
        """Loads data from the ground state .npz file."""
        filepath = self.settings.ground_state_file
        print(f"Loading ground state from '{filepath}'...")
        try:
            data = np.load(filepath, allow_pickle=True)
            # Use .item() to extract scalar values safely
            self.M = int(data['M_spatial'].item())
            self.e0 = float(data['energy'].item())
            self.h0 = data['final_h0']
            self.U = data['final_U']
            
            # Check for impurity indices, default to empty list if not found
            self.imp_indices = list(data.get('imp_indices', []))

            # Reconstruct the wavefunction
            coeffs = data['wf_coeffs']
            basis = [cc.SlaterDeterminant(self.M, a, b) for a, b in zip(data['basis_alpha_list'], data['basis_beta_list'])]
            self.psi0_wf = cc.Wavefunction(self.M, basis, coeffs)
            print("Ground state loaded successfully.")
        except FileNotFoundError:
            raise RuntimeError(f"Ground state file not found at: {filepath}")
        except KeyError as e:
            raise RuntimeError(f"Missing key {e} in ground state file. Ensure it was saved correctly.")

    def _determine_work_list(self, target_indices: list[int], sym_dict: dict) -> list[tuple[int, int]]:
        """Determines the minimal set of (i, j) pairs to compute based on symmetry."""
        
        # 1. Create a map from any orbital to its unique representative
        orbital_to_rep = {}
        processed_blocks = set()
        for group in sym_dict['identical_groups']:
            leader_block_idx = group[0]
            leader_block_indices = sym_dict['blocks'][leader_block_idx]
            
            for member_block_idx in group:
                member_block_indices = sym_dict['blocks'][member_block_idx]
                for i in range(len(leader_block_indices)):
                    # Map orbital in member block to corresponding orbital in leader block
                    orbital_to_rep[member_block_indices[i]] = leader_block_indices[i]
                processed_blocks.add(member_block_idx)
        
        # 2. Build the set of unique representative pairs needed
        work_set = set()
        for i in target_indices:
            for j in target_indices:
                rep_i = orbital_to_rep[i]
                rep_j = orbital_to_rep[j]
                # Store in a canonical order to handle G_ij = G_ji
                work_set.add(tuple(sorted((rep_i, rep_j))))

        print(f"Symmetry analysis complete. Need to compute {len(work_set)} unique G_ij elements.")
        return sorted(list(work_set))

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs the Green's function calculation and returns the results."""
        self.load_ground_state()

        p_gf = self.settings.green_function
        p_lanczos = self.settings.lanczos
        
        ws = np.linspace(p_gf.omega_mesh[0], p_gf.omega_mesh[1], int(p_gf.omega_mesh[2]))
        
        # 1. Determine the target indices for the GF block
        if p_gf.block_indices == "impurity":
            if not self.imp_indices:
                raise ValueError("GF block set to 'impurity' but no impurity indices found in ground state file.")
            # Full spin-orbital impurity indices
            target_indices = sorted(self.imp_indices + [i + self.M for i in self.imp_indices])
        elif p_gf.block_indices == "full":
            target_indices = list(range(2 * self.M))
        else:
            target_indices = sorted(list(set(p_gf.block_indices)))

        print(f"\nTargeting Green's function block for indices: {target_indices}")
        
        # 2. Analyze symmetries and find the minimal work list
        sym_dict = symmetries.analyze_symmetries(self.h0)
        work_list = self._determine_work_list(target_indices, sym_dict)
        
        # Precompute Hamiltonian terms
        one_bh = ops.get_one_body_terms(self.h0, self.M)
        two_bh = ops.get_two_body_terms(self.U, self.M)

        # 3. Compute the unique G_ij elements with a progress bar
        computed_gfs = {}
        for i, j in tqdm(work_list, desc="Calculating G_ij(ω)"):
            g_ij = gfs.green_function_from_time_propagation(
                i, j, self.M, self.psi0_wf, self.e0, ws, p_gf.eta,
                target_indices, p_lanczos.NappH, self.h0, self.U,
                one_bh, two_bh, p_lanczos.coeff_thresh, p_lanczos.L
            )
            computed_gfs[(i, j)] = g_ij
            
        # 4. Reconstruct the full target block using symmetry
        print("\nReconstructing full GF block from unique elements...")
        num_target = len(target_indices)
        idx_map = {orb_idx: k for k, orb_idx in enumerate(target_indices)}
        G_block = np.zeros((len(ws), num_target, num_target), dtype=np.complex128)
        
        # Build the same orbital-to-representative map as before
        orbital_to_rep = {i: i for i in range(2 * self.M)}
        for group in sym_dict['identical_groups']:
            leader_block_indices = sym_dict['blocks'][group[0]]
            for member_block_idx in group[1:]:
                member_block_indices = sym_dict['blocks'][member_block_idx]
                for i in range(len(leader_block_indices)):
                    orbital_to_rep[member_block_indices[i]] = leader_block_indices[i]

        for k in range(num_target):
            for l in range(k, num_target):
                i, j = target_indices[k], target_indices[l]
                rep_i, rep_j = orbital_to_rep[i], orbital_to_rep[j]
                
                # Look up the computed value using the canonical key
                key = tuple(sorted((rep_i, rep_j)))
                G_block[:, k, l] = computed_gfs[key]
                
                # Enforce Hermiticity
                if k != l:
                    G_block[:, l, k] = np.conj(G_block[:, k, l])

        print("Calculation finished.")

        # 5. Compute spectral function, save, and plot
        A_w = -(1 / np.pi) * np.imag(G_block)
        
        if self.settings.output.gf_data_file:
            np.savez_compressed(
                self.settings.output.gf_data_file,
                G_w=G_block, A_w=A_w, omega=ws, indices=np.array(target_indices)
            )
            print(f"GF data saved to '{self.settings.output.gf_data_file}'")

        if self.settings.output.plot_file:
            # The indices for plotting are now just range(num_target) as we are plotting the sub-block
            plotting.plot_spectral_function(
                ws, A_w, list(range(num_target)),
                "Spectral Function",
                self.settings.output.plot_file
            )
        
        return ws, G_block, A_w