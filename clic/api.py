# clic/api.py
import numpy as np
from . import bath_transform, clic_clib as cc # Import for Wavefunction, SlaterDeterminant
from . import hamiltonians, basis_1p, basis_Np, ops, sci, mf, gfs, plotting
from . import results
from .config_models import SolverParameters, GfConfig # Use Pydantic for validated settings
from typing import Literal, Union, List, Optional 

from scipy.linalg import eigh,block_diag
import copy
from . import symmetries, io_utils
from tqdm import tqdm    # For the progress bar
import h5py 

class Model:
    """Represents the physical system via its Hamiltonian integrals."""
    def __init__(self, h0: np.ndarray, U: np.ndarray, M_spatial: int, Nelec: int):
        self.h0 = np.ascontiguousarray(h0, dtype=np.complex128)

        print("diag h0 = ")
        for i in range(np.shape(h0)[0]):
            print(f"{i} : {np.real(h0[i,i]):.3e}")

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
        

        self.result: results.ThermalGroundState | None = None 
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
              
                C = block_diag(C_spin, C_spin)
                self.model.h0 = C.conj().T @ self.model.h0 @ C
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
        num_roots = 1 if max_iter_override is not None else ci_settings.num_roots
        print(f"Running SCI with max_iter={max_iter}, num_roots={num_roots} and seed size {len(seed)}...")
        
        result_obj = sci.selective_ci(
            h0=self.model.h0, 
            U=self.model.U,
            C=self.transformation_matrix,
            M=self.model.M, 
            Nelec=self.model.Nelec,
            seed=seed,  # Use the provided seed
            generator=sci.hamiltonian_generator, 
            selector=sci.cipsi_one_iter,
            num_roots=num_roots,
            max_iter=max_iter, 
            conv_tol=ci_settings.conv_tol,
            prune_thr=ci_settings.prune_thr,
            Nmul=ci_settings.Nmul, 
            verbose=True
        )
        return result_obj

    def solve(self) -> results.ThermalGroundState:
        """Runs the full workflow, including the optional NO step."""

        # Create a deep copy of the original model BEFORE any transformations are applied
        original_model = copy.deepcopy(self.model)

        # Edge case, Nelec = 0 
        if self.model.Nelec == 0 :
            vacuum_det = cc.SlaterDeterminant(self.model.M, [], [])
            psis = [cc.Wavefunction(self.model.M, [vacuum_det], [0+0j])]
            self.result = results.NelecLowEnergySubspace(M=self.model.M,Nelec=0,
                energies=[0],
                wavefunctions=psis,
                basis=[vacuum_det],
                transformation_matrix=None
            )
            # Even for this simple case, we create the full ThermalGroundState object
            thermal_result = results.ThermalGroundState(
                results_by_nelec={0: self.result},
                base_model=original_model,
                temperature=self.settings.initial_temperature
            )
            self.result = thermal_result
            return self.result


        self._prepare_basis()

        ci_settings = self.settings.ci_method
        if ci_settings.type == "fci":
            # FCI logic doesn't use NOs, handle it separately and early.
            print("Running FCI calculation...")
            result_obj = sci.do_fci(
                h0=self.model.h0, U=self.model.U, M=self.model.M, Nelec=self.model.Nelec, num_roots=ci_settings.num_roots,Sz=0, verbose=True)

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

            else: # --- Standard SCI Workflow (use_no == 'none') ---
                result_obj = self._run_sci(seed=initial_seed)

        # Finalize and store results
        result_obj.transformation_matrix = self.transformation_matrix
        
        print(f"\nCalculation finished. Ground state energy = {result_obj.ground_state_energy:.12f}")

        # Instantiate the final results object with the computed subspace and original model
        final_thermal_state = results.ThermalGroundState(
            results_by_nelec={result_obj.Nelec: result_obj},
            base_model=original_model, # Pass the whole original model
            temperature=self.settings.initial_temperature
        )

        self.result = final_thermal_state
        return self.result


    def get_one_rdm(self, wavefunction: cc.Wavefunction | None = None, spatial: bool = False) -> np.ndarray:
        """Computes the 1-RDM. Can compute for a provided wavefunction."""
        if wavefunction is None:
            if not self.result:
                raise RuntimeError("Solver has not been run and no wavefunction was provided.")
            # We need the ground state wavefunction from the correct Nelec sector
            _, _, wavefunction = self.result.find_absolute_ground_state()

        rdm = ops.one_rdm(wavefunction, self.model.M)
        
        if spatial:
            return rdm[:self.model.M, :self.model.M]
        return rdm
    
    def compute_stats(self, wavefunction: cc.Wavefunction):
        if self.model.is_impurity_model:
            imp_indices_spinfull = self.model.imp_indices + [iimp + self.model.M for iimp in self.model.imp_indices]
            gamma = ops.one_rdm(wavefunction,self.model.M,block=imp_indices_spinfull)
            occs = np.diag(gamma)
            print("* Occupations -------------")
            print(f"total: {np.sum(np.real(occs)):.3e}")
            for (i,iimp) in enumerate(imp_indices_spinfull):
                print(f"{i:>6} : {np.real(occs[i]):.3e}")
        else:
            gamma = ops.one_rdm(wavefunction,self.model.M)
            occs = np.diag(gamma)
            print(f"Total occupation: {np.sum(occs)}")
            for i in range(len(occs)):
                print(f"occ[{i}] = {np.real(occs[i]):.3e}")


    def save_result(self, filename: str):
        """Saves the ThermalGroundState result to a single HDF5 file."""
        if not self.result:
            raise RuntimeError("Solver has not been run yet.")
        self.result.save(filename)
        
class FockSpaceSolver:
    """
    API endpoint for finding the low-energy subspace across a range of particle
    numbers (Nelec). It orchestrates multiple fixed-Nelec calculations and
    combines them into a ThermalGroundState object for thermodynamic analysis.
    """
    def __init__(self, model: Model, settings: dict | SolverParameters, nelec_range: Union[tuple[int, int], Literal["auto"]]):
        self.base_model = model
        
        if isinstance(settings, dict):
            self.settings = SolverParameters(**settings)
        else:
            self.settings = settings
            
        self.nelec_setting = nelec_range
        self.result: results.ThermalGroundState | None = None

    def _solve_single_nelec(self, nelec: int, cache: dict) -> results.ThermalGroundState:
        """
        Helper to run a calculation for a single Nelec, using a cache.
        Returns a ThermalGroundState object, which contains the NelecLowEnergySubspace.
        """
        if nelec in cache:
            return cache[nelec]

        print(f"\n--- Solving for Nelec = {nelec} ---")
        current_model = copy.deepcopy(self.base_model)
        current_model.Nelec = nelec
        
        solver = GroundStateSolver(current_model, self.settings)
        # solver.solve() returns a ThermalGroundState object
        nelec_result_thermal = solver.solve() 
        
        cache[nelec] = nelec_result_thermal
        return nelec_result_thermal

    def _find_optimal_nelec(self) -> dict[int, results.NelecLowEnergySubspace]:
        """
        Iteratively searches for the Nelec that minimizes the ground state energy.
        Returns a dictionary of NelecLowEnergySubspace results centered around the minimum.
        """
        if not self.base_model.is_impurity_model:
            raise ValueError("Automatic Nelec search is only supported for impurity models.")

        nelec_start = self.base_model.Nelec
        print(f"\n--- Starting Automatic Search for Optimal Nelec (start = {nelec_start}) ---")

        energies = {}
        results_cache = {} # Caches ThermalGroundState objects
        subspace_cache = {} # Caches NelecLowEnergySubspace objects

        # --- Initial Point ---
        print(f"Calculating for starting Nelec = {nelec_start}...")
        result_thermal = self._solve_single_nelec(nelec_start, results_cache)
        _, e0, _ = result_thermal.find_absolute_ground_state()
        energies[nelec_start] = e0
        subspace_cache[nelec_start] = result_thermal.results_by_nelec[nelec_start]
        
        # --- Search Upwards ---
        print("\n--- Searching for minimum in increasing Nelec direction ---")
        nelec_curr = nelec_start
        while True:
            nelec_next = nelec_curr + 1
            if nelec_next > 2 * self.base_model.M:
                print("Reached maximum possible electrons. Stopping upward search.")
                break

            e_curr = energies[nelec_curr]
            result_thermal_next = self._solve_single_nelec(nelec_next, results_cache)
            _, e_next, _ = result_thermal_next.find_absolute_ground_state()
            energies[nelec_next] = e_next
            subspace_cache[nelec_next] = result_thermal_next.results_by_nelec[nelec_next]
            
            print(f"E({nelec_curr}) = {e_curr:.6f}, E({nelec_next}) = {e_next:.6f}")
            if e_next >= e_curr:
                print("Energy is no longer decreasing. Stopping upward search.")
                break
            nelec_curr = nelec_next

        # --- Search Downwards ---
        print("\n--- Searching for minimum in decreasing Nelec direction ---")
        nelec_curr = nelec_start
        while True:
            nelec_next = nelec_curr - 1
            if nelec_next < 0:
                print("Reached zero electrons. Stopping downward search.")
                break

            e_curr = energies[nelec_curr]
            result_thermal_next = self._solve_single_nelec(nelec_next, results_cache)
            _, e_next, _ = result_thermal_next.find_absolute_ground_state()
            energies[nelec_next] = e_next
            subspace_cache[nelec_next] = result_thermal_next.results_by_nelec[nelec_next]
            
            print(f"E({nelec_curr}) = {e_curr:.6f}, E({nelec_next}) = {e_next:.6f}")
            if e_next >= e_curr:
                print("Energy is no longer decreasing. Stopping downward search.")
                break
            nelec_curr = nelec_next
            
        # --- Find minimum and collect results ---
        nelec_min = min(energies, key=energies.get)
        
        print(f"\n--- Minimum energy found at Nelec = {nelec_min} (E = {energies[nelec_min]:.8f}) ---")
        print("Collecting results for thermal state around the minimum.")

        final_subspaces = {}
        for nelec_final in [nelec_min - 1, nelec_min, nelec_min + 1]:
            if 0 <= nelec_final <= 2 * self.base_model.M:
                if nelec_final in subspace_cache:
                    final_subspaces[nelec_final] = subspace_cache[nelec_final]
                else:
                    # This case should be rare but is included for robustness
                    result_thermal = self._solve_single_nelec(nelec_final, results_cache)
                    final_subspaces[nelec_final] = result_thermal.results_by_nelec[nelec_final]
        
        return final_subspaces


    def solve(self, initial_temperature: float = 300.0) -> results.ThermalGroundState:
        """
        Runs the full workflow for each Nelec and returns a combined result.
        If nelec_setting is 'auto', it finds the optimal Nelec first.
        """
        all_subspaces = {}

        if self.nelec_setting == "auto":
            all_subspaces = self._find_optimal_nelec()
        else:
            # Original behavior for a fixed range
            nelec_range = range(self.nelec_setting[0], self.nelec_setting[1] + 1)
            print(f"\n--- Starting Fock Space Calculation for Nelec in {list(nelec_range)} ---")
            results_cache = {} # Caches ThermalGroundState objects
            for nelec in nelec_range:
                # _solve_single_nelec returns a full ThermalGroundState object
                result_thermal = self._solve_single_nelec(nelec, results_cache)
                # We only need the NelecLowEnergySubspace from it
                all_subspaces[nelec] = result_thermal.results_by_nelec[nelec]
        
        # Instantiate the final results object with all computed subspaces and the base model
        self.result = results.ThermalGroundState(
            results_by_nelec=all_subspaces,
            base_model=self.base_model, # Pass the entire base model
            temperature=initial_temperature
        )
        
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

class GreenFunctionCalculator:
    """
    Calculates the thermally-averaged Green's function from a saved ThermalGroundState.
    """
    def __init__(self, settings: dict | GfConfig):
        if isinstance(settings, dict):
            self.settings = GfConfig(**settings)
        else:
            self.settings = settings
        
        self.thermal_state: results.ThermalGroundState | None = None

    def load_thermal_state(self):
        """Loads the ThermalGroundState from HDF5 and prepares it for calculation."""
        filepath = self.settings.ground_state_file
        print(f"Loading thermal state from HDF5 file '{filepath}'...")
        try:
            self.thermal_state = results.ThermalGroundState.load(filepath)
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"Failed to load ground state file: {e}")
        
        print(f"Thermal state loaded. Initial temperature: {self.thermal_state.temperature:.1f} K.")
        print(f"Prepared thermal state with {len(self.thermal_state._all_states)} states for GF calculation.")

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the thermally-averaged Green's function calculation.
        """
        self.load_thermal_state()

        p_gf = self.settings.green_function
        p_lanczos = self.settings.lanczos
        ws = np.linspace(p_gf.omega_mesh[0], p_gf.omega_mesh[1], int(p_gf.omega_mesh[2]))

        if p_gf.block_indices == "impurity":
            if not self.thermal_state.is_impurity_model:
                raise ValueError("GF block set to 'impurity' but loaded state is not an impurity model.")
            target_indices = sorted(self.thermal_state.imp_indices + [i + self.thermal_state.M for i in self.thermal_state.imp_indices])
        else:
            target_indices = sorted(list(set(p_gf.block_indices)))

        print(f"\nTargeting Green's function block for indices: {target_indices}")
        
        num_target = len(target_indices)
        # We will calculate the diagonal elements for now, as in your previous code.
        # This can be extended to the full block later if needed.
        G_total_diag = np.zeros((len(ws), num_target), dtype=np.complex128)
        
        # Cache for transformed Hamiltonians to avoid redundant calculations
        hamiltonian_cache = {}
        base_h0 = self.thermal_state.base_h0
        base_U = self.thermal_state.base_U
        M = self.thermal_state.M

        # Main loop over all states in the thermal ensemble
        iterator = zip(self.thermal_state._all_states, self.thermal_state.boltzmann_weights)
        for (state_info, weight) in tqdm(iterator, total=len(self.thermal_state._all_states), desc="Processing thermal states"):
            e_n, nelec_n, psi_n = state_info

            # Get the correct transformed Hamiltonian for this state's Nelec sector
            if nelec_n not in hamiltonian_cache:
                subspace = self.thermal_state.results_by_nelec[nelec_n]
                C = subspace.transformation_matrix
                if C is None:
                    h0_n, U_n = base_h0, base_U
                else:
                    h0_n, U_n = basis_1p.basis_change_h0_U(base_h0, base_U, C)
                
                one_bh_n = ops.get_one_body_terms(h0_n, M)
                two_bh_n = ops.get_two_body_terms(U_n, M)
                hamiltonian_cache[nelec_n] = (h0_n, U_n, one_bh_n, two_bh_n)
            
            h0_n, U_n, one_bh_n, two_bh_n = hamiltonian_cache[nelec_n]
            
            # Calculate the GF contribution from this single state
            for i, orb_idx in enumerate(target_indices):
                g_ii_n = gfs.green_function_from_time_propagation(
                    orb_idx, orb_idx, M, psi_n, e_n, ws, p_gf.eta,
                    target_indices, p_lanczos.NappH, h0_n, U_n,
                    one_bh_n, two_bh_n, p_lanczos.coeff_thresh, p_lanczos.L
                )
                G_total_diag[:, i] += weight * g_ii_n

        print("\nThermally-averaged calculation finished.")
        A_w_total = -(1 / np.pi) * np.imag(G_total_diag)
        
        # Save and plot the final, thermally-averaged results
        #if self.settings.output.gf_diag_txt_file:
        dodump=True 
        if dodump:
            io_utils.dump(
                A_w_total,
                ws,
                'A_w_thermal',
            )

        doplot=True
        #if self.settings.output.plot_file:
        if doplot:
            plotting.plot_spectral_function(
                ws, A_w_total, list(range(num_target)),
                f"Thermally-Averaged Spectral Function (T={self.thermal_state.temperature}K)",
                self.settings.output.plot_file
            )
        
        return ws, G_total_diag, A_w_total
    
