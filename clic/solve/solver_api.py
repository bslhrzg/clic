# clic/solve/solver_api.py
import numpy as np
from scipy.linalg import eigh, block_diag
import copy

# Internal imports
import clic_clib as cc 
from clic.model import bath_transform, double_chains, hamiltonians
from clic.basis import basis_1p, basis_Np
from clic.ops import ops
from clic.solve import sci,fci
from clic.mf import mf
from clic.results import results
from clic.symmetries import symmetries

class GroundStateSolver:
    """
    Solves the ground state for a specific Model and a FIXED particle number (Nelec).
    
    Args:
        Nelec (int): The total number of electrons to solve for.
        Nelec_imp (int, optional): The number of electrons on the impurity. 
                                   Used ONLY to generate a better starting seed for the solver.
    """
    def __init__(self, model, settings, Nelec: int, Nelec_imp: int = None):
        self.model = model
        self.settings = settings
        self.Nelec = int(Nelec)
        self.Nelec_imp = Nelec_imp
        
        self.result = None 
        self.transformation_matrix = None 

        print(f"model.is_impurity_model = {model.is_impurity_model}")

    def _prepare_basis(self):
        method = self.settings.basis_prep_method
        print(f"Preparing one-particle basis using method: '{method}' for Nelec={self.Nelec}")
        
        if method == "none":
            return
        
        # --- Restricted Hartree Fock ---
        elif method == 'rhf':
            hmf, es, Vs, rho = mf.mfscf(self.model.h0, self.model.U, self.Nelec, spinsym_only=True)
            h0, U_mat = basis_1p.basis_change_h0_U(self.model.h0, self.model.U, Vs)
            
            self.model.h0 = h0
            self.model.U = U_mat
            self.transformation_matrix = Vs

        # --- Impurity Bath Transforms ---
        elif method in ["dbl_chain", "bath_no"]:
            print(f"Applying multi-orbital transformation for {len(self.model.imp_indices_spatial)} impurity orbitals...")
            
            imp_indices = self.model.imp_indices_spatial
            
            if method == "dbl_chain":

                hmf, _, _, rho_mf = mf.mfscf(self.model.h0, self.model.U, self.Nelec)

                hmf_ab = basis_1p.transform_h0_alphafirst_to_interleaved(hmf)
                rhomf_ab = basis_1p.transform_h0_alphafirst_to_interleaved(rho_mf)

                Nimp = len(imp_indices) * 2
                hdc_ab, C_ab, meta = double_chains.double_chain_by_blocks(
                    hmf_ab, rhomf_ab, Nimp, self.Nelec,
                    symmetries.analyze_symmetries, double_chains.get_double_chain_transform_multi
                )

                hdc = basis_1p.transform_integrals_interleaved_to_alphafirst(hdc_ab)
                C = basis_1p.transform_integrals_interleaved_to_alphafirst(C_ab)
                
                self.model.h0 = C.conj().T @ self.model.h0 @ C 
                self.transformation_matrix = C
            
            elif method == "bath_no":
                h_final, C_total = bath_transform.get_multi_orbital_natural_orbital_transform(
                    self.model.h0, 
                    self.model.U, 
                    self.Nelec, 
                    imp_indices
                )
                _, U_final = basis_1p.basis_change_h0_U(self.model.h0, self.model.U, C_total)
                
                self.model.h0 = h_final
                self.model.U = U_final
                self.transformation_matrix = C_total
        else:
            raise NotImplementedError(f"Basis prep method '{method}' not implemented.")

    def _run_sci(self, seed: list[cc.SlaterDeterminant], max_iter_override: int | None = None):
        ci_settings = self.settings.ci_method
        max_iter = max_iter_override if max_iter_override is not None else ci_settings.max_iter
        num_roots = 1 if max_iter_override is not None else ci_settings.num_roots
        
        result_obj = sci.selective_ci(
            h0=self.model.h0, 
            U=self.model.U,
            C=self.transformation_matrix,
            M=self.model.M_spatial, 
            Nelec=self.Nelec,
            seed=seed,
            generator=sci.hamiltonian_generator, 
            selector=sci.cipsi_select,
            num_roots=num_roots,
            max_iter=max_iter, 
            conv_tol=ci_settings.conv_tol,
            prune_thr=ci_settings.prune_thr,
            Nmul=ci_settings.Nmul, 
            verbose=True
        )
        return result_obj

    def solve(self) -> results.ThermalGroundState:
        original_h0 = self.model.h0
        original_U = self.model.U
        
        # Work on a local copy to avoid side effects
        self.model = copy.copy(self.model) 
        self.model.h0 = np.copy(original_h0)
        self.model.U = np.copy(original_U)

        if self.Nelec == 0:
            vacuum_det = cc.SlaterDeterminant(self.model.M_spatial, [], [])
            psis = [cc.Wavefunction(self.model.M_spatial, [vacuum_det], [0+0j])]
            self.result = results.NelecLowEnergySubspace(M_spatial=self.model.M_spatial, Nelec=0,
                energies=[0], wavefunctions=psis, basis=[vacuum_det], transformation_matrix=None)
            
            return results.ThermalGroundState(
                results_by_nelec={0: self.result},
                base_model=self.model, 
                temperature=self.settings.temperature
            )

        self._prepare_basis()

        ci_settings = self.settings.ci_method
        if ci_settings.type == "fci":
            result_obj = fci.do_fci(
                h0=self.model.h0, U=self.model.U, M=self.model.M_spatial, 
                Nelec=self.Nelec, num_roots=ci_settings.num_roots, Sz=None, verbose=True
            )

        elif ci_settings.type == "sci":
            # --- Seed Generation ---
            if self.settings.basis_prep_method == 'rhf':
                initial_seed = basis_Np.get_rhf_determinant(self.Nelec, self.model.M_spatial)
            else:
                
                M_imp = len(self.model.imp_indices_spatial)

                # Use Nelec_imp if provided to generate a better seed
                if self.model.is_impurity_model and self.Nelec_imp is not None \
                and self.model.M_spatial > M_imp: # Not HIA right ? Else regular starting basis

                        initial_seed = basis_Np.get_imp_starting_basis(
                            np.real(self.model.h0), 
                            self.Nelec, 
                            self.Nelec_imp, 
                            self.model.imp_indices_spatial
                        )
                else: 
                    initial_seed = basis_Np.get_starting_basis(np.real(self.model.h0), self.Nelec)

            # --- Run SCI ---
            if self.settings.use_no == 'no0':
                pre_sci_result = self._run_sci(seed=initial_seed, max_iter_override=1)
                psi_approx = pre_sci_result.ground_state_wavefunction
                
                rdm1_spatial = self.get_one_rdm(wavefunction=psi_approx, spatial=True)
                occ_numbers, C_no_spin_unsorted = eigh(rdm1_spatial)
                sort_indices = np.argsort(occ_numbers)[::-1]
                C_no_spin = C_no_spin_unsorted[:, sort_indices]
                C_no = block_diag(C_no_spin, C_no_spin)
                
                h0_no, U_no = basis_1p.basis_change_h0_U(self.model.h0, self.model.U, C_no)
                self.model.h0 = h0_no
                self.model.U = U_no
                
                if self.transformation_matrix is not None:
                    self.transformation_matrix = self.transformation_matrix @ C_no
                else:
                    self.transformation_matrix = C_no
                
                final_seed = pre_sci_result.basis
                result_obj = self._run_sci(seed=final_seed)
            else:
                result_obj = self._run_sci(seed=initial_seed)

        result_obj.transformation_matrix = self.transformation_matrix
        
        clean_model = copy.copy(self.model)
        clean_model.h0 = original_h0
        clean_model.U = original_U
        print(f"clean_model.is_impurity_model = {clean_model.is_impurity_model}")

        self.result = results.ThermalGroundState(
            results_by_nelec={self.Nelec: result_obj},
            base_model=clean_model, 
            temperature=self.settings.temperature
        )
        return self.result

    def get_one_rdm(self, wavefunction=None, spatial=False):
        if wavefunction is None:
            if not self.result:
                raise RuntimeError("Solver has not been run")
            _, _, wavefunction = self.result.find_absolute_ground_state()
        
        rdm = ops.one_rdm(wavefunction, self.model.M_spatial)
        if spatial:
            return rdm[:self.model.M_spatial, :self.model.M_spatial]
        return rdm


class FockSpaceSolver:
    """
    Top-level solver that manages the calculation across different particle number (Nelec) sectors.
    
    It determines which Nelec values to solve for, and delegates the actual calculation 
    to GroundStateSolver.
    """
    def __init__(self, model, settings, nelec_range="auto", Nelec_imp=None):
        self.base_model = model    
        self.settings = settings
        
        # nelec_range can be:
        # - "auto": Will automatically search for the minimum energy around an estimated filling
        # - int: Will solve for exactly one Nelec
        # - tuple/list: Will solve for a fixed range/list of Nelecs
        self.nelec_setting = nelec_range 
        
        self.Nelec_imp = Nelec_imp
        self.result = None

    def _solve_single_nelec(self, nelec, cache):
        """Helper to run a GroundStateSolver for a single Nelec."""
        if nelec in cache:
            return cache[nelec]

        print(f"\n--- Solving for Nelec = {nelec} ---")
        
        # Pass the Nelec_imp to help GroundStateSolver generate good seeds
        solver = GroundStateSolver(
            self.base_model, 
            self.settings, 
            Nelec=nelec, 
            Nelec_imp=self.Nelec_imp
        )
        nelec_result_thermal = solver.solve() 
        
        cache[nelec] = nelec_result_thermal
        return nelec_result_thermal

    def _get_nelec_start_guess(self):
        """Determines the starting Nelec for an automatic search."""
        
        # Case 1: Impurity model with Nelec_imp known -> Calculate bath filling
        if self.base_model.is_impurity_model \
            and self.Nelec_imp is not None:
            

            M_imp = len(self.base_model.imp_indices_spatial)

            if self.base_model.M_spatial > M_imp :

                nelec_bath = hamiltonians.calculate_bath_filling(self.base_model.h0, M_imp)
                start = int(self.Nelec_imp + nelec_bath)
                print(f"INFO: 'auto' range. Estimated filling: {self.Nelec_imp} (imp) + {nelec_bath} (bath) = {start}")
                return start
            else : # HIA 
                start = int(self.Nelec_imp)
                return start

        # Case 2: Fallback to half-filling if no better info available
        print("INFO: 'auto' range. No impurity info found, defaulting to half-filling.")
        return self.base_model.M_spatial

    def _find_optimal_nelec(self):
        """Iteratively finds the ground state by searching around a starting guess."""
        nelec_start = self._get_nelec_start_guess()
        
        print(f"\n--- Starting Automatic Search for Optimal Nelec (start = {nelec_start}) ---")

        energies = {}
        results_cache = {}
        subspace_cache = {}

        # --- 1. Calculate Initial Point ---
        result_thermal = self._solve_single_nelec(nelec_start, results_cache)
        _, e0, _ = result_thermal.find_absolute_ground_state()
        energies[nelec_start] = e0
        subspace_cache[nelec_start] = result_thermal.results_by_nelec[nelec_start]
        
        # --- 2. Search Upwards ---
        nelec_curr = nelec_start
        while True:
            nelec_next = nelec_curr + 1
            if nelec_next > 2 * self.base_model.M_spatial:
                break
            
            e_curr = energies[nelec_curr]
            result_thermal_next = self._solve_single_nelec(nelec_next, results_cache)
            _, e_next, _ = result_thermal_next.find_absolute_ground_state()
            energies[nelec_next] = e_next
            subspace_cache[nelec_next] = result_thermal_next.results_by_nelec[nelec_next]
            
            if e_next >= e_curr:
                print("Energy increasing. Stopping upward search.")
                break
            nelec_curr = nelec_next

        # --- 3. Search Downwards ---
        nelec_curr = nelec_start
        while True:
            nelec_next = nelec_curr - 1
            if nelec_next < 0:
                break

            e_curr = energies[nelec_curr]
            result_thermal_next = self._solve_single_nelec(nelec_next, results_cache)
            _, e_next, _ = result_thermal_next.find_absolute_ground_state()
            energies[nelec_next] = e_next
            subspace_cache[nelec_next] = result_thermal_next.results_by_nelec[nelec_next]
            
            if e_next >= e_curr:
                print("Energy increasing. Stopping downward search.")
                break
            nelec_curr = nelec_next
            
        # --- 4. Collect Results around the minimum ---
        nelec_min = min(energies, key=energies.get)
        print(f"Minimum found at Nelec={nelec_min} with E={energies[nelec_min]}")
        
        final_subspaces = {}
        # We keep the min and its immediate neighbors for T > 0 stats
        for n in [nelec_min - 1, nelec_min, nelec_min + 1]:
            if n in subspace_cache:
                final_subspaces[n] = subspace_cache[n]
        
        return final_subspaces

    def solve(self):
        """
        Main entry point. Decides the strategy based on self.nelec_setting.
        """
        all_subspaces = {}

        # Strategy A: Automatic Search
        if self.nelec_setting == "auto":
            all_subspaces = self._find_optimal_nelec()
            
        # Strategy B: Single Fixed Nelec
        elif isinstance(self.nelec_setting, int):
            nelec = self.nelec_setting
            results_cache = {}
            result_thermal = self._solve_single_nelec(nelec, results_cache)
            all_subspaces[nelec] = result_thermal.results_by_nelec[nelec]

        # Strategy C: Fixed Range/List
        else:
            # Handle both tuple (min, max) or list [n1, n2, n3]
            if isinstance(self.nelec_setting, tuple):
                nelec_list = range(self.nelec_setting[0], self.nelec_setting[1] + 1)
            else:
                nelec_list = self.nelec_setting

            results_cache = {} 
            for nelec in nelec_list:
                result_thermal = self._solve_single_nelec(nelec, results_cache)
                all_subspaces[nelec] = result_thermal.results_by_nelec[nelec]
        
        # Combine all results into one ThermalGroundState
        self.result = results.ThermalGroundState(
            results_by_nelec=all_subspaces,
            base_model=self.base_model,
            temperature= self.settings.temperature
        )
        
        self.result.prune()
        return self.result

    def save_result(self, filename: str):
        if not self.result:
            raise RuntimeError("Solver has not been run yet.")
        self.result.save(filename)