# clic/api.py
import numpy as np
from . import bath_transform, clic_clib as cc # Import for Wavefunction, SlaterDeterminant
from . import hamiltonians, basis_1p, basis_Np, ops, sci, mf, gfs, plotting
from . import results
from .config_models import SolverParameters, GfConfig # Use Pydantic for validated settings
from scipy.linalg import eig,block_diag
import copy


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
            hmf, es, Vs, rho = mf.mfscf(self.model.h0,self.model.U,self.model.Nelec)
            h0,U_mat = basis_1p.basis_change_h0_U(self.model.h0,self.model.U,Vs)
            self.model.h0 = h0
            print(f"shape h0 = {np.shape(h0)}")
            print("new diag h0 : ")
            print(np.diag(h0))
            self.model.U = U_mat
            self.transformation_matrix = Vs

        elif method in ["dbl_chain","bath_no"]:

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



        
        else:
            raise NotImplementedError(f"Basis prep method '{method}' not implemented.")


    def solve(self) -> results.NelecLowEnergySubspace:
        """Runs the full workflow and returns a dictionary of results."""

        self._prepare_basis()
        
        ci_settings = self.settings.ci_method
        
        if ci_settings.type == "sci":
            print("Starting Selective CI calculation...")

            # Initial basis 
            if self.settings.basis_prep_method == 'rhf' :
                seed = basis_Np.get_rhf_determinant(self.model.Nelec, self.model.M)
            else :
                if self.model.is_impurity_model :
                    seed =  basis_Np.get_imp_starting_basis(
                        np.real(self.model.h0), self.model.Nelec, self.model.Nelec_imp, self.model.imp_indices)
                else : 
                    seed = basis_Np.get_starting_basis(np.real(self.model.h0), self.model.Nelec)  # returns list of SlaterDeterminant

            result_obj = sci.selective_ci(
                h0=self.model.h0, 
                U=self.model.U,
                M=self.model.M, 
                Nelec=self.model.Nelec,
                seed = seed,
                generator=sci.hamiltonian_generator, 
                selector=sci.cipsi_one_iter,
                max_iter=ci_settings.max_iter, 
                conv_tol=ci_settings.conv_tol,
                prune_thr=ci_settings.prune_thr,
                Nmul=ci_settings.Nmul, 
                verbose=True
            )
        elif ci_settings.type == "fci":
            print("Careful, only Sz=0 sector computed in fci for now")
            result_obj = sci.do_fci(
                h0=self.model.h0, U=self.model.U, M=self.model.M, Nelec=self.model.Nelec
                ,Sz=0,verbose=True
            )

        result_obj.transformation_matrix = self.transformation_matrix
        self.result = result_obj
        
        print(f"\nCalculation finished. Ground state energy = {self.result.ground_state_energy:.12f}")
        return self.result


    def get_one_rdm(self) -> np.ndarray:
        """Computes and returns the one-particle reduced density matrix, 
        for the ground state only
        """
        if not self.result:
            raise RuntimeError("Solver has not been run yet.")
        
        wavefunction = self.result.wavefunctions[0]

        return ops.one_rdm(wavefunction, self.model.M)
    
    # In the GroundStateSolver class, inside save_result method

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

class GreenFunctionCalculator:
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