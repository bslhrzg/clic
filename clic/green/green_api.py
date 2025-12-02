# clic/green/green_api.py 

import numpy as np
#from tqdm import tqdm    # For the progress bar
from typing import Optional 
from scipy.linalg import eigh, block_diag
from time import time

from clic.basis import basis_1p, basis_Np
from clic.ops import ops
from clic.green import plotting, green_sym, self_energy as se 
from clic.io_clic import io_utils
from clic.results import results # For type hinting
from clic.model.config_models import GreenFunctionConfig, OutputConfig
from clic.symmetries import symmetries


class GreenFunctionCalculator:
    """
    Calculates the thermally-averaged Green's function from a saved ThermalGroundState.
    Also computes Self-Energy via the Dyson equation.
    """
    def __init__(self, gf_config: GreenFunctionConfig, output_config: OutputConfig, ground_state_filepath: str):
        self.gf_config = gf_config
        self.output_config = output_config
        self.ground_state_filepath = ground_state_filepath
        self.thermal_state: results.ThermalGroundState | None = None

    def load_thermal_state(self):
        """Loads the ThermalGroundState from HDF5 and prepares it for calculation."""
        filepath = self.ground_state_filepath
        print(f"Loading thermal state from HDF5 file '{filepath}'...")
        try:
            self.thermal_state = results.ThermalGroundState.load(filepath)
        except (FileNotFoundError, KeyError) as e:
            raise RuntimeError(f"Failed to load ground state file: {e}")
        
        print(f"Thermal state loaded. Initial temperature: {self.thermal_state.temperature:.1f} K.")
        print(f"Prepared thermal state with {len(self.thermal_state._all_states)} states for GF calculation.")

    def get_target_indices(self) -> list[int]:
        """Helper to resolve the indices of the Green's function block."""
        if self.thermal_state is None:
            raise ValueError("Thermal state not loaded.")
            
        if self.gf_config.block_indices == "impurity":
            if not self.thermal_state.is_impurity_model:
                raise ValueError("GF block set to 'impurity' but loaded state is not an impurity model.")
            # Combine spatial indices for both spin sectors (Up and Down)
            # Assuming h0 is spin-full (2*M_spatial x 2*M_spatial)
            M_sp = self.thermal_state.M_spatial
            target_indices = sorted(self.thermal_state.imp_indices_spatial
                                    + [i + M_sp for i in self.thermal_state.imp_indices_spatial])
        else:
            target_indices = sorted(list(set(self.gf_config.block_indices)))
        
        return target_indices

    def run(self, ground_state_result: Optional[results.ThermalGroundState] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the thermally-averaged Green's function calculation.
        
        Returns:
            ws: Frequency mesh
            G_total: Full Green's function matrix (nw, norb, norb)
            A_w_total: Spectral function (diagonal part)
        """
        if ground_state_result:
            print("Using in-memory ground state result from solver step.")
            self.thermal_state = ground_state_result
        else:
            self.load_thermal_state() # This method loads from file

        p_gf = self.gf_config
        p_lanczos = self.gf_config.lanczos
        if len(p_gf.omega_mesh) == 3:
            ws = np.linspace(p_gf.omega_mesh[0], p_gf.omega_mesh[1], int(p_gf.omega_mesh[2]))
        else : 
            ws = np.array(p_gf.omega_mesh)


        iws = p_gf.matsubara_mesh
        print(f"DEBUG: here in green_api run : iws is not None : {iws is not None}")
        if iws is not None :
            iws = 1j * np.array(iws)

        target_indices = self.get_target_indices()
        num_target = len(target_indices)
        print(f"\nTargeting Green's function block for indices: {target_indices}")
        
        # We initialize the full matrix for the block (nw, N_imp, N_imp)
        # This is required for subsequent Self-Energy calculations (matrix inversion)
        G_total = np.zeros((len(ws), num_target, num_target), dtype=np.complex128)
        
        if iws is not None : 
            G_total_iw = np.zeros((len(iws), num_target, num_target), dtype=np.complex128)
        else : 
            G_total_iw = None

        # Cache for transformed Hamiltonians to avoid redundant calculations
        hamiltonian_cache = {}
        base_h0 = self.thermal_state.base_h0
        base_U = self.thermal_state.base_U
        M = self.thermal_state.M_spatial

        # Main loop over all states in the thermal ensemble
        iterator = zip(self.thermal_state._all_states, self.thermal_state.boltzmann_weights)
        #for (state_info, weight) in tqdm(iterator, total=len(self.thermal_state._all_states), desc="Processing thermal states"):
        for (state_info, weight) in iterator:
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

            #gfmeth = "block"
            #gfmeth = "scalar_continued_fraction"
            # gfmeth = "time_prop"

            # G_sub_block_n has shape (nw, num_target, num_target)
            G_sub_block_n, G_sub_block_n_iw = green_sym.get_green_block(M, psi_n, e_n, p_lanczos.NappH, p_gf.eta, 
                                                      h0_n, U_n, ws, iws,  target_indices, 
                                                      one_bh_n, two_bh_n, p_lanczos.coeff_thresh, p_lanczos.L
                                                      )
 
            # Add the weighted contribution to the total Green's function matrix
            # Note: In the original code this was a loop over diagonals. 
            # Now we accumulate the whole matrix to support self-energy calculation.
            G_total += weight * G_sub_block_n

            if iws is not None:
                G_total_iw += weight * G_sub_block_n_iw

        print("\nThermally-averaged calculation finished.")
        
        # Extract diagonal for spectral function plotting
        # np.diagonal returns (nw, norb), we want (nw, norb) but carefully checked
        # diagonal() doc: if axis1=1, axis2=2, returns (nw, norb)
        G_total_diag = np.diagonal(G_total, axis1=1, axis2=2)        
        A_w_total = -(1 / np.pi) * np.imag(G_total_diag)
        
        # Save and plot the final, thermally-averaged results
        dodump = True 
        if dodump:
            io_utils.dump(
                A_w_total,
                ws,
                'A_w_thermal',
            )
            # Also dump the Green's function if needed for post-processing
            # io_utils.dump_complex(G_total, ws, 'G_w_thermal')

        if self.output_config.plot_file:
            plotting.plot_spectral_function(
                ws, A_w_total, list(range(num_target)),
                f"Thermally-Averaged Spectral Function (T={self.thermal_state.temperature}K)",
                self.output_config.plot_file
            )
        
        return ws, G_total, G_total_iw, A_w_total


    def calculate_self_energy(self, 
                              ws: np.ndarray, 
                              G_imp: np.ndarray, 
                              hyb: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the Self-Energy matrix using the Dyson equation block by block based on symmetry.
            Sigma(w) = G0_inv(w) - G_imp_inv(w)
        
        """
        if self.thermal_state is None:
            raise RuntimeError("Thermal state is not loaded. Cannot extract impurity Hamiltonian.")

        # 1. Extract Impurity Hamiltonian (h_imp)
        target_indices = self.get_target_indices()
        h_imp = self.thermal_state.base_h0[np.ix_(target_indices, target_indices)]
        
        # 2. Analyze Symmetries of h_imp
        # This gives us indices relative to the N_imp x N_imp matrix
        symdict = symmetries.analyze_symmetries(h_imp)
        blocks = symdict["blocks"]
        identical_groups = symdict["identical_groups"]
        is_diagonal = symdict["is_diagonal"]

        print(f"Self-Energy Calculation: Found {len(identical_groups)} unique symmetry groups.")
        print(f"blocks : {blocks}")
        
        # 3. Prepare Hybridization
        hyb_to_use = hyb
        if hyb_to_use is None:
            if hasattr(self.thermal_state, 'hyb_data') and self.thermal_state.hyb_data is not None:
                print("Using hybridization data found in thermal_state.")
                hyb_data = self.thermal_state.hyb_data
                hyb_to_use = hyb_data['fitted'] if isinstance(hyb_data, dict) and 'fitted' in hyb_data else hyb_data
        
        # If still None, treat as zero matrix
        if hyb_to_use is None:
            print("No hybridization found. Sigma = Sigma_atomic.")
            hyb_to_use = np.zeros((len(ws), h_imp.shape[0], h_imp.shape[1]), dtype=np.complex128)

        # 4. Initialize Sigma container
        Sigma_total = np.zeros_like(G_imp)
        G0_total = np.zeros_like(G_imp)

        if np.imag(ws[0]) == 0:
            eta = self.gf_config.eta
        else:
            eta = 0

        # 5. Loop over symmetry groups (Compute Leader -> Copy to Followers)
        for group in identical_groups:
            # --- A. Process the Representative Block ---
            rep_block_idx = group[0]
            # Flatten indices to ensure we get a list of integers
            local_indices = np.array(blocks[rep_block_idx]).flatten()
            
            # Slice matrices for this specific block
            # h_sub: (N_sub, N_sub)
            h_sub = h_imp[np.ix_(local_indices, local_indices)]
            
            # G_sub, Delta_sub: (nw, N_sub, N_sub)
            # Use sophisticated slicing to extract the sub-block over all frequencies
            G_sub = G_imp[:, local_indices, :][:, :, local_indices]
            Delta_sub = hyb_to_use[:, local_indices, :][:, :, local_indices]

            # --- B. Compute Sigma for this block ---
            # Construct G0_inv for this block: (w + i*eta)I - h - Delta
            # Identity matrix needs to match block size
            eye_sub = np.eye(len(local_indices))
            
            # Broadcast w to (nw, N_sub, N_sub)
            z = ws[:, None, None] * eye_sub + 1j * eta * eye_sub
            
            # G0^{-1} calculation
            inv_G0_sub = z - h_sub - Delta_sub
            
            # G^{-1} calculation
            #if is_diagonal:
            if len(group) == 1:
                # Optimized path for diagonal blocks (1x1 or diagonal matrices)
                # Avoids linalg.inv numerical noise completely
                inv_G_sub = np.zeros_like(G_sub)
                # Safe division handling could be added here if G is exactly 0
                # Using diagonal() to view, assuming G_sub is diagonal
                diags = np.diagonal(G_sub, axis1=1, axis2=2)
                inv_diags = 1.0 / diags
                # Reconstruct diagonal matrix (broadcasting is tricky, loop is safe for small N)
                for k in range(len(local_indices)):
                    inv_G_sub[:, k, k] = inv_diags[:, k]
            else:
                # Standard block inversion
                #inv_G_sub = np.linalg.inv(G_sub)
                #G_sub = 0.5 * (G_sub + np.swapaxes(G_sub.conj(), 1, 2))
                N_sub = G_sub.shape[-1]
                I_sub = np.eye(N_sub)[None, :, :]           # (1, N_sub, N_sub)
                inv_G_sub = np.linalg.solve(G_sub, I_sub)   # broadcast over first axis

            # Dyson: Sigma = G0^{-1} - G^{-1}
            Sigma_sub = inv_G0_sub - inv_G_sub

            # --- C. Place into Total Matrix ---
            # 1. Fill the representative block
            for i_src, i_dest in enumerate(local_indices):
                for j_src, j_dest in enumerate(local_indices):
                    Sigma_total[:, i_dest, j_dest] = Sigma_sub[:, i_src, j_src]
                    G0_total[:, i_dest, j_dest] = np.linalg.inv(inv_G0_sub)[:, i_src, j_src]

            # 2. Copy to Equivalent Blocks
            for other_block_idx in group[1:]:
                other_indices = np.array(blocks[other_block_idx]).flatten()
                
                # Copy element by element from the calculated Sigma_sub
                for i_src, i_dest in enumerate(other_indices):
                    for j_src, j_dest in enumerate(other_indices):
                        Sigma_total[:, i_dest, j_dest] = Sigma_sub[:, i_src, j_src]

        print(f"Self-Energy calculated via symmetry blocks. Shape: {Sigma_total.shape}")
        return Sigma_total, G0_total
    
