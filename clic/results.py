import numpy as np
import h5py
from . import clic_clib as cc # for Wavefunction type hint

class NelecLowEnergySubspace:
    """
    Holds the results of a CI calculation for a fixed particle number.
    This object contains the computed eigenstates and the basis information
    they are expressed in.
    """
    def __init__(self,
                 M: int,
                 Nelec: int,
                 energies: np.ndarray,
                 wavefunctions: list[cc.Wavefunction],
                 basis: list[cc.SlaterDeterminant],
                 transformation_matrix: np.ndarray | None = None):
        
        self.M = M
        self.Nelec = Nelec
        
        self.energies = np.asarray(energies)
        self.wavefunctions = wavefunctions
        self.basis = basis # The common basis for all wavefunctions in this result
        self.transformation_matrix = transformation_matrix # From original to final basis

    @property
    def ground_state_energy(self) -> float:
        """Convenience property for the lowest energy eigenvalue."""
        return self.energies[0]

    @property
    def ground_state_wavefunction(self) -> cc.Wavefunction:
        """Convenience property for the ground state wavefunction."""
        return self.wavefunctions[0]

    def save(self, filename: str):
        """Saves the complete result to a structured HDF5 file."""
        print(f"Saving eigenstate results to '{filename}'...")
        if not self.basis:
            print("Warning: Cannot save result with an empty basis.")
            return

        # --- Flatten the basis occupation lists for robust HDF5 saving ---
        # For alpha electrons
        alpha_indices_flat = []
        alpha_endpoints = [0] # Store the end index for each determinant's data
        for det in self.basis:
            occ = det.alpha_occupied_indices()
            alpha_indices_flat.extend(occ)
            alpha_endpoints.append(len(alpha_indices_flat))
        
        # For beta electrons
        beta_indices_flat = []
        beta_endpoints = [0]
        for det in self.basis:
            occ = det.beta_occupied_indices()
            beta_indices_flat.extend(occ)
            beta_endpoints.append(len(beta_indices_flat))

        # Convert to plain, simple NumPy arrays
        alpha_indices_flat = np.asarray(alpha_indices_flat, dtype=np.int64)
        alpha_endpoints = np.asarray(alpha_endpoints, dtype=np.int64)
        beta_indices_flat = np.asarray(beta_indices_flat, dtype=np.int64)
        beta_endpoints = np.asarray(beta_endpoints, dtype=np.int64)

        with h5py.File(filename, 'w') as f:
            # === Metadata Group ===
            meta = f.create_group("metadata")
            meta.attrs["M"] = self.M
            meta.attrs["Nelec"] = self.Nelec
            meta.attrs["num_states"] = len(self.wavefunctions)
            meta.attrs["basis_size"] = len(self.basis)
            meta.create_dataset("energies", data=self.energies)
            
            # === Basis Group ===
            basis_gp = f.create_group("basis")
            if self.transformation_matrix is not None:
                basis_gp.create_dataset("transformation_matrix", data=self.transformation_matrix)
            
            # Save the flattened basis data structures
            basis_gp.create_dataset("alpha_indices_flat", data=alpha_indices_flat)
            basis_gp.create_dataset("alpha_endpoints", data=alpha_endpoints)
            basis_gp.create_dataset("beta_indices_flat", data=beta_indices_flat)
            basis_gp.create_dataset("beta_endpoints", data=beta_endpoints)
            
            # === Wavefunctions Group ===
            wf_gp = f.create_group("wavefunctions")
            for i, wf in enumerate(self.wavefunctions):
                if wf.get_basis() != self.basis:
                    raise RuntimeError(f"Basis mismatch for wavefunction {i} during save.")
                wf_gp.create_dataset(f"state_{i}_coeffs", data=wf.get_amplitudes())
        
        print("Save complete.")


    @classmethod
    def load(cls, filename: str):
        """Loads a complete result from an HDF5 file."""
        print(f"Loading eigenstate results from '{filename}'...")
        with h5py.File(filename, 'r') as f:
            # === Load Metadata ===
            meta = f["metadata"]
            M = int(meta.attrs["M"])
            Nelec = int(meta.attrs["Nelec"])
            num_states = int(meta.attrs["num_states"])
            basis_size = int(meta.attrs["basis_size"])
            energies = meta["energies"][:]
            
            # === Reconstruct Basis ===
            basis_gp = f["basis"]
            transformation_matrix = basis_gp["transformation_matrix"][:] if "transformation_matrix" in basis_gp else None
            
            alpha_indices_flat = basis_gp["alpha_indices_flat"][:]
            alpha_endpoints = basis_gp["alpha_endpoints"][:]
            beta_indices_flat = basis_gp["beta_indices_flat"][:]
            beta_endpoints = basis_gp["beta_endpoints"][:]

            alpha_list = [alpha_indices_flat[alpha_endpoints[i]:alpha_endpoints[i+1]] for i in range(basis_size)]
            beta_list = [beta_indices_flat[beta_endpoints[i]:beta_endpoints[i+1]] for i in range(basis_size)]
            
            basis = [cc.SlaterDeterminant(M, a, b) for a, b in zip(alpha_list, beta_list)]

            # === Reconstruct Wavefunctions ===
            wf_gp = f["wavefunctions"]
            wavefunctions = []
            for i in range(num_states):
                coeffs = wf_gp[f"state_{i}_coeffs"][:]
                wf = cc.Wavefunction(M, basis, coeffs)
                wavefunctions.append(wf)
            
            print("Load complete.")
            
            # Create and return a new instance of the class using the user-defined __init__
            return cls(
                M=M,
                Nelec=Nelec,
                energies=energies,
                wavefunctions=wavefunctions,
                basis=basis,
                transformation_matrix=transformation_matrix
            )