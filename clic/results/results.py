# clic/results.py
import numpy as np
import h5py
import clic_clib as cc # for Wavefunction type hint

# Use TYPE_CHECKING block to import for type hints only, preventing circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .api import Model


class NelecLowEnergySubspace:
    """
    Holds the results of a CI calculation for a fixed particle number.
    This object contains the computed eigenstates and the basis information
    they are expressed in.
    """
    def __init__(self,
                 M_spatial: int,
                 Nelec: int,
                 energies: np.ndarray,
                 wavefunctions: list[cc.Wavefunction],
                 basis: list[cc.SlaterDeterminant],
                 transformation_matrix: np.ndarray | None = None):
        
        self.M_spatial = M_spatial
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

    def save(self, target: str | h5py.Group):
        """
        Saves the complete result to a structured HDF5 file or group.
        
        Args:
            target: A filename (str) or an h5py.Group object.
        """
        def _save_to_group(group: h5py.Group):
            # Flatten basis occupation lists for robust HDF5 saving
            alpha_indices_flat = []
            alpha_endpoints = [0]
            for det in self.basis:
                alpha_indices_flat.extend(det.alpha_occupied_indices())
                alpha_endpoints.append(len(alpha_indices_flat))
            
            beta_indices_flat = []
            beta_endpoints = [0]
            for det in self.basis:
                beta_indices_flat.extend(det.beta_occupied_indices())
                beta_endpoints.append(len(beta_indices_flat))

            # === Metadata Group ===
            meta = group.create_group("metadata")
            meta.attrs["M_spatial"] = self.M_spatial
            meta.attrs["Nelec"] = self.Nelec
            meta.attrs["num_states"] = len(self.wavefunctions)
            meta.attrs["basis_size"] = len(self.basis)
            meta.create_dataset("energies", data=self.energies)
            
            # === Basis Group ===
            basis_gp = group.create_group("basis")
            if self.transformation_matrix is not None:
                basis_gp.create_dataset("transformation_matrix", data=self.transformation_matrix)
            
            basis_gp.create_dataset("alpha_indices_flat", data=np.asarray(alpha_indices_flat, dtype=np.int64))
            basis_gp.create_dataset("alpha_endpoints", data=np.asarray(alpha_endpoints, dtype=np.int64))
            basis_gp.create_dataset("beta_indices_flat", data=np.asarray(beta_indices_flat, dtype=np.int64))
            basis_gp.create_dataset("beta_endpoints", data=np.asarray(beta_endpoints, dtype=np.int64))
            
            # === Wavefunctions Group ===
            wf_gp = group.create_group("wavefunctions")
            for i, wf in enumerate(self.wavefunctions):
                if wf.get_basis() != self.basis:
                    raise RuntimeError(f"Basis mismatch for wavefunction {i} during save.")
                wf_gp.create_dataset(f"state_{i}_coeffs", data=wf.get_amplitudes())

        if isinstance(target, str):
            print(f"Saving Nelec={self.Nelec} subspace to file '{target}'...")
            with h5py.File(target, 'w') as f:
                _save_to_group(f)
        else: # Assumes target is an h5py.Group
            print(f"Saving Nelec={self.Nelec} subspace to HDF5 group '{target.name}'...")
            _save_to_group(target)
        
        print("Save complete.")

    @classmethod
    def load(cls, source: str | h5py.Group):
        """
        Loads a complete result from an HDF5 file or group.
        
        Args:
            source: A filename (str) or an h5py.Group object.
        """
        def _load_from_group(group: h5py.Group):
            # Load Metadata
            meta = group["metadata"]
            M_spatial = int(meta.attrs["M_spatial"])
            Nelec = int(meta.attrs["Nelec"])
            num_states = int(meta.attrs["num_states"])
            basis_size = int(meta.attrs["basis_size"])
            energies = meta["energies"][:]
            
            # Reconstruct Basis
            basis_gp = group["basis"]
            transformation_matrix = basis_gp["transformation_matrix"][:] if "transformation_matrix" in basis_gp else None
            
            alpha_indices_flat = basis_gp["alpha_indices_flat"][:]
            alpha_endpoints = basis_gp["alpha_endpoints"][:]
            beta_indices_flat = basis_gp["beta_indices_flat"][:]
            beta_endpoints = basis_gp["beta_endpoints"][:]

            alpha_list = [alpha_indices_flat[alpha_endpoints[i]:alpha_endpoints[i+1]] for i in range(basis_size)]
            beta_list = [beta_indices_flat[beta_endpoints[i]:beta_endpoints[i+1]] for i in range(basis_size)]
            
            basis = [cc.SlaterDeterminant(M_spatial, a, b) for a, b in zip(alpha_list, beta_list)]

            # Reconstruct Wavefunctions
            wf_gp = group["wavefunctions"]
            wavefunctions = []
            for i in range(num_states):
                coeffs = wf_gp[f"state_{i}_coeffs"][:]
                wf = cc.Wavefunction(M_spatial, basis, coeffs)
                wavefunctions.append(wf)
            
            return cls(M_spatial=M_spatial, Nelec=Nelec, energies=energies, wavefunctions=wavefunctions,
                       basis=basis, transformation_matrix=transformation_matrix)

        if isinstance(source, str):
            print(f"Loading eigenstate results from file '{source}'...")
            with h5py.File(source, 'r') as f:
                instance = _load_from_group(f)
        else: # Assumes source is an h5py.Group
            print(f"Loading eigenstate results from HDF5 group '{source.name}'...")
            instance = _load_from_group(source)
            
        print("Load complete.")
        return instance

# We define the Boltzmann constant in eV/K.
K_B_IN_EV_PER_K = 8.617333262e-5 # eV/K
k_B_IN_RY_PER_K = 0.0000063336   # Ry/K 

class ThermalGroundState:
    """
    Holds and manages a collection of low-energy eigenstates from different
    particle number (Nelec) sectors to describe a system at a given temperature.

    It is assumed that the chemical potential has already been absorbed into the
    one-body part of the Hamiltonian used to generate these eigenstates.
    """
    def __init__(self,
                 results_by_nelec: dict[int, 'NelecLowEnergySubspace'],
                 base_model: 'Model',
                 temperature: float = 300.0):
        
        if not isinstance(results_by_nelec, dict):
            raise TypeError("`results_by_nelec` must be a dictionary.")

        # --- Store base model information ---
        self.base_h0 = base_model.h0
        self.base_U = base_model.U
        self.M_spatial = base_model.M_spatial
        print(f"base_model.is_impurity_model = {base_model.is_impurity_model}")
        self.is_impurity_model = base_model.is_impurity_model
        self.imp_indices_spatial = base_model.imp_indices_spatial
        
        self.results_by_nelec = results_by_nelec
        self._temperature = temperature
        print(f"WOLOLOOOOO: TEMPERATURE HERE ? {temperature}")
        
        self._all_states: list[tuple[float, int, cc.Wavefunction]] = []
        for nelec, result in self.results_by_nelec.items():
            for i, energy in enumerate(result.energies):
                self._all_states.append((energy, nelec, result.wavefunctions[i]))

        self._all_states.sort(key=lambda s: s[0])
        self.boltzmann_weights: np.ndarray | None = None
        self.partition_function: float | None = None
        self._recalculate_thermal_properties()

    @property
    def temperature(self) -> float:
        """The temperature of the system in Kelvin."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """Sets the temperature and recalculates thermal properties."""
        if value <= 0:
            raise ValueError("Temperature must be positive.")
        self._temperature = value
        self._recalculate_thermal_properties()
        
    def _recalculate_thermal_properties(self):
        """
        Internal method to update Boltzmann weights and partition function
        whenever temperature changes.
        """
        if not self._all_states:
            self.boltzmann_weights = np.array([])
            self.partition_function = 0.0
            return

        # Assumes energies are in eV. Change k_B constant if units differ.
        #beta = 1.0 / (K_B_IN_EV_PER_K * self._temperature)
        beta = 1.0 / (k_B_IN_RY_PER_K * self._temperature)
        
        energies = np.array([s[0] for s in self._all_states])
        
        # Subtract the ground state energy to prevent numerical overflow in exp()
        # This factor cancels out upon normalization.
        ground_state_energy = energies[0] # Since the list is sorted
        unnormalized_weights = np.exp(-beta * (energies - ground_state_energy))
        
        self.partition_function = np.sum(unnormalized_weights)
        self.boltzmann_weights = unnormalized_weights / self.partition_function
        
        print(f"Recalculated thermal properties for T={self.temperature} K.")

    def prune(self, threshold: float = 1e-3):
        """
        Removes states whose Boltzmann weight is below a given threshold.
        This method completely rebuilds the internal state of the object,
        purging all information related to the pruned states.
        """
        print("*"*42)
        print(f"DEBUG: PRUNING WITH threshold = {threshold}")
        if self.boltzmann_weights is None:
            self._recalculate_thermal_properties()

        initial_count = len(self._all_states)
        if initial_count == 0:
            print("No states to prune.")
            return

        # 1. Identify which states to keep
        indices_to_keep = np.where(self.boltzmann_weights >= threshold)[0]
        
        if len(indices_to_keep) == initial_count:
            print(f"No states pruned with threshold {threshold:.1e}.")
            return

        # 2. Keep only the surviving states in the flattened list
        self._all_states = [self._all_states[i] for i in indices_to_keep]
        
        # --- 3. CRUCIAL STEP: Rebuild the results_by_nelec dictionary from scratch ---
        new_results_by_nelec = {}
        
        # Group surviving states by their Nelec
        grouped_states = {}
        for energy, nelec, wf in self._all_states:
            if nelec not in grouped_states:
                grouped_states[nelec] = []
            grouped_states[nelec].append({"energy": energy, "wavefunction": wf})

        # For each group, create a new, minimal NelecLowEnergySubspace object
        for nelec, states_info in grouped_states.items():
            states_info.sort(key=lambda x: x["energy"])
            surviving_wfs = [s["wavefunction"] for s in states_info]
            surviving_energies = [s["energy"] for s in states_info]
            
            # Create a minimal basis just for the surviving wavefunctions in this sector
            new_basis_set = set()
            for wf in surviving_wfs:
                new_basis_set.update(wf.data().keys())
            pruned_basis = sorted(list(new_basis_set))
            
            # Re-express the wavefunctions in this new minimal basis
            new_wavefunctions = []
            det_to_idx = {det: i for i, det in enumerate(pruned_basis)}
            original_M_spatial = self.results_by_nelec[nelec].M_spatial
            for wf in surviving_wfs:
                new_coeffs = np.zeros(len(pruned_basis), dtype=np.complex128)
                for det, coeff in wf.data().items():
                    new_coeffs[det_to_idx[det]] = coeff
                new_wavefunctions.append(cc.Wavefunction(original_M_spatial, pruned_basis, new_coeffs))
            
            # Get other metadata from the original object (this is the only time we need it)
            original_transform = self.results_by_nelec[nelec].transformation_matrix

            # Create the new, clean result object for this Nelec sector
            new_results_by_nelec[nelec] = NelecLowEnergySubspace(
                M_spatial=original_M_spatial,
                Nelec=nelec,
                energies=np.array(surviving_energies),
                wavefunctions=new_wavefunctions,
                basis=pruned_basis,
                transformation_matrix=original_transform
            )

        # 4. Replace the old, complete dictionary with the new, pruned one
        self.results_by_nelec = new_results_by_nelec

        # 5. Finally, recalculate the Boltzmann weights for the pruned set
        self._recalculate_thermal_properties()
        final_count = len(self._all_states)
        print(f"Pruned {initial_count - final_count} states. {final_count} states remaining.")

    def find_absolute_ground_state(self) -> tuple[int, float, cc.Wavefunction]:
        """
        Finds the state with the lowest energy across all calculated Nelec sectors.

        Returns:
            A tuple of (Nelec, ground_state_energy, ground_state_wavefunction).
        """
        if not self._all_states:
            raise ValueError("No states are stored.")

        # Since _all_states is sorted by energy, the ground state is the first one.
        gs_energy, gs_nelec, gs_wf = self._all_states[0]
        return gs_nelec, gs_energy, gs_wf

    def save(self, filename: str):
        """Saves all contained results and model info into a single HDF5 file."""
        print(f"Saving thermal state data to '{filename}'...")
        with h5py.File(filename, 'w') as f:
            f.attrs["file_type"] = "ThermalGroundState"
            f.attrs["temperature"] = self.temperature
            
            # --- Save the base model context ---
            f.attrs["M_spatial"] = self.M_spatial
            f.attrs["is_impurity_model"] = self.is_impurity_model
            if self.is_impurity_model:
                f.attrs["imp_indices_spatial"] = self.imp_indices_spatial
            f.create_dataset("base_h0", data=self.base_h0)
            f.create_dataset("base_U", data=self.base_U)

            for nelec, result in self.results_by_nelec.items():
                nelec_group = f.create_group(f"nelec_{nelec}")
                print(f"Saving Nelec={nelec} subspace to HDF5 group '{nelec_group.name}'...")
                result.save(nelec_group)
        print("Save complete.")


    @classmethod
    def load(cls, filename: str):
        """Loads a thermal state result from a single HDF5 file."""
        print(f"Loading thermal state data from '{filename}'...")
        results = {}
        with h5py.File(filename, 'r') as f:
            if f.attrs.get("file_type") != "ThermalGroundState":
                 print(f"Warning: File '{filename}' may not be a valid ThermalGroundState file.")
            
            temp = f.attrs.get("temperature", 300.0)
            
            # Import Model locally to avoid circular dependency at module level
            from .api import Model

            # --- Load the base model context ---
            M_spatial = int(f.attrs["M_spatial"])
            is_imp = bool(f.attrs["is_impurity_model"])
            imp_indices_spatial = list(f.attrs.get("imp_indices_spatial", []))
            base_h0 = f["base_h0"][:]
            base_U = f["base_U"][:]
            
            # Reconstruct the model object. Nelec is just a placeholder here.
            loaded_model = Model(h0=base_h0, U=base_U, M_spatial=M_spatial, Nelec=-1)
            loaded_model.is_impurity_model = is_imp
            loaded_model.imp_indices_spatial = imp_indices_spatial

            for key in f.keys():
                if key.startswith("nelec_"):
                    nelec = int(key.split("_")[1])
                    nelec_group = f[key]
                    print(f"Loading Nelec={nelec} subspace from HDF5 group '{nelec_group.name}'...")
                    results[nelec] = NelecLowEnergySubspace.load(nelec_group)
        
        print("Load complete.")
        return cls(results, base_model=loaded_model, temperature=temp)
