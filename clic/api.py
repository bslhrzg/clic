# clic/api.py
import numpy as np
from . import hamiltonians, basis_1p, ops, sci, mf, basis_transforms
from .config_models import SolverParameters # Use Pydantic for validated settings

class Model:
    """Represents the physical system via its Hamiltonian integrals."""
    def __init__(self, h0: np.ndarray, U: np.ndarray, M_spatial: int, Nelec: int):
        self.h0 = np.ascontiguousarray(h0, dtype=np.complex128)
        self.U = np.ascontiguousarray(U, dtype=np.complex128)
        self.M = M_spatial
        self.Nelec = Nelec
        # We can add u, mu etc. if basis_transforms needs them
        self.u = U[0, M_spatial, 0, M_spatial].real if U.ndim == 4 and U.shape[0] > M_spatial else 0.0

class GroundStateSolver:
    """The main API endpoint for running a ground state calculation."""
    def __init__(self, model: Model, settings: dict | SolverParameters):
        self.model = model
        # If raw dict is passed, validate it with our Pydantic model
        if isinstance(settings, dict):
            self.settings = SolverParameters(**settings)
        else:
            self.settings = settings
        
        self.result = {}

    def solve(self) -> dict:
        """Runs the full workflow and returns a dictionary of results."""
        self._prepare_basis()
        
        ci_settings = self.settings.ci_method
        if ci_settings.type == "sci":
            print("Starting Selective CI calculation...")
            self.result = sci.selective_ci(
                h0=self.model.h0, U=self.model.U, M=self.model.M, Nelec=self.model.Nelec,
                generator=sci.hamiltonian_generator, selector=sci.cipsi_one_iter,
                max_iter=ci_settings.max_iter, conv_tol=ci_settings.conv_tol,
                prune_thr=ci_settings.prune_thr, Nmul=ci_settings.Nmul, verbose=True
            )
        elif ci_settings.type == "fci":
            raise NotImplementedError("FCI solver not implemented in API yet.")
        
        print(f"\nGround state solution found. Final Energy = {self.result['energy']:.12f}")
        return self.result

    def _prepare_basis(self):
        method = self.settings.basis_prep_method
        print(f"Preparing one-particle basis using method: '{method}'")
        if method == "none":
            return

        elif method == "dbl_chain":
            h0_spin = np.real(self.model.h0[:self.model.M, :self.model.M])
            Nelec_half = self.model.Nelec // 2
            
            final_params = basis_transforms.perform_natural_orbital_transform(h0_spin, self.model.u, Nelec_half)
            h_final_matrix = basis_transforms.construct_final_hamiltonian_matrix(final_params, self.model.M)
            h_final_matrix[0, 0] = -self.model.u / 2
            
            h0_new = basis_1p.double_h(h_final_matrix, self.model.M)
            self.model.h0 = h0_new # Update model's h0 in place
        
        else:
            raise NotImplementedError(f"Basis prep method '{method}' not implemented.")

    def get_one_rdm(self) -> np.ndarray:
        """Computes and returns the one-particle reduced density matrix."""
        if "wavefunction" not in self.result:
            raise RuntimeError("Solver has not been run yet. Call solve() first.")
        return ops.one_rdm(self.result["wavefunction"], self.model.M)