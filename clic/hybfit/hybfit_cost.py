# hybfit/hybfit_cost.py

import numpy as np
from scipy import optimize
from ..io_utils import vprint

class HybFitCost:
    """
    Fit a pole representation to a SCALAR hybridization function Δ(ω) by
    minimizing a weighted cost function using global optimization.
    """

    def __init__(self, omega_grid, delta_complex, n_target_poles,logfile=None):
        delta_in = np.asarray(delta_complex, dtype=np.complex128)
        if delta_in.ndim != 3 or delta_in.shape[1] != 1 or delta_in.shape[2] != 1:
            raise ValueError(
                "HybFitCost only supports scalar hybridization of shape (N, 1, 1)."
            )

        self.logfile = logfile

        self.omega = np.asarray(omega_grid, dtype=float)
        self.delta_input_scalar = delta_in[:, 0, 0]
        self.n_target_poles = int(n_target_poles)


        # Final results
        self.eps_final = None
        self.R_final = None

        vprint(3,"HybFitCost initialized:",filename=self.logfile)
        vprint(3,f"  Method: Global optimization (scalar only)",filename=self.logfile)
        vprint(3,f"  Target poles = {self.n_target_poles}",filename=self.logfile)

    def run_fit(self, eta_0, bounds_e=None, weight_func='const', broadening_Gamma=0.00, **opt_kwargs):
        """
        Execute the fitting by running the global optimization algorithm.
        """
        vprint(3,"--- Fitting poles via Cost Minimization ---",filename=self.logfile)
        
        if bounds_e is None:
            bounds_e = [np.min(self.omega), np.max(self.omega)]
        vprint(3,f"Using energy bounds for bath sites: [{bounds_e[0]:.2f}, {bounds_e[1]:.2f}]",filename=self.logfile)

        if broadening_Gamma > 0:
            vprint(3,f"Applying Lorentzian broadening of width Gamma = {broadening_Gamma}",filename=self.logfile)
            target_delta = self._lorentzian_convolution(self.delta_input_scalar, broadening_Gamma)
        else:
            target_delta = self.delta_input_scalar
        
        eta_fit = eta_0 + broadening_Gamma
        vprint(3,f"Fit will be performed with effective broadening eta_fit = {eta_fit:.4f}",filename=self.logfile)

        # Set optimizer defaults if not provided
        opt_defaults = {'strategy': 'best1bin', 'maxiter': 500, 'popsize': 15, 'tol': 1e-2, 'disp': False}
        opt_settings = {**opt_defaults, **opt_kwargs}
        
        bounds_list = ([(bounds_e[0], bounds_e[1])] * self.n_target_poles +
                       [(-10.0, 10.0)] * (2 * self.n_target_poles))

        vprint(3,f"Starting global optimization ({opt_settings['strategy']})...",filename=self.logfile)
        result = optimize.differential_evolution(
            self._cost_function,
            bounds=bounds_list,
            args=(target_delta, eta_fit, weight_func),
            **opt_settings
        )

        if result.success:
            vprint(3,f"Optimization successful. Final cost (χ²): {result.fun:.6e}",filename=self.logfile)
        else:
            vprint(3,f"WARNING: Optimization may not have converged. Final cost (χ²): {result.fun:.6e}",filename=self.logfile)

        # Unpack, sort, and store results
        opt_params = result.x
        e = opt_params[:self.n_target_poles]
        v = opt_params[self.n_target_poles:2*self.n_target_poles] + 1j * opt_params[2*self.n_target_poles:]
        
        sort_idx = np.argsort(e)
        self.eps_final = e[sort_idx]
        v_final = v[sort_idx]
        self.R_final = [np.array([[np.abs(vi)**2]]) for vi in v_final] # Store as standard residue matrix
        
        vprint(3,"--- Cost Fit Done ---",filename=self.logfile)
        vprint(3,"Final optimized poles:",filename=self.logfile)
        for i in range(len(self.eps_final)):
            c = (np.abs(self.R_final[i][0,0]))
            vprint(3,f"  pole {i}: e = {self.eps_final[i]:+.6f}, coupling |v|^2 = {c:.6f}",filename=self.logfile)
        return self

    def _cost_function(self, params, target_delta, eta, weight_str):
        n_b = self.n_target_poles
        e = params[:n_b]
        v = params[n_b : 2*n_b] + 1j * params[2*n_b:]
        
        model_delta = self._hybridization_model(e, v, eta)
        difference = target_delta - model_delta
        
        if weight_str == "const": weight = 1.0
        elif weight_str == "inv2": weight = 1.0 / (self.omega**2 + 1e-2)
        else: raise ValueError(f"Unknown weight: {weight_str}")
            
        return np.sum((weight * np.abs(difference))**2)

    def _hybridization_model(self, e, v, eta):
        return np.sum(np.abs(v)**2 / (self.omega[:, None] + 1j * eta - e), axis=1)

    def _lorentzian_convolution(self, y, width):
        result = np.zeros_like(y, dtype=np.complex128)
        for i, omega_i in enumerate(self.omega):
            kernel = (1 / np.pi) * width / ((omega_i - self.omega)**2 + width**2)
            result[i] = np.trapz(y * kernel, self.omega)
        return result