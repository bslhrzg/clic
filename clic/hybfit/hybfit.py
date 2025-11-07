# hybfit/hybfit.py

import numpy as np
from . import utils
from .hybfit_poles import HybFitPoles
from .hybfit_cost import HybFitCost
from clic.io_clic.io_utils import vprint

def fit(omega, delta, n_poles, method, *, 
        eta_0=None, 
        n_lanczos_blocks=None, 
        broadening_Gamma=None,
        weight_func=None,
        bounds_e=None,
        logfile=None,
        **kwargs):    
    """
    High-level function to fit a hybridization function.

    Args:
        omega (np.ndarray): Frequency grid.
        delta (np.ndarray): Complex hybridization function, shape (N, M, M) or (N,).
        n_poles (int): The target number of poles for the model.
        method (str): The fitting method to use.
                      Options: 'poles_reconstruction', 'cost_minimization'.
    Keyword-Only Args:
        eta_0 (float): Intrinsic broadening. Required for 'cost_minimization'.
        n_lanczos_blocks (int): Blocks for Lanczos. Default: 10 * n_poles.
        broadening_Gamma (float): Extra broadening for 'cost_minimization'. Default: 0.00.
        weight_func (str): Weight for 'cost_minimization'. Default: 'const'.
        bounds_e (list): Energy bounds for 'cost_minimization'. Default: min/max of omega.
        **kwargs: Other advanced arguments for the specific methods.
            - For 'poles_reconstruction': warp_kind, warp_w0.
            - For 'cost_minimization': Scipy optimizer args (maxiter, tol, etc.).

    Returns:
        tuple[np.ndarray, list[np.ndarray]]:
            - eps_fit: Array of fitted pole energies.
            - R_fit: List of fitted residue matrices.
    """
    if method == 'poles_reconstruction':

        # Set default for n_lanczos_blocks if not provided
        if n_lanczos_blocks is None:
            n_lanczos_blocks = 10 * n_poles

        # This includes explicit args and any relevant ones from kwargs.
        pole_args = {
            'warp_kind': kwargs.pop('warp_kind', 'atanh'),
            'warp_w0': kwargs.pop('warp_w0', 0.01)}
        
        fitter = HybFitPoles(
            omega_grid=omega,
            delta_complex=delta,
            n_lanczos_blocks=n_lanczos_blocks,
            n_target_poles=n_poles,
            logfile=logfile
        )
        fitter.run_fit(**pole_args)

        return fitter.eps_merged, fitter.R_merged

    elif method == 'cost_minimization':
        # --- Handle arguments for Cost Minimization ---
        if eta_0 is None:
            raise ValueError("'eta_0' is a required argument for 'cost_minimization' method.")

        # 1. Set defaults for explicit arguments if not provided
        if broadening_Gamma is None:
            broadening_Gamma = 0.00
        if weight_func is None:
            weight_func = 'const'
        # bounds_e can remain None, the class will handle it
            
        # 2. The remaining kwargs are assumed to be for the optimizer
        #    and are passed directly.
        fitter = HybFitCost(
            omega_grid=omega,
            delta_complex=delta,
            n_target_poles=n_poles,
            logfile=logfile
        )
        # Pass the explicit arguments by name, and the rest in kwargs
        fitter.run_fit(
            eta_0=eta_0, 
            broadening_Gamma=broadening_Gamma,
            weight_func=weight_func,
            bounds_e=bounds_e,
            #**kwargs  # Pass through advanced optimizer settings
        )
        return fitter.eps_final, fitter.R_final

    else:
        raise ValueError(f"Unknown fitting method: {method}.")


def analyze_fit(omega, delta_true, eps_fit, R_fit, eta, logfile = None):
    """
    Performs a standard analysis of a hybridization fit.

    Calculates the model hybridization from the fitted parameters and computes
    a set of standard error metrics, printing a summary.

    Args:
        omega (np.ndarray): Frequency grid.
        delta_true (np.ndarray): The original hybridization function.
        eps_fit (np.ndarray): Fitted pole energies.
        R_fit (list[np.ndarray]): Fitted residue matrices.
        eta (float): Broadening to use for reconstructing the model for comparison.

    Returns:
        dict: A dictionary containing the calculated error metrics.
    """
    vprint(1,"\n" + "="*50,filename=logfile)
    vprint(1," " * 18 + "FIT ANALYSIS",filename=logfile)
    vprint(1,"="*50,filename=logfile)

    # 1. Reconstruct the model
    z = omega + 1j * eta
    delta_model = utils.delta_from_poles(z, eps_fit, R_fit)

    # 2. Calculate errors
    err_l2_re = utils.rel_l2_error(np.real(delta_true), np.real(delta_model))
    err_l2_im = utils.rel_l2_error(np.imag(delta_true), np.imag(delta_model))
    
    chi_const = utils.cost_l2_integral(delta_true, delta_model, omega, weight='const')
    chi_inv2 = utils.cost_l2_integral(delta_true, delta_model, omega, weight='inv2')
    
    vprint(1,f"Analysis performed with broadening eta = {eta:.4f}",filename=logfile)
    vprint(1,"\n--- Error Metrics ---",filename=logfile)
    vprint(1,f"  Relative L2 Error (Re): {err_l2_re:.4e}",filename=logfile)
    vprint(1,f"  Relative L2 Error (Im): {err_l2_im:.4e}",filename=logfile)
    vprint(1,f"  Cost L2 Integral (const): {chi_const:.4e}",filename=logfile)
    vprint(1,f"  Cost L2 Integral (inv2):  {chi_inv2:.4e}",filename=logfile)

    # 3. Check moments
    moment_results = utils.check_moment_conservation(omega, delta_true, eps_fit, R_fit, logfile=logfile)
    vprint(1,"="*50,filename=logfile)
    
    errors = {
        'rel_l2_re': err_l2_re,
        'rel_l2_im': err_l2_im,
        'cost_const': chi_const,
        'cost_inv2': chi_inv2,
        'moments': moment_results
    }
    return errors