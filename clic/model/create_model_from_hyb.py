# clic/create_model_from_hyb.py
import numpy as np
from .config_models import HybFitConfig
import sys 
from clic.hybfit.utils import *
from clic.hybfit.process_hyb import *

def build_model_from_hyb(
    h_imp: np.ndarray, 
    omega: np.ndarray, 
    hyb: np.ndarray, 
    hybfit_config: HybFitConfig
) -> np.ndarray:
    """
    Builds the full one-body hamiltonian from impurity data and a fit.
    Returns: The full h0 matrix.
    """

    n_target_poles = hybfit_config.n_target_poles
    warp_kind = hybfit_config.warp_kind
    eta = hybfit_config.eta_in

    logfile = "hybfit_details.txt"

    if hybfit_config.method == "poles_reconstruction":

        warp_k=None
        if warp_kind == "none":
            warp_k = "const"
        elif warp_kind == "emph0":
            warp_k = "asinh"
        else :
            print(f"ERROR: Unknown model parameter type '{warp_kind}'", file=sys.stderr)
            sys.exit(1)

        warp_w0 = hybfit_config.warp_w0

        H_full, map = process_hyb_poles(
        omega, hyb, h_imp, n_target_poles,n_lanczos_blocks=101, 
        warp_kind=warp_k, warp_w0=warp_w0,logfile=logfile
        )

        block_errs = analyze_block_fits(omega, hyb, map, eta=eta)
        global_errs = evaluate_full_fit_and_plots(
            omega, hyb, H_full, map, eta=eta, out_dir="hyb_plots", case_tag=""
        )

        print_summary("Fit block summary ",H_full, map)

    elif hybfit_config.method == "cost_minimization":

        broadening_Gamma = hybfit_config.eta_broad

        weight_func=None
        if warp_kind == "none":
            weight_func = "const"
        elif warp_kind == "emph0":
            weight_func = "inv2"



        print("----- COST -------")
        H_full, map = process_hyb_cost(
            omega, hyb, h_imp,
            n_target_poles=n_target_poles,
            eta_0=eta,                 # same broadening you used to generate hyb1
            bounds_e=[omega.min(), omega.max()],   # or a tighter physical window
            weight_func='const',
            broadening_Gamma=broadening_Gamma,
            logfile=logfile
            )
        
        block_errs = analyze_block_fits(omega, hyb, map, eta=eta, logfile=logfile)
        #global_errs = evaluate_full_fit_and_plots(
        #    omega, hyb, H_full, map, eta=eta, out_dir="hyb_plots", case_tag=""
        #)

        print_summary("Fit block summary ",H_full, map)

    # 2. Reconstruct the model hybridization delta_fit for return
    # Extract bath parameters from the permuted H_full
    Nimp = len(map["alpha_imp_idx"]) + len(map["beta_imp_idx"])
    _, _, V, Hb = unpermute_to_block_form(H_full, map["perm_full_to_spin_sorted"], Nimp)
    
    # Rebuild delta using the same eta as the input data for direct comparison
    delta_fit = delta_from_bath(omega, Hb, V, eta=eta)
        

    return H_full, delta_fit