import numpy as np
from .model.config_models import SolverParameters, CiMethodConfig, GreenFunctionConfig, LanczosParameters, OutputConfig

from .model.model_api import Model
from .solve.solver_api import FockSpaceSolver
from .green.green_api import GreenFunctionCalculator
from . import *



def dmft_step(
        ws,
        iws,
        hyb,h_imp,
        U_imp,
        ):


    # ==============================================================================
    # 2. DEFINE THE MODEL
    # ==============================================================================

    n_bath_poles = 0

    fit_method = "cost_minimization"
    #fit_method = "poles_reconstruction"
    eta_hyb = 0.005
    eta_broad = 0.00

    model = Model.from_hybridization(h_imp, U_imp, ws, hyb, n_bath_poles, eta_hyb,
                            fit_method=fit_method,
                            warp_kind = "emph0",
                            warp_w0 = 0.01,
                            eta_broad = eta_broad,
                            iws = iws)

    print(f"model.is_impurity_model : {model.is_impurity_model}")
    print(f"model.imp_indices_spatial : {model.imp_indices_spatial}")

    Nelec_imp = 1
    # ==============================================================================
    # 3. RUN THE SOLVER
    # ==============================================================================
    # Create the settings object (using your existing Pydantic structures)
    if n_bath_poles > 0:
        basis_prep_method = "dbl_chain" # or "dbl_chain"
        ci_type = "sci" # or "fci"
        num_roots = 4
    else : 
        basis_prep_method = "none"
        ci_type = "fci"
        num_roots = 14
    max_iter = 6
    conv_tol = 1e-5
    prune_thr = 1e-5
    Nmul = 1


    solver_settings = SolverParameters(
        basis_prep_method=basis_prep_method, # or "dbl_chain" or "rhf"
        ci_method=CiMethodConfig(
            type=ci_type, 
            num_roots=num_roots, 
            max_iter=max_iter, 
            conv_tol=conv_tol,
            prune_thr=prune_thr,
            Nmul=Nmul,
        ),
        initial_temperature=300.0
    )

    # Instantiate the Manager (FockSpaceSolver)
    # We pass 'auto' so it finds the correct bath filling automatically
    # We pass Nelec_imp so it knows how to calculate that filling
    solver = FockSpaceSolver(
        model=model, 
        settings=solver_settings, 
        nelec_range="auto", 
        Nelec_imp=Nelec_imp
    )

    result = solver.solve()

    print("\n--- Post-Solver Analysis ---")
    analyzer = StateAnalyzer(result, model)
    analyzer.do_analysis()
    # ==============================================================================
    # 4. CALCULATE GREEN'S FUNCTION
    # ==============================================================================
    L_lanczos = 150 
    NappH = 2
    coeff_thresh = 1e-7

    gf_config = GreenFunctionConfig(
        omega_mesh=ws,
        matsubara_mesh = iws,
        eta=eta_hyb,
        block_indices="impurity",
        lanczos=LanczosParameters(L=L_lanczos, 
                                NappH=NappH, 
                                coeff_thresh=coeff_thresh)
    )

    out_config = OutputConfig(basename="my_script_run", plot_file="spectral.pdf")

    gf_calc = GreenFunctionCalculator(
        gf_config=gf_config,
        output_config=out_config,
        ground_state_filepath="" # Ignored because we pass result directly below
    )

    # Pass the result from the solver directly to the GF calculator
    ws, G_imp, G_imp_iw, A_imp = gf_calc.run(ground_state_result=result)

    dump(np.real(G_imp),ws,'real-G_real')
    dump(np.imag(G_imp),ws,'imag-G_real')
    dump(np.real(G_imp_iw),iws,'real-G_mats')
    dump(np.imag(G_imp_iw),iws,'imag-G_mats')


    # ==============================================================================
    # 4. CALCULATE SELF ENERGY
    # ==============================================================================

    hyb_approx = model.hyb_data["fitted"]
    hyb_approx_iw = model.hyb_data["fitted_iw"]

    if n_bath_poles > 0:
        hyb_sig = hyb_approx 
        hyb_sig_iw = hyb_approx_iw

    else :
        hyb_sig = None
        hyb_sig_iw = None

    Sigma = gf_calc.calculate_self_energy( 
                                ws, 
                                G_imp, 
                                hyb_sig)

    Sigma_iw = gf_calc.calculate_self_energy( 
                                1j * iws, 
                                G_imp_iw, 
                                hyb_sig_iw)


    dump(np.real(Sigma),ws,'real-sig_real')
    dump(np.imag(Sigma),ws,'imag-sig_real')
    dump(np.real(Sigma_iw),iws,'real-sig_mats')
    dump(np.imag(Sigma_iw),iws,'imag-sig_mats')

    # Compute static self energy 
    avg_rdm_imp = analyzer.rho_imp_thermal
    sig_static = np.einsum('ikjl,ij->kl', U_imp, avg_rdm_imp) - \
                np.einsum('iklj,ij->kl', U_imp, avg_rdm_imp)

    np.savetxt("real-sig_static.dat", np.real(sig_static), fmt="% 8.5f")
    np.savetxt("imag-sig_static.dat", np.imag(sig_static), fmt="% 8.5f")

    return sig_static,Sigma,Sigma_iw


def dmft_step_(
        ws,
        iws,
        hyb,h_imp,
        U_imp,
        ):
    return None,None,None