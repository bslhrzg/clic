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
        clic_params
        ):


    # ==============================================================================
    # 2. DEFINE THE MODEL
    # ==============================================================================

    n_bath_poles = clic_params["n_bath_poles"]

    fit_method = "cost_minimization"
    #fit_method = "poles_reconstruction"
    eta_hyb = 0.005
    eta_broad = 0.02

    model = Model.from_hybridization(h_imp, U_imp, ws, hyb, n_bath_poles, eta_hyb,
                            fit_method=fit_method,
                            warp_kind = "emph0",
                            warp_w0 = 0.01,
                            eta_broad = eta_broad,
                            iws = iws)

    print(f"model.is_impurity_model : {model.is_impurity_model}")
    print(f"model.imp_indices_spatial : {model.imp_indices_spatial}")

    Nelec_imp = clic_params["Nelec_imp"]
    # ==============================================================================
    # 3. RUN THE SOLVER
    # ==============================================================================
    # Create the settings object (using your existing Pydantic structures)
    if n_bath_poles > 0:
        if n_bath_poles > 3:
            basis_prep_method = "dbl_chain" # or "dbl_chain"
        else : 
            basis_prep_method = "none"
        ci_type = "sci" # or "fci"
        num_roots = clic_params["num_roots"]
    else : 
        basis_prep_method = "none"
        ci_type = "fci"
        num_roots = 14
    max_iter = clic_params["num_roots"]
    conv_tol = clic_params["conv_tol"]
    prune_thr = 1e-5
    Nmul = clic_params["Nmul"]


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
        temperature=clic_params["temperature"]
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
    NappH = clic_params["NappH"]
    coeff_thresh = clic_params["lanczos_thr"]

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

    dirdump = "dump_"+clic_params["label"]

    dump(np.real(G_imp),ws,'real-G_real',output_dir=dirdump)
    dump(np.imag(G_imp),ws,'imag-G_real',output_dir=dirdump)
    dump(np.real(G_imp_iw),iws,'real-G_mats',output_dir=dirdump)
    dump(np.imag(G_imp_iw),iws,'imag-G_mats',output_dir=dirdump)


    # ==============================================================================
    # 4. CALCULATE SELF ENERGY
    # ==============================================================================

    hyb_approx = model.hyb_data["fitted"]
    hyb_approx_iw = model.hyb_data["fitted_iw"]

    #dump(np.imag(hyb_approx),ws,'imag-hyb_real')
    #dump(np.imag(hyb_approx_iw),iws,'imag-hyb_mats')


    if n_bath_poles > 0:
        hyb_sig = hyb_approx 
        hyb_sig_iw = hyb_approx_iw

    else :
        hyb_sig = None
        hyb_sig_iw = None

    Sigma, G0 = gf_calc.calculate_self_energy( 
                                ws, 
                                G_imp, 
                                hyb_sig)

    Sigma_iw, G0_iw = gf_calc.calculate_self_energy( 
                                1j * iws, 
                                G_imp_iw, 
                                hyb_sig_iw)
    

    def check_imag_diag_negative(Sigma, name="Sigma",eps_sigma=1e-12):
        imag_diag = np.imag(np.diagonal(Sigma, axis1=1, axis2=2))
        bad = imag_diag > eps_sigma
        if np.any(bad):
            idx = np.argwhere(bad)
            raise ValueError(
                f"{name}: Im Sigma_ii > 0 detected at indices (iw, orb): {idx[:10]}"
            )
        print(f"{name}: diagonal Im parts OK (<= 0 within tol)")

    check_imag_diag_negative(Sigma, "Sigma_real")
    check_imag_diag_negative(Sigma_iw, "Sigma_mats")




    dump(np.real(Sigma),ws,'real-sig_real',output_dir=dirdump)
    dump(np.imag(Sigma),ws,'imag-sig_real',output_dir=dirdump)
    dump(np.real(Sigma_iw),iws,'real-sig_mats',output_dir=dirdump)
    dump(np.imag(Sigma_iw),iws,'imag-sig_mats',output_dir=dirdump)

    dump(np.real(G0),ws,'real-G0_real',output_dir=dirdump)
    dump(np.imag(G0),ws,'imag-G0_real',output_dir=dirdump)
    dump(np.real(G0_iw),iws,'real-G0_mats',output_dir=dirdump)
    dump(np.imag(G0_iw),iws,'imag-G0_mats',output_dir=dirdump)

    # Compute static self energy 
    avg_rdm_imp = analyzer.rho_imp_thermal
    sig_static = np.einsum('ikjl,ij->kl', U_imp, avg_rdm_imp) - \
                np.einsum('iklj,ij->kl', U_imp, avg_rdm_imp)

    np.savetxt("real-sig_static.dat", np.real(sig_static), fmt="% 8.5f")
    np.savetxt("imag-sig_static.dat", np.imag(sig_static), fmt="% 8.5f")

    return sig_static,Sigma,Sigma_iw
