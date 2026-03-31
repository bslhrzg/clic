import numpy as np

from ..solve.solver_api import FockSpaceSolver
from ..green.green_api import GreenFunctionCalculator
from .. import *

from clicvars import *
from hybfit_new import *
from solver_new import *

def dmft_step(
        ws,
        iws,
        hyb,
        h_imp,
        U_imp,
        rspt_clic_params
        ):

   


    dirdump = "dump_"+rspt_clic_params["label"] 
    clicvars = ClicVars()

    n_bath_poles = rspt_clic_params["n_bath_poles"]
    clicvars.nb = n_bath_poles

    fit_method = "cost_minimization"
    eta_hyb = 0.01
    clicvars.eta_hyb = eta_hyb

    eta_broad = 0.03
    clicvars.eta_broad = eta_broad

    warp_kind = "none"
    clicvars.warp_kind = warp_kind

    diag_fit = True 
    clicvars.windowing = False 
    window_width = 0.1 
    clicvars.window_width = window_width

    window_pos = 0.0
    clicvars.window_pos = window_pos 

    Nelec_imp = rspt_clic_params["Nelec_imp"]
    clicvars.Nelec_imp = Nelec_imp

    if n_bath_poles > 1:
        if n_bath_poles > 3:
            basis_prep_method = "dbl_chain" # or "dbl_chain"
        else : 
            basis_prep_method = "none"
        ci_type = "sci" # or "fci"
        num_roots = rspt_clic_params["num_roots"]
    else : 
        basis_prep_method = "none"
        ci_type = "fci"
        num_roots = 14

    clicvars.ci_type = ci_type
    clicvars.basis_prep_method = basis_prep_method

    max_iter = rspt_clic_params["num_roots"]
    clicvars.max_iter = max_iter
    
    conv_tol = rspt_clic_params["conv_tol"]
    clicvars.conv_tol = conv_tol

    prune_thr = 1e-12
    clicvars.prune_thr = prune_thr

    Nmul = rspt_clic_params["Nmul"]
    clicvars.Nmul = Nmul

    temperature=rspt_clic_params["temperature"]
    clicvars.temperature = temperature

    L_lanczos = 150 
    clicvars.L_lanczos = L_lanczos

    NappH = rspt_clic_params["NappH"]
    clicvars.NappH = NappH
    coeff_thresh = rspt_clic_params["lanczos_thr"]
    clicvars.coeff_thresh = coeff_thresh


    print("ENTERING DMFT STEP with params: ")

    for key in rspt_clic_params: 
        print(f"{key}: {rspt_clic_params[key]}")

    ################################################
    # Windowing the hybridization 
    # 
    def windowing_hyb(hyb, width, position):
        print("----------------------------------")
        print(f"APPLYING WINDOW ON HYB WITH WIDTH {width} AT {position}")
        def gpd(mu,sigma,x):
            C = 1 /(sigma * np.sqrt(2*np.pi))
            return C * np.exp(-0.5*(x-mu)**2 / (sigma**2))
        n_orb = hyb.shape[1]
        for i in range(n_orb):
            for j in range(n_orb):
                hyb[:,i,j] *= gpd(position,width,ws)
        
        return hyb 
    
    if clicvars.windowing:
        hyb = windowing_hyb(hyb,clicvars.window_width,clicvars.window_pos)
    ##################################################


    norb = hyb.shape[1]
    print(f"hyb.shape = {hyb.shape}")
    av_im_hyb = np.zeros((norb,norb))
    for i in range(norb):
        for j in range(norb):
            av_im_hyb[i,j] = np.mean(np.abs(np.imag(hyb[:,i,j])))

    print("<Im hyb_ij>")
    print(av_im_hyb)
    #assert 1 == 0
    # ==============================================================================
    # 2. DEFINE THE MODEL
    # ==============================================================================

    dump(np.real(hyb),ws,'real-hyb',output_dir=dirdump)
    dump(np.imag(hyb),ws,'imag-hyb',output_dir=dirdump)




    print("----------------------------------")
    print("FITTING DIAGONAL PART OF HYB ONLY")    
    hyb_to_fit = np.zeros_like(hyb)
    norb = hyb.shape[1]
    for i in range(norb):
        hyb_to_fit[:,i,i] = hyb[:,i,i]

    h_imp_to_fit = np.zeros_like(h_imp)
    for i in range(norb):
        h_imp_to_fit[i,i] = h_imp[i,i]
    
    #########################################
    h0_0,delta_fit, mapping = discretize_hyb(
        clicvars.ws,
        hyb_to_fit,           # (Nw, Nimp, Nimp)
        h_imp_to_fit,          # (Nimp, Nimp)
        clicvars.nb,
        clicvars.eta_hyb, 
        weight_func=clicvars.warp_kind,
        broadening_Gamma=clicvars.eta_broad,
    )
    #########################################

    hybdos = -np.trace(hyb, axis1=1, axis2=2).imag
    hybappdos = -np.trace(delta_fit,axis1=1, axis2=2).imag

    dump(hybdos,ws,"imhyb_0","./")
    dump(hybappdos,ws,"imhyb_fit","./")
    
    # ==============================================================================
    # 3. RUN THE SOLVER
    # ==============================================================================
    # Create the settings object (using your existing Pydantic structures)
   

   
    N_target = clicvars.Nelec_target #13.3
    U_0 = U_imp.copy()
    nelecs_resuls = solve_fockspace(h0_0,U_0,clicvars)

    thermal_gs, Ne_dict = build_state_list_and_ne_dict(nelecs_resuls)
    k_B_IN_RY_PER_K = 0.0000063336   # Ry/K 
    set_boltzmann_weights(thermal_gs, temperature, k_B_IN_RY_PER_K)

    prn_tgs_thr=1e-3
    thermal_gs = prune_states(thermal_gs, prn_tgs_thr)
    set_boltzmann_weights(thermal_gs, temperature, k_B_IN_RY_PER_K)

#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
#########################################
    print("\n--- Post-Solver Analysis ---")
    analyzer = StateAnalyzer(result, model)
    analyzer.do_analysis()
    # ==============================================================================
    # 4. CALCULATE GREEN'S FUNCTION
    # ==============================================================================


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


    dump(np.real(G_imp),ws,'real-G_real',output_dir=dirdump)
    dump(np.imag(G_imp),ws,'imag-G_real',output_dir=dirdump)
    dump(np.real(G_imp_iw),iws,'real-G_mats',output_dir=dirdump)
    dump(np.imag(G_imp_iw),iws,'imag-G_mats',output_dir=dirdump)


    # ==============================================================================
    # 4. CALCULATE SELF ENERGY
    # ==============================================================================

    hyb_approx = model.hyb_data["fitted"]
    hyb_approx_iw = model.hyb_data["fitted_iw"]

    dump(np.imag(hyb_approx),ws,'imag-hyb_app_real')


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
    

    def check_imag_diag_negative_(Sigma, name="Sigma",eps_sigma=1e-12):
        imag_diag = np.imag(np.diagonal(Sigma, axis1=1, axis2=2))
        bad = imag_diag > eps_sigma
        if np.any(bad):
            idx = np.argwhere(bad)
            print(f"{name}: Im Sigma_ii > 0 detected : Sigma[{idx[1::10]}] =  {Sigma[idx[::10]]}")
            Sigma[idx] = np.real(Sigma[idx]) - 0.001 *1j
        print(f"{name}: diagonal Im parts OK (<= 0 within tol)")


    def check_imag_diag_negative(Sigma, name="Sigma", eps_sigma=1e-12, clamp_im=-1e-3):
        # Sigma: (n_w, n, n)
        diag = np.diagonal(Sigma, axis1=1, axis2=2)          # shape (n_w, n)
        bad = np.imag(diag) > eps_sigma

        if np.any(bad):
            iw, i = np.nonzero(bad)                           # 1D arrays of same length
            print(f"{name}: Im Sigma_ii > 0 at {len(iw)} points. Showing a few:")
            for k in range(min(5, len(iw))):
                print(f"  (iw={iw[k]}, i={i[k]}): {Sigma[iw[k], i[k], i[k]]}")

            # clamp ONLY those diagonal entries
            Sigma[iw, i, i] = np.real(Sigma[iw, i, i]) + 1j*clamp_im

        else:
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
