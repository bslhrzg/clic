import numpy as np

from .clicvars import *
from . import *

def dmft_step(
        ws,
        iws,
        hyb,
        h_imp,
        U_imp,
        rspt_clic_params=None,
        ):




   
    clicvars = ClicVars.from_toml("input.toml")

    if rspt_clic_params is not None:
        print("ENTERING DMFT STEP with params: ")

        for key in rspt_clic_params: 
            print(f"{key}: {rspt_clic_params[key]}")

        dirdump = "dump_"+rspt_clic_params["label"] 
        clicvars.dirdump = dirdump

        n_bath_poles = rspt_clic_params["n_bath_poles"]
        clicvars.nb = n_bath_poles

        Nelec_imp = rspt_clic_params["Nelec_imp"]
        clicvars.Nelec_imp = Nelec_imp

        num_roots = rspt_clic_params["num_roots"]

        conv_tol = rspt_clic_params["conv_tol"]
        clicvars.conv_tol = conv_tol


        Nmul = rspt_clic_params["Nmul"]
        clicvars.Nmul = Nmul

        temperature=rspt_clic_params["temperature"]
        clicvars.temperature = temperature

        NappH = rspt_clic_params["NappH"]
        clicvars.NappH = NappH

        coeff_thresh = rspt_clic_params["lanczos_thr"]
        clicvars.coeff_thresh = coeff_thresh



    clicvars.M_imp = h_imp.shape[0] // 2
    clicvars.imp_indices_spatial = [i for i in range(clicvars.M_imp)]


    clicvars.ws = ws 
    clicvars.iws = iws

    hybdos = -np.trace(hyb, axis1=1, axis2=2).imag
    dump(hybdos,ws,"imhyb_0_dos_test",output_dir=clicvars.dirdump)  


    #clicvars.windowing = False 
    #window_width = 0.1 
    #clicvars.window_width = window_width
    #window_pos = 0.0
    #clicvars.window_pos = window_pos 


    if clicvars.nb > 0:
        if clicvars.nb > 3:
            basis_prep_method = "dbl_chain" # or "dbl_chain"
        else : 
            basis_prep_method = "none"
        ci_type = "sci" # or "fci"
        
        do_hia = False
    else : 
        basis_prep_method = "none"
        ci_type = "fci"
        num_roots = 14
        do_hia = True
        print("----- GOING WITH HIA -----")


    clicvars.ci_type = ci_type
    clicvars.basis_prep_method = basis_prep_method

    

    prune_thr = 1e-12
    clicvars.prune_thr = prune_thr



    L_lanczos = 150 
    clicvars.L_lanczos = L_lanczos
    clicvars.eta = clicvars.eta_hyb







    print("CLICVARS : ")
    clicvars.print()
    # ==============================================================================
    # 2. DEFINE THE MODEL
    # ==============================================================================

    if not do_hia and not clicvars.freeze_bath :

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
        
        #if clicvars.windowing:
        #    hyb = windowing_hyb(hyb,clicvars.window_width,clicvars.window_pos)
        ##################################################


        norb = hyb.shape[1]
        print(f"hyb.shape = {hyb.shape}")
        av_im_hyb = np.zeros((norb,norb))
        for i in range(norb):
            for j in range(norb):
                av_im_hyb[i,j] = np.mean(np.abs(np.imag(hyb[:,i,j])))

        print("<Im hyb_ij>")
        print(av_im_hyb)

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
        if clicvars.windowing : 
            e_min = clicvars.window_pos - clicvars.window_width / 2
            e_max = clicvars.window_pos + clicvars.window_width / 2
            bounds_e = [e_min,e_max]
        else : 
            bounds_e = None
            
        h0_0,delta_fit, hyb_approx, mapping, hyb_approx_iw = discretize_hyb(
            clicvars.ws,
            hyb_to_fit,           # (Nw, Nimp, Nimp)
            h_imp_to_fit,          # (Nimp, Nimp)
            clicvars.nb,
            clicvars.eta_hyb, 
            weight_func=clicvars.warp_kind,
            broadening_Gamma=clicvars.eta_broad,
            i_omegas=iws, 
            bounds_e=bounds_e
        )

        NF = np.shape(h0_0)[0]
        M_spatial = NF // 2
        iis = clicvars.imp_indices_spatial + [i+M_spatial for i in clicvars.imp_indices_spatial]
        h0_0[np.ix_(iis,iis)] = h_imp
        
        np.savetxt("h0_0_real.txt",h0_0.real)
        np.savetxt("h0_0_imag.txt",h0_0.imag)

        #########################################

        hybdos = -np.trace(hyb, axis1=1, axis2=2).imag
        hybappdos = -np.trace(hyb_approx,axis1=1, axis2=2).imag

        dump(hybdos,ws,"imhyb_0_dos",output_dir=clicvars.dirdump)
        dump(hybappdos,ws,"imhyb_fit_dos",output_dir=clicvars.dirdump)
        
        dump(np.imag(hyb_approx),ws,'imag-hyb_app',output_dir=clicvars.dirdump)
        dump(np.real(hyb_approx),ws,'real-hyb_app',output_dir=clicvars.dirdump)

        dump(np.imag(hyb_approx_iw),iws,'imag-hyb-mats_app',output_dir=clicvars.dirdump)
        dump(np.real(hyb_approx_iw),iws,'real-hyb-mats_app',output_dir=clicvars.dirdump)

    else : 
        if do_hia:
            h0_0 = h_imp.copy()
            hyb_approx = None 
            hyb_approx_iw = None
        else : 
            if clicvars.freeze_bath: 
                print("USING FROZEN BATH IF CAN BE FOUND")
                h0_0_real = np.loadtxt("h0_0_real.txt")    
                h0_0_imag = np.loadtxt("h0_0_imag.txt")    
                h0_0 = h0_0_real + 1j*h0_0_imag
                NF = np.shape(h0_0)[0]
                M_spatial = NF // 2
                iis = clicvars.imp_indices_spatial + [i+M_spatial for i in clicvars.imp_indices_spatial]
                h0_0[np.ix_(iis,iis)] = h_imp

                norb = h_imp.shape[0]
                ws2, hyb_app_imag = load_3d("imag-hyb_app", shape_2d=(norb, norb), output_dir=clicvars.dirdump)
                ws2, hyb_app_real = load_3d("real-hyb_app", shape_2d=(norb, norb), output_dir=clicvars.dirdump)
                hyb_approx = hyb_app_real + 1j * hyb_app_imag

                iws2, hyb_app_mats_imag = load_3d("imag-hyb-mats_app", shape_2d=(norb, norb), output_dir=clicvars.dirdump)
                iws2, hyb_app_mats_real = load_3d("real-hyb-mats_app", shape_2d=(norb, norb), output_dir=clicvars.dirdump)
                hyb_approx_iw = hyb_app_mats_real + 1j * hyb_app_mats_imag

                hybdos = -np.trace(hyb, axis1=1, axis2=2).imag
                hybappdos = -np.trace(hyb_approx,axis1=1, axis2=2).imag

            dump(hybdos,ws,"imhyb_0_dos",output_dir=clicvars.dirdump)


    dump(np.real(hyb),ws,'real-hyb',output_dir=clicvars.dirdump)
    dump(np.imag(hyb),ws,'imag-hyb',output_dir=clicvars.dirdump)

    # ==============================================================================
    # 3. RUN THE SOLVER
    # ==============================================================================
    # Create the settings object (using your existing Pydantic structures)
   

   
    N_target = clicvars.Nelec_target #13.3
    NF = np.shape(h0_0)[0]
    M_spatial = NF // 2 
    clicvars.M_spatial = M_spatial
    clicvars.NF = NF 
    U_0 = np.zeros((NF,NF,NF,NF),dtype=complex)
    clicvars.imp_indices_spinfull = clicvars.imp_indices_spatial + [i+M_spatial for i in clicvars.imp_indices_spatial]
    iis =  clicvars.imp_indices_spinfull

    print(f"imp_spinorb_index = {clicvars.imp_indices_spinfull}")
    U_0[np.ix_(iis,iis,iis,iis)] = U_imp

    h0_0 = np.ascontiguousarray(h0_0, dtype=np.complex128)
    U_0 = np.ascontiguousarray(U_0, dtype=np.complex128)


    nelecs_resuls = solve_fockspace(h0_0,U_0,clicvars)
    thermal_gs, Ne_dict = build_state_list_and_ne_dict(nelecs_resuls)
    k_B_IN_RY_PER_K = 0.0000063336   # Ry/K 
    set_boltzmann_weights(thermal_gs, clicvars.temperature, k_B_IN_RY_PER_K)

    prn_tgs_thr=1e-3
    thermal_gs = prune_states(thermal_gs, prn_tgs_thr)
    set_boltzmann_weights(thermal_gs, clicvars.temperature, k_B_IN_RY_PER_K)

    print("\n--- Post-Solver Analysis ---")
    thermal_avgs = analyze_thermal_gs(thermal_gs, clicvars)

    clicvars.green_block_indices = clicvars.imp_indices_spinfull

    ws, G_imp, G_imp_iw, A_imp = get_green(clicvars,Ne_dict,h0_0,U_0,thermal_gs,plot_sf = True)
    
    dump(np.real(G_imp),ws,'real-G_real',output_dir=clicvars.dirdump)
    dump(np.imag(G_imp),ws,'imag-G_real',output_dir=clicvars.dirdump)
    dump(np.real(G_imp_iw),iws,'real-G_mats',output_dir=clicvars.dirdump)
    dump(np.imag(G_imp_iw),iws,'imag-G_mats',output_dir=clicvars.dirdump)


    # ==============================================================================
    # 4. CALCULATE SELF ENERGY
    # ==============================================================================


    if clicvars.nb > 0:
        hyb_sig = hyb_approx 
        hyb_sig_iw = hyb_approx_iw

    else :
        hyb_sig = None
        hyb_sig_iw = None

    print(f"DEBUG: iws[0] = {iws[0]}, clicvars.iws[0] = {clicvars.iws[0]}")
    Sigma, G0 = calculate_self_energy( 
                                clicvars,
                                ws, 
                                h_imp,
                                G_imp, 
                                hyb_sig)

    Sigma_iw, G0_iw = calculate_self_energy( 
                                clicvars,
                                1j * iws, 
                                h_imp,
                                G_imp_iw, 
                                hyb_sig_iw)
    

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




    dump(np.real(Sigma),ws,'real-sig_real',output_dir=clicvars.dirdump)
    dump(np.imag(Sigma),ws,'imag-sig_real',output_dir=clicvars.dirdump)
    dump(np.real(Sigma_iw),iws,'real-sig_mats',output_dir=clicvars.dirdump)
    dump(np.imag(Sigma_iw),iws,'imag-sig_mats',output_dir=clicvars.dirdump)

    dump(np.real(G0),ws,'real-G0_real',output_dir=clicvars.dirdump)
    dump(np.imag(G0),ws,'imag-G0_real',output_dir=clicvars.dirdump)
    dump(np.real(G0_iw),iws,'real-G0_mats',output_dir=clicvars.dirdump)
    dump(np.imag(G0_iw),iws,'imag-G0_mats',output_dir=clicvars.dirdump)

    # Compute static self energy 
    avg_rdm_imp = thermal_avgs["rho_imp_thermal"]
    sig_static = np.einsum('ikjl,ij->kl', U_imp, avg_rdm_imp) - \
                np.einsum('iklj,ij->kl', U_imp, avg_rdm_imp)

    np.savetxt("real-sig_static.dat", np.real(sig_static), fmt="% 8.5f")
    np.savetxt("imag-sig_static.dat", np.imag(sig_static), fmt="% 8.5f")

    if clicvars.spin_avg_sigma : 
        print("*"*42)
        print("Performing spin average over self energies before return")
        sig_static = spin_average_one_particle(sig_static)
        Sigma = spin_average_one_particle(Sigma)
        Sigma_iw = spin_average_one_particle(Sigma_iw)


    return sig_static,Sigma,Sigma_iw