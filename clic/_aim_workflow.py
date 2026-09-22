"""Shared many-body and impurity Green-function workflow."""

import numpy as np

from .solve.solver_api import (
    solve_fockspace, build_state_list_and_ne_dict, set_boltzmann_weights, prune_states,
)
from .solve.postprocessing import analyze_thermal_gs
from .green.green_api import get_green
from .green.gfs import occupation_from_green
from .io_clic.io_utils import dump


def _solve_aim_green(h0_0, U_imp, clicvars, *, plot_sf=True, full_block=False):
    """Solve an assembled AlphaFirst AIM; retain DMFT's thermal-state policy."""
    iws = clicvars.iws
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


    print(f"DEBUG: NF = {NF}, h0_0.shape = {h0_0.shape}")
    nelecs_results = solve_fockspace(h0_0,U_0,clicvars)


    thermal_gs, Ne_dict = build_state_list_and_ne_dict(nelecs_results)
    k_B_IN_RY_PER_K = 0.0000063336   # Ry/K 
    set_boltzmann_weights(thermal_gs, clicvars.temperature, k_B_IN_RY_PER_K)

    prn_tgs_thr=1e-3
    thermal_gs = prune_states(thermal_gs, prn_tgs_thr)
    set_boltzmann_weights(thermal_gs, clicvars.temperature, k_B_IN_RY_PER_K)

    print("\n--- Post-Solver Analysis ---")
    thermal_avgs = analyze_thermal_gs(thermal_gs, clicvars)

    clicvars.green_block_indices = clicvars.imp_indices_spinfull

    ws, G_imp, G_imp_iw, A_imp = get_green(
        clicvars, Ne_dict, h0_0, U_0, thermal_gs,
        plot_sf=plot_sf, full_block=full_block,
    )

    n_orb, n_tot = occupation_from_green(ws, G_imp, beta=np.inf, mu=0.0)
    print("Occupation from Green function:")
    for i, n in enumerate(n_orb):
        print(f"  orb {i:3d}: {n:.8f}")
    print(f"Total occupation from G: {n_tot:.8f}")
    print(f"Total occupation variance <N_f^2> - <N_f>^2: {thermal_avgs['var_occ']:.8f}")

    dump(np.real(G_imp),ws,'real-G_real',output_dir=clicvars.dirdump)
    dump(np.imag(G_imp),ws,'imag-G_real',output_dir=clicvars.dirdump)
    dump(np.real(G_imp_iw),iws,'real-G_mats',output_dir=clicvars.dirdump)
    dump(np.imag(G_imp_iw),iws,'imag-G_mats',output_dir=clicvars.dirdump)

    return {
        "ws": ws,
        "iws": iws,
        "G_imp": G_imp,
        "G_imp_iw": G_imp_iw,
        "A_imp": A_imp,
        "thermal_avgs": thermal_avgs,
    }
