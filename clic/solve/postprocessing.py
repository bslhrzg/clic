
# postprocessing.py

import numpy as np
from clic.ops import ops


def get_1p_Sz_matrix(M_imp):
    """
    One-particle Sz operator on the impurity spinful space:
    [up block, down block] with eigenvalues +1/2, -1/2.
    """
    sz_diag = np.concatenate([np.full(M_imp, 0.5), np.full(M_imp, -0.5)])
    return np.diag(sz_diag)


def analyze_state(state, clicvars):
    """
    Analyze one state dict of the form
        {"ne": ..., "psi": ..., "e": ..., "bw": ...}

    Returns a dict with a few observables.
    """
    wf = state["psi"]
    nelec = state["ne"]
    M = clicvars.M_spatial

    stats = {}

    _, Sz_full = ops.expect_S2(wf, M)

    if clicvars.is_impurity_model:
        M_imp = clicvars.M_imp
        imp_indices_spatial = list(range(M_imp))
        imp_spinfull = imp_indices_spatial + [i + M for i in imp_indices_spatial]

        rdm_imp = ops.one_rdm(wf, M, block=imp_spinfull)
        Sz_op = get_1p_Sz_matrix(M_imp)

        stats["occ"] = float(np.sum(np.real(np.diag(rdm_imp))))
        stats["rdm"] = rdm_imp
        stats["Sz"] = float(np.real(np.trace(Sz_op @ rdm_imp)))
    else:
        stats["occ"] = float(nelec)
        stats["rdm"] = None
        stats["Sz"] = float(np.real(Sz_full))

    return stats


def analyze_thermal_gs(states, clicvars, save_rdm=True, thr_print=None):
    """
    Analyze a list of retained thermal states.

    Parameters
    ----------
    states : list of dict
        Each state must look like
            {"ne": ..., "psi": ..., "e": ..., "bw": ...}
    clicvars : object
        Must contain at least:
            M_spatial
            is_impurity_model
            M_imp
    save_rdm : bool
        If True, save thermally averaged impurity density matrix.
    thr_print : float or None
        If not None, only print states with bw >= thr_print.

    Returns
    -------
    dict
        Summary with thermal averages and optional impurity thermal rdm.
    """
    if len(states) == 0:
        print("No states to analyze.")
        return {
            "avg_occ": None,
            "avg_Sz": None,
            "rho_imp_thermal": None,
            "state_stats": [],
        }

    states = sorted(states, key=lambda s: s["e"])
    gs_energy = states[0]["e"]

    print("-" * 50)
    print("RETAINED STATES:")
    print("-" * 50)
    print(f"GS: e0 = {gs_energy:.12f}")

    state_stats = []
    avg_occ = 0.0
    avg_Sz = 0.0
    rho_imp_thermal = None

    if clicvars.is_impurity_model:
        M_imp = clicvars.M_imp
        rho_imp_thermal = np.zeros((2 * M_imp, 2 * M_imp), dtype=np.complex128)

    for state in states:
        bw = state["bw"]

        if thr_print is not None and bw < thr_print:
            continue

        stats = analyze_state(state, clicvars)
        state_stats.append(stats)

        avg_occ += bw * stats["occ"]
        avg_Sz += bw * stats["Sz"]

        if clicvars.is_impurity_model:
            rho_imp_thermal += bw * stats["rdm"]

        print(
            f"e-e0: {state['e'] - gs_energy:10.8f}, "
            f"ne: {state['ne']}, "
            f"weight: {bw:10.4f}, "
            f"occ: {stats['occ']:10.4f}, "
            f"Sz: {stats['Sz']:10.4f}"
        )

    if clicvars.is_impurity_model and save_rdm:
        print("Saving thermally-averaged impurity density matrix...")
        np.savetxt("real-imp-dens.dat", np.real(rho_imp_thermal), fmt="% 8.5f")
        np.savetxt("imag-imp-dens.dat", np.imag(rho_imp_thermal), fmt="% 8.5f")
        print("-> Saved 'real-imp-dens.dat'")
        print("-> Saved 'imag-imp-dens.dat'")

    print("thermal averages:")
    print(f"<occ> = {avg_occ:.8f}")
    print(f"<Sz>  = {avg_Sz:.8f}")
    print("-" * 50)

    return {
        "avg_occ": float(avg_occ),
        "avg_Sz": float(avg_Sz),
        "rho_imp_thermal": rho_imp_thermal,
        "state_stats": state_stats,
    }