
# postprocessing.py

import numpy as np
from clic.ops import ops


def get_1p_angular_momentum_matrices(n_orbitals):
    """Return Lx, Ly, and Lz in the spherical AlphaFirst basis."""
    l = (n_orbitals - 1) / 2
    if not l.is_integer():
        raise ValueError(
            f"Angular-momentum analysis requires M_shell = 2*l+1; got {n_orbitals} orbitals"
        )
    l = int(l)
    ml = np.arange(-l, l + 1, dtype=float)
    lz_orb = np.diag(ml)
    lp_orb = np.zeros((n_orbitals, n_orbitals), dtype=np.complex128)
    for j, m in enumerate(ml[:-1]):
        lp_orb[j + 1, j] = np.sqrt(l * (l + 1) - m * (m + 1))
    lm_orb = lp_orb.T.conj()

    zero = np.zeros_like(lz_orb)
    lz = np.block([[lz_orb, zero], [zero, lz_orb]])
    lp = np.block([[lp_orb, zero], [zero, lp_orb]])
    lm = np.block([[lm_orb, zero], [zero, lm_orb]])
    return 0.5 * (lp + lm), -0.5j * (lp - lm), lz


def get_1p_spin_matrices(n_orbitals):
    """Return Sx, Sy, and Sz in the spherical AlphaFirst basis."""
    eye = np.eye(n_orbitals, dtype=np.complex128)
    zero = np.zeros_like(eye)
    return (
        0.5 * np.block([[zero, eye], [eye, zero]]),
        0.5 * np.block([[zero, -1j * eye], [1j * eye, zero]]),
        0.5 * np.block([[eye, zero], [zero, -eye]]),
    )


def angular_quantum_number(moment_squared):
    """Convert <J^2> to the corresponding effective J value."""
    return 0.5 * (np.sqrt(max(0.0, 1.0 + 4.0 * moment_squared)) - 1.0)


def _transform_components(components, to_spherical):
    if to_spherical is None:
        return components
    rotation = np.asarray(to_spherical, dtype=np.complex128)
    dim = components[0].shape[0]
    if rotation.shape != (dim, dim):
        raise ValueError(
            f"impurity_to_spherical must have shape {(dim, dim)}, got {rotation.shape}"
        )
    return tuple(rotation.conj().T @ op @ rotation for op in components)


def _expect_vector_squared(wf, M, components, block):
    applied = [ops.apply_one_body_matrix(wf, M, op, block=block) for op in components]
    return float(sum(np.real(phi.dot(phi)) for phi in applied))


def analyze_spin_and_orbital(wf, M, block, to_spherical=None):
    n_orbitals = len(block) // 2
    l_ops = _transform_components(
        get_1p_angular_momentum_matrices(n_orbitals), to_spherical
    )
    s_ops = _transform_components(get_1p_spin_matrices(n_orbitals), to_spherical)
    j_ops = tuple(l_op + s_op for l_op, s_op in zip(l_ops, s_ops))

    L2 = _expect_vector_squared(wf, M, l_ops, block)
    S2 = _expect_vector_squared(wf, M, s_ops, block)
    J2 = _expect_vector_squared(wf, M, j_ops, block)
    Lz, _ = ops.expect_one_body_matrix(wf, M, l_ops[2], block=block)
    Sz, _ = ops.expect_one_body_matrix(wf, M, s_ops[2], block=block)
    Jz, _ = ops.expect_one_body_matrix(wf, M, j_ops[2], block=block)

    return {
        "S2": float(S2),
        "S": float(angular_quantum_number(S2)),
        "Sz": float(Sz),
        "L2": float(L2),
        "L": float(angular_quantum_number(L2)),
        "Lz": float(Lz),
        "J2": float(J2),
        "J": float(angular_quantum_number(J2)),
        "Jz": float(Jz),
        "LdotS": float(0.5 * (J2 - L2 - S2)),
    }


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

    if clicvars.is_impurity_model:
        M_imp = clicvars.M_imp
        imp_indices_spatial = list(range(M_imp))
        imp_spinfull = imp_indices_spatial + [i + M for i in imp_indices_spatial]

        rdm_imp = ops.one_rdm(wf, M, block=imp_spinfull)
        stats["occ"] = float(np.sum(np.real(np.diag(rdm_imp))))
        stats["rdm"] = rdm_imp
        stats.update(
            analyze_spin_and_orbital(
                wf,
                M,
                imp_spinfull,
                to_spherical=getattr(clicvars, "impurity_to_spherical", None),
            )
        )
    else:
        stats["occ"] = float(nelec)
        stats["rdm"] = None
        stats.update(analyze_spin_and_orbital(wf, M, list(range(2 * M))))

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
            "avg_S": None,
            "avg_S2": None,
            "avg_L": None,
            "avg_L2": None,
            "avg_Lz": None,
            "avg_J": None,
            "avg_J2": None,
            "avg_Jz": None,
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
    avg_S2 = 0.0
    avg_L2 = 0.0
    avg_Lz = 0.0
    avg_J2 = 0.0
    avg_Jz = 0.0
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
        avg_S2 += bw * stats["S2"]
        avg_L2 += bw * stats["L2"]
        avg_Lz += bw * stats["Lz"]
        avg_J2 += bw * stats["J2"]
        avg_Jz += bw * stats["Jz"]

        if clicvars.is_impurity_model:
            rho_imp_thermal += bw * stats["rdm"]

        print(
            f"e-e0: {state['e'] - gs_energy:10.8f}, "
            f"ne: {state['ne']}, "
            f"weight: {bw:10.4f}, "
            f"occ: {stats['occ']:10.4f}, "
            f"S: {stats['S']:10.4f}, "
            f"Sz: {stats['Sz']:10.4f}, "
            f"L: {stats['L']:10.4f}, "
            f"Lz: {stats['Lz']:10.4f}, "
            f"Jz: {stats['Jz']:10.4f}, "
            f"J_eff: {stats['J']:10.4f}, "
            f"<J2>: {stats['J2']:10.4f}, "
            f"<L.S>: {stats['LdotS']:10.4f}"
        )

    print("-" * 50)
    if clicvars.is_impurity_model:
        for i in range(clicvars.M_imp * 2):
            print(f"n_imp({i}) = {np.round(rho_imp_thermal[i, i].real, 4)}")

    if clicvars.is_impurity_model and save_rdm:
        print("Saving thermally-averaged impurity density matrix...")
        np.savetxt("real-imp-dens.dat", np.real(rho_imp_thermal), fmt="% 8.5f")
        np.savetxt("imag-imp-dens.dat", np.imag(rho_imp_thermal), fmt="% 8.5f")
        print("-> Saved 'real-imp-dens.dat'")
        print("-> Saved 'imag-imp-dens.dat'")

    print("thermal averages:")
    avg_S = angular_quantum_number(avg_S2)
    avg_L = angular_quantum_number(avg_L2)
    avg_J = angular_quantum_number(avg_J2)
    print(f"<occ> = {avg_occ:.8f}")
    print(f"<Sz>  = {avg_Sz:.8f}")
    print(f"S from <S^2> = {avg_S:.8f}")
    print(f"L from <L^2> = {avg_L:.8f}")
    print(f"<Lz>  = {avg_Lz:.8f}")
    print(f"<Jz>  = {avg_Jz:.8f}")
    print(f"J_eff from <J^2> = {avg_J:.8f}")
    print("-" * 50)

    return {
        "avg_occ": float(avg_occ),
        "avg_Sz": float(avg_Sz),
        "avg_S": float(avg_S),
        "avg_S2": float(avg_S2),
        "avg_L": float(avg_L),
        "avg_L2": float(avg_L2),
        "avg_Lz": float(avg_Lz),
        "avg_J": float(avg_J),
        "avg_J2": float(avg_J2),
        "avg_Jz": float(avg_Jz),
        "rho_imp_thermal": rho_imp_thermal,
        "state_stats": state_stats,
    }
