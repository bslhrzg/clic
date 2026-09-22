
# postprocessing.py

import numpy as np
from clic.ops import ops
from clic.solve.magnetic_dipole import (
    expect_from_rdm,
    get_1p_magnetic_dipole_matrices,
    validate_rdm_observables,
)


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


def _apply_components(wf, M, components, block):
    return tuple(
        ops.apply_one_body_matrix(wf, M, op, block=block) for op in components
    )


def _component_expectations(wf, applied):
    return tuple(float(np.real(wf.dot(phi))) for phi in applied)


def _vector_squared(applied):
    return float(sum(np.real(phi.dot(phi)) for phi in applied))


def _components_from_z_and_plus(z, plus, name, dim):
    z = np.asarray(z, dtype=np.complex128)
    plus = np.asarray(plus, dtype=np.complex128)
    if z.shape != (dim, dim) or plus.shape != (dim, dim):
        raise ValueError(
            f"{name} operators must have shape {(dim, dim)}; "
            f"got {z.shape} and {plus.shape}"
        )
    minus = plus.conj().T
    return 0.5 * (plus + minus), -0.5j * (plus - minus), z


def analyze_spin_and_orbital(
    wf, M, block, to_spherical=None, angular_operators=None, rdm=None,
    reference_occupation=None,
):
    """Analyze a complete shell and validate its RDM before evaluating T.

    Tx/Ty/Tz use Quanty's spin-dipole convention with hbar=1.  An optional
    rdm is the untransposed result of ops.one_rdm in block order; passing it
    avoids recomputing the density.  Occupation and every L/S component must
    agree with direct wavefunction expectations.  Supplied angular_operators
    define the physical axes for both the angular and magnetic dipole outputs.
    """
    n_orbitals = len(block) // 2
    if angular_operators is None:
        l_ops = _transform_components(
            get_1p_angular_momentum_matrices(n_orbitals), to_spherical
        )
        s_ops = _transform_components(get_1p_spin_matrices(n_orbitals), to_spherical)
    else:
        dim = len(block)
        l_ops = _components_from_z_and_plus(
            angular_operators["Lz"], angular_operators["Lplus"], "orbital", dim
        )
        s_ops = _components_from_z_and_plus(
            angular_operators["Sz"], angular_operators["Splus"], "spin", dim
        )
    l_applied = _apply_components(wf, M, l_ops, block)
    s_applied = _apply_components(wf, M, s_ops, block)
    j_applied = tuple(l_phi + s_phi for l_phi, s_phi in zip(l_applied, s_applied))

    L2 = _vector_squared(l_applied)
    S2 = _vector_squared(s_applied)
    J2 = _vector_squared(j_applied)
    Lx, Ly, Lz = _component_expectations(wf, l_applied)
    Sx, Sy, Sz = _component_expectations(wf, s_applied)
    Jx, Jy, Jz = _component_expectations(wf, j_applied)

    stats = {
        "S2": float(S2),
        "S": float(angular_quantum_number(S2)),
        "Sx": float(Sx),
        "Sy": float(Sy),
        "Sz": float(Sz),
        "L2": float(L2),
        "L": float(angular_quantum_number(L2)),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "Lz": float(Lz),
        "J2": float(J2),
        "J": float(angular_quantum_number(J2)),
        "Jx": float(Jx),
        "Jy": float(Jy),
        "Jz": float(Jz),
        "LdotS": float(0.5 * (J2 - L2 - S2)),
    }

    # Check the same density convention against independent wavefunction
    # expectations before using it for the magnetic dipole.
    if rdm is None:
        rdm = ops.one_rdm(wf, M, block=block)
    if reference_occupation is None:
        reference_occupation, _ = ops.expect_one_body_matrix(
            wf, M, np.eye(len(block)), block=block
        )
    density_values = validate_rdm_observables(
        rdm, l_ops, s_ops, {**stats, "occ": reference_occupation}
    )
    if angular_operators is None:
        t_ops = _transform_components(
            get_1p_magnetic_dipole_matrices(n_orbitals), to_spherical
        )
    else:
        t_ops = get_1p_magnetic_dipole_matrices(
            n_orbitals, l_components=l_ops, s_components=s_ops
        )
    stats.update({"T" + axis: expect_from_rdm(rdm, op)
                  for axis, op in zip("xyz", t_ops)})
    stats["rdm_observables_validated"] = True
    stats["mS_z_muB"] = density_values["mS_z_muB"]
    stats["mL_z_muB"] = density_values["mL_z_muB"]
    return stats


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
        occ_direct, occ2 = ops.expect_one_body_matrix(
            wf,
            M,
            np.eye(len(imp_spinfull), dtype=np.complex128),
            block=imp_spinfull,
        )
        stats["occ2"] = float(occ2)
        stats["rdm"] = rdm_imp
        stats.update(
            analyze_spin_and_orbital(
                wf,
                M,
                imp_spinfull,
                to_spherical=getattr(clicvars, "impurity_to_spherical", None),
                angular_operators=getattr(
                    clicvars, "impurity_angular_operators", None
                ),
                rdm=rdm_imp,
                reference_occupation=occ_direct,
            )
        )
    else:
        stats["occ"] = float(nelec)
        stats["occ2"] = float(nelec**2)
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
            "avg_occ2": None,
            "var_occ": None,
            "avg_Sx": None,
            "avg_Sy": None,
            "avg_Sz": None,
            "avg_Tx": None,
            "avg_Ty": None,
            "avg_Tz": None,
            "avg_S": None,
            "avg_S2": None,
            "avg_L": None,
            "avg_L2": None,
            "avg_Lx": None,
            "avg_Ly": None,
            "avg_Lz": None,
            "avg_J": None,
            "avg_J2": None,
            "avg_Jx": None,
            "avg_Jy": None,
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
    avg_occ2 = 0.0
    avg_Sx = 0.0
    avg_Sy = 0.0
    avg_Sz = 0.0
    avg_Tx = 0.0
    avg_Ty = 0.0
    avg_Tz = 0.0
    avg_S2 = 0.0
    avg_L2 = 0.0
    avg_Lx = 0.0
    avg_Ly = 0.0
    avg_Lz = 0.0
    avg_J2 = 0.0
    avg_Jx = 0.0
    avg_Jy = 0.0
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
        avg_occ2 += bw * stats["occ2"]
        avg_Sx += bw * stats["Sx"]
        avg_Sy += bw * stats["Sy"]
        avg_Sz += bw * stats["Sz"]
        avg_Tx += bw * stats["Tx"]
        avg_Ty += bw * stats["Ty"]
        avg_Tz += bw * stats["Tz"]
        avg_S2 += bw * stats["S2"]
        avg_L2 += bw * stats["L2"]
        avg_Lx += bw * stats["Lx"]
        avg_Ly += bw * stats["Ly"]
        avg_Lz += bw * stats["Lz"]
        avg_J2 += bw * stats["J2"]
        avg_Jx += bw * stats["Jx"]
        avg_Jy += bw * stats["Jy"]
        avg_Jz += bw * stats["Jz"]

        if clicvars.is_impurity_model:
            rho_imp_thermal += bw * stats["rdm"]

        print(
            f"e-e0: {state['e'] - gs_energy:10.8f}, "
            f"ne: {state['ne']}, "
            f"weight: {bw:10.4f}, "
            f"occ: {stats['occ']:10.4f}, "
            f"S: {stats['S']:10.4f}, "
            f"Sx: {stats['Sx']:10.4f}, "
            f"Sy: {stats['Sy']:10.4f}, "
            f"Sz: {stats['Sz']:10.4f}, "
            f"Tx: {stats['Tx']:10.4f}, "
            f"Ty: {stats['Ty']:10.4f}, "
            f"Tz: {stats['Tz']:10.4f}, "
            f"L: {stats['L']:10.4f}, "
            f"Lx: {stats['Lx']:10.4f}, "
            f"Ly: {stats['Ly']:10.4f}, "
            f"Lz: {stats['Lz']:10.4f}, "
            f"Jx: {stats['Jx']:10.4f}, "
            f"Jy: {stats['Jy']:10.4f}, "
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
    var_occ = avg_occ2 - avg_occ**2
    print(f"<occ> = {avg_occ:.8f}")
    print(f"<Sx>  = {avg_Sx:.8f}")
    print(f"<Sy>  = {avg_Sy:.8f}")
    print(f"<Sz>  = {avg_Sz:.8f}")
    print(f"<Tx>  = {avg_Tx:.8f}")
    print(f"<Ty>  = {avg_Ty:.8f}")
    print(f"<Tz>  = {avg_Tz:.8f}")
    print(f"S from <S^2> = {avg_S:.8f}")
    print(f"L from <L^2> = {avg_L:.8f}")
    print(f"<Lx>  = {avg_Lx:.8f}")
    print(f"<Ly>  = {avg_Ly:.8f}")
    print(f"<Lz>  = {avg_Lz:.8f}")
    print(f"<Jx>  = {avg_Jx:.8f}")
    print(f"<Jy>  = {avg_Jy:.8f}")
    print(f"<Jz>  = {avg_Jz:.8f}")
    print(f"J_eff from <J^2> = {avg_J:.8f}")
    print("-" * 50)

    return {
        "avg_occ": float(avg_occ),
        "avg_occ2": float(avg_occ2),
        "var_occ": float(var_occ),
        "avg_Sx": float(avg_Sx),
        "avg_Sy": float(avg_Sy),
        "avg_Sz": float(avg_Sz),
        "avg_Tx": float(avg_Tx),
        "avg_Ty": float(avg_Ty),
        "avg_Tz": float(avg_Tz),
        "avg_S": float(avg_S),
        "avg_S2": float(avg_S2),
        "avg_L": float(avg_L),
        "avg_L2": float(avg_L2),
        "avg_Lx": float(avg_Lx),
        "avg_Ly": float(avg_Ly),
        "avg_Lz": float(avg_Lz),
        "avg_J": float(avg_J),
        "avg_J2": float(avg_J2),
        "avg_Jx": float(avg_Jx),
        "avg_Jy": float(avg_Jy),
        "avg_Jz": float(avg_Jz),
        "rho_imp_thermal": rho_imp_thermal,
        "state_stats": state_stats,
    }
