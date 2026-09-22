import numpy as np
import clic_clib as cc

from clic.solve.postprocessing import (
    analyze_state,
    analyze_spin_and_orbital,
    analyze_thermal_gs,
    angular_quantum_number,
    get_1p_angular_momentum_matrices,
    get_1p_spin_matrices,
)
from clic.clicvars import ClicVars


def test_angular_momentum_matrices():
    for n_orbitals, l in ((1, 0), (3, 1), (5, 2), (7, 3)):
        lx, ly, lz = get_1p_angular_momentum_matrices(n_orbitals)
        expected_ml = np.tile(np.arange(-l, l + 1), 2)

        np.testing.assert_allclose(np.diag(lz), expected_ml)
        assert np.isclose(angular_quantum_number(l * (l + 1)), l)

        for offset in (0, n_orbitals):
            shell = slice(offset, offset + n_orbitals)
            l2 = sum(op[shell, shell] @ op[shell, shell] for op in (lx, ly, lz))
            np.testing.assert_allclose(l2, l * (l + 1) * np.eye(n_orbitals))


def test_single_electron_spin_and_orbital_quantum_numbers():
    # p shell followed by one bath orbital; electron is |ml=-1, ms=+1/2>.
    M = 4
    wf = cc.Wavefunction(M)
    wf.add_term(cc.SlaterDeterminant(M, [0], []), 1.0)
    impurity_block = [0, 1, 2, M, M + 1, M + 2]

    stats = analyze_spin_and_orbital(wf, M, impurity_block)

    expected = {
        "S2": 0.75,
        "S": 0.5,
        "Sx": 0.0,
        "Sy": 0.0,
        "Sz": 0.5,
        "L2": 2.0,
        "L": 1.0,
        "Lx": 0.0,
        "Ly": 0.0,
        "Lz": -1.0,
        "J2": 1.75,
        "J": 0.9142135623730951,
        "Jx": 0.0,
        "Jy": 0.0,
        "Jz": -0.5,
        "LdotS": -0.5,
    }
    for key, value in expected.items():
        assert np.isclose(stats[key], value)


def test_impurity_occupation_operator_variance():
    M = 4
    wf = cc.Wavefunction(M)
    wf.add_term(cc.SlaterDeterminant(M, [0], []), 1 / np.sqrt(2))
    wf.add_term(cc.SlaterDeterminant(M, [3], []), 1 / np.sqrt(2))
    clicvars = ClicVars(M_spatial=M, M_imp=3, is_impurity_model=True)

    state = {"ne": 1, "psi": wf, "e": 0.0, "bw": 1.0}
    stats = analyze_state(state, clicvars)
    thermal = analyze_thermal_gs([state], clicvars, save_rdm=False)

    assert np.isclose(stats["occ"], 0.5)
    assert np.isclose(stats["occ2"], 0.5)
    assert np.isclose(thermal["avg_occ2"], 0.5)
    assert np.isclose(thermal["var_occ"], 0.25)


def test_spin_x_and_y_expectation_values():
    M = 3
    block = list(range(2 * M))

    spin_x = cc.Wavefunction(M)
    spin_x.add_term(cc.SlaterDeterminant(M, [1], []), 1 / np.sqrt(2))
    spin_x.add_term(cc.SlaterDeterminant(M, [], [1]), 1 / np.sqrt(2))
    stats_x = analyze_spin_and_orbital(spin_x, M, block)
    assert np.isclose(stats_x["Sx"], 0.5)
    assert np.isclose(stats_x["Sy"], 0.0)
    assert np.isclose(stats_x["Jx"], 0.5)

    spin_y = cc.Wavefunction(M)
    spin_y.add_term(cc.SlaterDeterminant(M, [1], []), 1 / np.sqrt(2))
    spin_y.add_term(cc.SlaterDeterminant(M, [], [1]), 1j / np.sqrt(2))
    stats_y = analyze_spin_and_orbital(spin_y, M, block)
    assert np.isclose(stats_y["Sx"], 0.0)
    assert np.isclose(abs(stats_y["Sy"]), 0.5)
    assert np.isclose(stats_y["Jy"], stats_y["Sy"])


def test_observables_are_invariant_under_one_particle_basis_rotation():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    rotation, _ = np.linalg.qr(raw)

    # Physical spherical state |ml=-1, ms=+1/2>, represented in the rotated basis.
    solver_coefficients = rotation.conj().T[:, 0]
    wf = cc.Wavefunction(3)
    for index, coefficient in enumerate(solver_coefficients):
        alpha = [index] if index < 3 else []
        beta = [index - 3] if index >= 3 else []
        wf.add_term(cc.SlaterDeterminant(3, alpha, beta), coefficient)

    stats = analyze_spin_and_orbital(wf, 3, list(range(6)), to_spherical=rotation)

    expected = {"S2": 0.75, "Sz": 0.5, "L2": 2.0, "Lz": -1.0, "Jz": -0.5}
    for key, value in expected.items():
        assert np.isclose(stats[key], value)


def test_reversed_rspt_spin_blocks_preserve_f1_j_quantum_number():
    l = 3
    n_orbitals = 2 * l + 1
    dim = 2 * n_orbitals
    spin_swap = np.block(
        [
            [np.zeros((n_orbitals, n_orbitals)), np.eye(n_orbitals)],
            [np.eye(n_orbitals), np.zeros((n_orbitals, n_orbitals))],
        ]
    )

    # |j=5/2, mj=5/2> in canonical (up block, down block) ordering.
    canonical = np.zeros(dim, dtype=np.complex128)
    canonical[2 + l] = np.sqrt(1.0 / 7.0)  # |ml=2, up>
    canonical[n_orbitals + 3 + l] = -np.sqrt(6.0 / 7.0)  # |ml=3, down>
    rspt_order = spin_swap @ canonical

    wf = cc.Wavefunction(n_orbitals)
    for index, coefficient in enumerate(rspt_order):
        if abs(coefficient) < 1e-14:
            continue
        alpha = [index] if index < n_orbitals else []
        beta = [index - n_orbitals] if index >= n_orbitals else []
        wf.add_term(cc.SlaterDeterminant(n_orbitals, alpha, beta), coefficient)

    stats = analyze_spin_and_orbital(
        wf, n_orbitals, list(range(dim)), to_spherical=spin_swap
    )

    assert np.isclose(stats["J"], 2.5)
    assert np.isclose(stats["J2"], 8.75)
    assert np.isclose(stats["Jz"], 2.5)
    assert np.isclose(stats["LdotS"], -2.0)


def test_direct_angular_operators_preserve_global_components_after_rotation():
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    corr_to_solver, _ = np.linalg.qr(raw)

    l_ops = get_1p_angular_momentum_matrices(3)
    s_ops = get_1p_spin_matrices(3)
    angular_operators = {
        "Lz": corr_to_solver.conj().T @ l_ops[2] @ corr_to_solver,
        "Lplus": corr_to_solver.conj().T @ (l_ops[0] + 1j * l_ops[1]) @ corr_to_solver,
        "Sz": corr_to_solver.conj().T @ s_ops[2] @ corr_to_solver,
        "Splus": corr_to_solver.conj().T @ (s_ops[0] + 1j * s_ops[1]) @ corr_to_solver,
    }

    solver_coefficients = corr_to_solver.conj().T[:, 0]
    wf = cc.Wavefunction(3)
    for index, coefficient in enumerate(solver_coefficients):
        alpha = [index] if index < 3 else []
        beta = [index - 3] if index >= 3 else []
        wf.add_term(cc.SlaterDeterminant(3, alpha, beta), coefficient)

    stats = analyze_spin_and_orbital(
        wf, 3, list(range(6)), angular_operators=angular_operators
    )

    expected = {"S2": 0.75, "Sz": 0.5, "L2": 2.0, "Lz": -1.0, "Jz": -0.5}
    for key, value in expected.items():
        assert np.isclose(stats[key], value)
