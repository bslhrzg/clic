import numpy as np
import clic_clib as cc

from clic.solve.postprocessing import (
    analyze_spin_and_orbital,
    angular_quantum_number,
    get_1p_angular_momentum_matrices,
)


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
        "Sz": 0.5,
        "L2": 2.0,
        "L": 1.0,
        "Lz": -1.0,
        "J2": 1.75,
        "J": 0.9142135623730951,
        "Jz": -0.5,
    }
    for key, value in expected.items():
        assert np.isclose(stats[key], value)


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

    stats = analyze_spin_and_orbital(
        wf, 3, list(range(6)), to_spherical=rotation
    )

    expected = {"S2": 0.75, "Sz": 0.5, "L2": 2.0, "Lz": -1.0, "Jz": -0.5}
    for key, value in expected.items():
        assert np.isclose(stats[key], value)
