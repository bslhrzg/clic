"""Independent angular integrals and CLIC density/basis-convention checks."""

import numpy as np
import pytest
import clic_clib as cc
from scipy.special import sph_harm_y

from clic.clicvars import ClicVars
from clic.ops import ops
from clic.solve.magnetic_dipole import (
    expect_from_rdm,
    gaunt_quadrupole_matrices,
    get_1p_magnetic_dipole_matrices,
    validate_rdm_observables,
)
from clic.solve.postprocessing import (
    analyze_spin_and_orbital,
    analyze_state,
    analyze_thermal_gs,
    get_1p_angular_momentum_matrices,
    get_1p_spin_matrices,
)


def angular_integral(l):
    # Direct integration of delta_ab - 3 n_a n_b, independent of Gaunt
    # coefficients and the equivalent-operator construction using L.
    z, weights = np.polynomial.legendre.leggauss(24)
    phi = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    theta, azimuth = np.meshgrid(np.arccos(z), phi, indexing="ij")
    direction = np.array([
        np.sin(theta) * np.cos(azimuth),
        np.sin(theta) * np.sin(azimuth),
        np.cos(theta),
    ])
    harmonics = np.array([sph_harm_y(l, m, theta, azimuth) for m in range(-l, l + 1)])
    tensor = np.eye(3)[:, :, None, None] - 3 * np.einsum(
        "aij,bij->abij", direction, direction
    )
    return np.einsum(
        "mij,abij,nij,i->abmn", harmonics.conj(), tensor, harmonics, weights
    ) * (2 * np.pi / len(phi))


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_gaunt_tensor_against_real_space_angular_integral(l):
    q = gaunt_quadrupole_matrices(l)
    np.testing.assert_allclose(q, angular_integral(l), atol=2e-14)
    np.testing.assert_allclose(q, q.conj().swapaxes(-1, -2), atol=1e-14)
    np.testing.assert_allclose(sum(q[a, a] for a in range(3)), 0, atol=1e-14)


@pytest.mark.parametrize("n", [1, 3, 5, 7])
def test_gaunt_and_equivalent_operator_agree_after_complex_basis_rotation(n):
    rng = np.random.default_rng(n)
    rotation, _ = np.linalg.qr(rng.normal(size=(2*n, 2*n)) + 1j*rng.normal(size=(2*n, 2*n)))
    l_ops = tuple(rotation.conj().T @ op @ rotation
                  for op in get_1p_angular_momentum_matrices(n))
    s_ops = tuple(rotation.conj().T @ op @ rotation
                  for op in get_1p_spin_matrices(n))
    actual = get_1p_magnetic_dipole_matrices(n, l_components=l_ops, s_components=s_ops)
    for a, canonical in zip(actual, get_1p_magnetic_dipole_matrices(n)):
        np.testing.assert_allclose(a, rotation.conj().T @ canonical @ rotation, atol=1e-14)
        np.testing.assert_allclose(a, a.conj().T, atol=1e-14)


def test_dz2_spin_along_each_axis_and_isotropic_spin_density():
    n = 5
    t_ops = get_1p_magnetic_dipole_matrices(n)
    spinors = (np.array([1, 1])/np.sqrt(2), np.array([1, 1j])/np.sqrt(2), np.array([1, 0]))
    orbital = np.eye(n)[:, 2]  # m=0, i.e. dz2
    for axis, spinor in enumerate(spinors):
        psi = np.kron(spinor, orbital)
        rdm = np.outer(psi, psi.conj())
        expected = np.zeros(3)
        expected[axis] = (1/7, 1/7, -2/7)[axis]
        np.testing.assert_allclose([expect_from_rdm(rdm, t) for t in t_ops], expected, atol=1e-14)
    # Fully spin-polarized but orbitally isotropic density has T=0.
    rdm = np.diag([1/n]*n + [0]*n)
    np.testing.assert_allclose([expect_from_rdm(rdm, t) for t in t_ops], 0, atol=1e-14)


def impurity_wavefunction(coefficients, occupation):
    n = len(coefficients) // 2
    M = n + 1  # extra bath orbital permits fractional impurity occupation
    wf = cc.Wavefunction(M)
    for i, value in enumerate(coefficients):
        wf.add_term(cc.SlaterDeterminant(M, [i] if i < n else [],
                                        [i-n] if i >= n else []),
                    np.sqrt(occupation) * value)
    wf.add_term(cc.SlaterDeterminant(M, [n], []), np.sqrt(1 - occupation))
    return wf


@pytest.mark.parametrize("direct_operators", [False, True])
def test_f_shell_density_moments_and_T_in_rotated_solver_basis(direct_operators):
    n = 7
    rng = np.random.default_rng(45)
    rotation, _ = np.linalg.qr(rng.normal(size=(14, 14)) + 1j*rng.normal(size=(14, 14)))
    # General coherent spin-orbital state exercises imaginary entries, spin
    # flips and all Cartesian components; occupation is deliberately fractional.
    psi = rng.normal(size=14) + 1j*rng.normal(size=14)
    psi /= np.linalg.norm(psi)
    occupation = 0.63
    wf = impurity_wavefunction(rotation.conj().T @ psi, occupation)
    canonical_l = get_1p_angular_momentum_matrices(n)
    canonical_s = get_1p_spin_matrices(n)
    l_ops = tuple(rotation.conj().T @ op @ rotation for op in canonical_l)
    s_ops = tuple(rotation.conj().T @ op @ rotation for op in canonical_s)
    kwargs = {"impurity_to_spherical": rotation}
    if direct_operators:
        kwargs = {"impurity_angular_operators": {
            "Lz": l_ops[2], "Lplus": l_ops[0] + 1j*l_ops[1],
            "Sz": s_ops[2], "Splus": s_ops[0] + 1j*s_ops[1],
        }}
    cv = ClicVars(M_spatial=n+1, M_imp=n, is_impurity_model=True, **kwargs)
    stats = analyze_state({"psi": wf, "ne": 1, "e": 0, "bw": 1}, cv)
    gamma = stats["rdm"]

    # Check occupation and both moment conventions before checking T.
    assert stats["rdm_observables_validated"]
    assert np.isclose(np.trace(gamma), occupation)
    assert np.isclose(-2*np.trace(gamma @ s_ops[2]), -2*stats["Sz"])
    assert np.isclose(-np.trace(gamma @ l_ops[2]), -stats["Lz"])
    assert np.isclose(stats["mS_z_muB"], -2*stats["Sz"])
    assert np.isclose(stats["mL_z_muB"], -stats["Lz"])
    for label, components in (("S", canonical_s), ("L", canonical_l)):
        for axis, op in zip("xyz", components):
            assert np.isclose(stats[label+axis], occupation*np.vdot(psi, op @ psi))

    q = angular_integral(3)
    spin = get_1p_spin_matrices(1)
    block = list(range(n)) + list(range(n+1, 2*n+1))
    for a, axis in enumerate("xyz"):
        t = sum(np.kron(spin[b], q[a, b]) for b in range(3))
        expected = occupation*np.vdot(psi, t @ psi)
        assert np.isclose(stats["T"+axis], expected)
        t_solver = rotation.conj().T @ t @ rotation
        direct, _ = ops.expect_one_body_matrix(wf, n+1, t_solver, block=block)
        assert np.isclose(stats["T"+axis], direct)


def test_wrong_density_transpose_or_reference_fails_validation():
    n = 3
    l_ops = get_1p_angular_momentum_matrices(n)
    s_ops = get_1p_spin_matrices(n)
    psi = np.zeros(2*n, dtype=complex)
    psi[1], psi[n+1] = 1/np.sqrt(2), 1j/np.sqrt(2)
    wf = impurity_wavefunction(psi, 0.6)
    block = [0, 1, 2, 4, 5, 6]
    stats = analyze_spin_and_orbital(wf, 4, block)
    rdm = ops.one_rdm(wf, 4, block)
    reference = {**stats, "occ": 0.6}
    validate_rdm_observables(rdm, l_ops, s_ops, reference)
    with pytest.raises(ValueError, match="Sy"):
        validate_rdm_observables(rdm.T, l_ops, s_ops, reference)
    with pytest.raises(ValueError, match="occ"):
        validate_rdm_observables(rdm, l_ops, s_ops, {**reference, "occ": 1})
    with pytest.raises(ValueError, match="Sy"):
        analyze_spin_and_orbital(wf, 4, block, rdm=rdm.T)


def test_f1_j5_2_state_with_reversed_spin_blocks():
    # Analytic |j=5/2,mj=5/2> includes a spin-flip contribution to Tz.
    # Its diagonal-density contribution alone is -2/7, but full Tz is -4/7.
    n = 7
    canonical = np.zeros(14, dtype=complex)
    canonical[5] = np.sqrt(1/7)
    canonical[13] = -np.sqrt(6/7)
    spin_swap = np.kron(np.array([[0, 1], [1, 0]]), np.eye(n))
    wf = impurity_wavefunction(spin_swap @ canonical, 1)
    cv = ClicVars(M_spatial=8, M_imp=7, is_impurity_model=True,
                  impurity_to_spherical=spin_swap)
    stats = analyze_state({"psi": wf, "ne": 1, "e": 0, "bw": 1}, cv)
    assert np.isclose(stats["Sz"], -5/14)
    assert np.isclose(stats["Lz"], 20/7)
    np.testing.assert_allclose([stats["Tx"], stats["Ty"], stats["Tz"]],
                               [0, 0, -4/7], atol=1e-14)


def test_correlated_two_electron_state_uses_full_one_body_density():
    M = 4
    wf = cc.Wavefunction(M)
    configurations = (([0, 1], []), ([0], [1]), ([2], [3]), ([], [0, 1]))
    coefficients = np.array([1, 2j, -0.7, 1+1j])
    coefficients /= np.linalg.norm(coefficients)
    for (alpha, beta), coefficient in zip(configurations, coefficients):
        wf.add_term(cc.SlaterDeterminant(M, alpha, beta), coefficient)
    cv = ClicVars(M_spatial=M, M_imp=3, is_impurity_model=True)
    stats = analyze_state({"psi": wf, "ne": 2, "e": 0, "bw": 1}, cv)
    block = [0, 1, 2, 4, 5, 6]
    for axis, t in zip("xyz", get_1p_magnetic_dipole_matrices(3)):
        direct, _ = ops.expect_one_body_matrix(wf, M, t, block=block)
        assert np.isclose(stats["T"+axis], direct)
    assert stats["rdm_observables_validated"]


def test_thermal_dipole_uses_same_density_and_weights():
    cv = ClicVars(M_spatial=6, M_imp=5, is_impurity_model=True)
    psi1 = np.eye(10)[:, 2].astype(complex)
    psi2 = (np.eye(10)[:, 1] + 1j*np.eye(10)[:, 7])/np.sqrt(2)
    states = [
        {"psi": impurity_wavefunction(psi1, 0.7), "ne": 1, "e": 0, "bw": 0.3},
        {"psi": impurity_wavefunction(psi2, 0.4), "ne": 1, "e": 1, "bw": 0.7},
    ]
    result = analyze_thermal_gs(states, cv, save_rdm=False)
    for axis, t in zip("xyz", get_1p_magnetic_dipole_matrices(5)):
        assert np.isclose(result["avg_T"+axis], expect_from_rdm(result["rho_imp_thermal"], t))
    empty = analyze_thermal_gs([], cv, save_rdm=False)
    assert all(empty["avg_T"+axis] is None for axis in "xyz")


def test_reject_projected_shell_and_invalid_matrices():
    # Truncating a full d shell to three orbitals is not a p shell.
    indices = [0, 1, 2, 5, 6, 7]
    l_ops = tuple(op[np.ix_(indices, indices)] for op in get_1p_angular_momentum_matrices(5))
    s_ops = get_1p_spin_matrices(3)
    with pytest.raises(ValueError, match="complete single-l shell"):
        get_1p_magnetic_dipole_matrices(3, l_components=l_ops, s_components=s_ops)
    with pytest.raises(ValueError, match="complete shell"):
        get_1p_magnetic_dipole_matrices(4)
    with pytest.raises(ValueError, match="Hermitian"):
        expect_from_rdm(np.array([[0, 1j], [0, 0]]), np.eye(2))
