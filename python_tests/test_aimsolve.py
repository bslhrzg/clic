"""Small AIM checks against analytic Green functions (requires clic_clib)."""
import h5py
import numpy as np
import pytest

from clic import aimsolve
from clic.aimsolve import _assemble_aim


@pytest.fixture
def options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return dict(input_path=None, eim=0.025, solver_params={
        "ci_type": "fci", "basis_prep_method": "none", "nelec_range": 2,
        "num_roots": 1, "NappH": 8, "L_lanczos": 40,
        "coeff_thresh": 1e-12, "temperature": 5.,
    })


def test_shared_complex_bath_resolvent_and_archive(options, tmp_path):
    h = np.diag([-0.1, -0.1])
    u = np.zeros((2,) * 4)
    bath = {"energies": [-0.3, 0.4],
            "V": [[0.12, 0.03j], [0.05j, 0.18]]}
    ws = np.linspace(-1, 1, 101)
    iw = np.array([0.1, 0.3, 0.9])
    path = tmp_path / "aim.h5"
    options["solver_params"].update(nb=0, spin_avg_sigma=True, freeze_bath=True)
    result = aimsolve(ws, h, u, bath, iws=iw, archive_path=path, **options)
    # Independently construct impurity-first Hamiltonian for the resolvent.
    v = np.asarray(bath["V"])
    matrix = np.block([[h, v], [v.conj().T, np.diag(bath["energies"])]])
    for z, key in ((ws + 0.025j, "G_imp"), (1j * iw, "G_imp_iw")):
        expected = np.linalg.inv(z[:, None, None] * np.eye(4) - matrix)[:, :2, :2]
        np.testing.assert_allclose(result[key], expected, atol=1e-8)
    with h5py.File(path) as archive:
        np.testing.assert_array_equal(archive["model/bath/V"], bath["V"])
        np.testing.assert_array_equal(archive["data/G_imp"], result["G_imp"])
        assert archive["settings"].attrs["eta"] == 0.025
        assert "Sigma" not in archive["data"]
    with pytest.raises(FileExistsError):
        aimsolve(ws, h, u, bath, archive_path=path, **options)


def test_atomic_hubbard_green_without_matsubara(options):
    interaction = 0.4
    h = -interaction / 2 * np.eye(2)
    u = np.zeros((2,) * 4)
    u[0, 1, 0, 1] = u[1, 0, 1, 0] = interaction
    options["solver_params"].update(nelec_range=1, num_roots=2)
    ws = np.linspace(-1, 1, 101)
    result = aimsolve(ws, h, u, **options)
    z = ws + 0.025j
    expected = 0.5 / (z + interaction / 2) + 0.5 / (z - interaction / 2)
    np.testing.assert_allclose(result["G_imp"][:, 0, 0], expected, atol=1e-8)
    np.testing.assert_allclose(result["G_imp"][:, 1, 1], expected, atol=1e-8)
    assert result["G_imp_iw"].shape == (0, 2, 2)
    assert result["thermal_avgs"]["avg_occ"] == pytest.approx(1.)


@pytest.mark.parametrize("bath", [
    {"energies": [0.1], "V": [[0.2], [0.2]]},
    {"energies": [0.1j, 0.1], "V": np.eye(2)},
    {"energies": [0.1, 0.2], "V": [[0.2, 0.2]]},
    {"energies": [0.1, np.nan], "V": np.eye(2)},
])
def test_invalid_bath(bath):
    with pytest.raises(ValueError):
        _assemble_aim(np.eye(2), np.zeros((2,) * 4), bath)


def test_alpha_first_placement():
    h = np.diag([1, 2, 3, 4])
    v = np.arange(8).reshape(4, 2) + 1j
    h0, *_ = _assemble_aim(h, np.zeros((4,) * 4),
                           {"energies": [5, 6], "V": v})
    imp, bath = [0, 1, 3, 4], [2, 5]
    np.testing.assert_array_equal(h0[np.ix_(imp, imp)], h)
    np.testing.assert_array_equal(h0[np.ix_(imp, bath)], v)
    np.testing.assert_array_equal(h0, h0.conj().T)


def test_dmft_atomic_self_energy_after_shared_workflow(options, tmp_path):
    from clic import dmft_step
    (tmp_path / "input.toml").write_text(
        'nb = 0\nci_type = "fci"\nbasis_prep_method = "none"\n'
        'nelec_range = 1\nnum_roots = 2\nNappH = 8\n'
        'eta = 0.025\ntemperature = 5.0\n'
    )
    interaction = 0.4
    ws, iw = np.linspace(-1, 1, 101), np.array([0.1, 0.3, 0.9])
    h = -interaction / 2 * np.eye(2)
    u = np.zeros((2,) * 4)
    u[0, 1, 0, 1] = u[1, 0, 1, 0] = interaction
    static, sigma, sigma_iw = dmft_step(ws, iw, np.zeros((len(ws), 2, 2)), h, u)
    np.testing.assert_allclose(static, 0.2 * np.eye(2), atol=1e-8)
    for z, values in ((ws + 0.025j, sigma), (1j * iw, sigma_iw)):
        expected = (interaction / 2 + (interaction / 2)**2 / z)[:, None, None] * np.eye(2)
        np.testing.assert_allclose(values, expected, atol=1e-8)


@pytest.mark.parametrize("overrides, message", [
    ({"eim": 0.}, "positive"),
    ({"solver_params": {"basis_prep_method": "rhf"}}, "input impurity basis"),
])
def test_invalid_solver_settings(overrides, message):
    with pytest.raises(ValueError, match=message):
        aimsolve([-1., 0., 1.], np.eye(2), np.zeros((2,) * 4),
                 input_path=None, **overrides)
