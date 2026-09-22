"""Solve an Anderson impurity model with explicitly specified bath levels."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from ._aim_workflow import _solve_aim_green
from .clicvars import ClicVars


def _assemble_aim(h_imp, U_imp, bath):
    """Validate inputs and assemble the AlphaFirst one-body Hamiltonian."""
    h_imp = np.asarray(h_imp, dtype=np.complex128)
    if (h_imp.ndim != 2 or h_imp.shape[0] != h_imp.shape[1]
            or h_imp.shape[0] == 0 or h_imp.shape[0] % 2):
        raise ValueError("h_imp must be a nonempty square matrix of even dimension.")
    if not np.all(np.isfinite(h_imp)) or not np.allclose(
        h_imp, h_imp.conj().T, rtol=0, atol=1e-12
    ):
        raise ValueError("h_imp must be finite and Hermitian.")
    nimp = len(h_imp)
    U_imp = np.asarray(U_imp, dtype=np.complex128)
    if U_imp.shape != (nimp,) * 4 or not np.all(np.isfinite(U_imp)):
        raise ValueError("U_imp must be finite with shape (N_imp_spin_orbitals,) * 4.")

    if bath is None:
        eps = np.empty(0)
        V = np.empty((nimp, 0), dtype=np.complex128)
    else:
        if not isinstance(bath, Mapping) or set(bath) != {"energies", "V"}:
            raise ValueError("bath must contain exactly 'energies' and 'V'.")
        eps = np.asarray(bath["energies"])
        if eps.ndim != 1 or len(eps) % 2:
            raise ValueError("bath energies must be a vector with paired up/down slots.")
        if not np.all(np.isfinite(eps)) or np.any(np.imag(eps) != 0):
            raise ValueError("bath energies must be finite and real.")
        eps = np.asarray(eps.real, dtype=float)
        V = np.asarray(bath["V"], dtype=np.complex128)
        if V.shape != (nimp, len(eps)) or not np.all(np.isfinite(V)):
            raise ValueError("bath V must be finite with shape (N_imp_spin_orbitals, N_bath_spin_orbitals).")

    mi, mb = nimp // 2, len(eps) // 2
    m = mi + mb
    imp = np.r_[np.arange(mi), np.arange(mi) + m]
    baths = np.r_[np.arange(mi, m), np.arange(mi, m) + m]
    h0 = np.zeros((2 * m, 2 * m), dtype=np.complex128)
    h0[np.ix_(imp, imp)] = h_imp
    h0[np.ix_(baths, baths)] = np.diag(eps)
    h0[np.ix_(imp, baths)] = V
    h0[np.ix_(baths, imp)] = V.conj().T
    return h0, h_imp, U_imp, eps, V


def _mesh(values, name, *, allow_empty=False):
    values = np.asarray(values)
    if (values.ndim != 1 or not np.all(np.isfinite(values))
            or np.any(np.imag(values) != 0)
            or (not allow_empty and values.size == 0)):
        raise ValueError(f"{name} must be a finite real frequency vector.")
    values = np.asarray(values.real, dtype=float)
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return values


def _save_archive(path, result, h_imp, U_imp, eps, V, settings):
    # Exclusive creation protects existing DMFT or model archives.
    with h5py.File(path, "x") as archive:
        archive.attrs["calculation"] = "aimsolve"
        archive.attrs["energy_unit"] = "Ry"
        archive.attrs["temperature_unit"] = "K"
        model = archive.create_group("model")
        model.attrs["basis"] = "AlphaFirst: all up, then all down"
        for name, value in (("h_imp", h_imp), ("U_imp", U_imp),
                            ("bath/energies", eps), ("bath/V", V)):
            model.create_dataset(name, data=value)
        config = archive.create_group("settings")
        for name, value in settings.items():
            if value is None:
                config.attrs[name] = "null"
            elif isinstance(value, Mapping):
                group = config.create_group(name)
                for key, operator in value.items():
                    group.create_dataset(key, data=operator)
            elif isinstance(value, (str, bool, int, float, np.number)):
                config.attrs[name] = value
            else:
                config.create_dataset(name, data=np.asarray(value))
        data = archive.create_group("data")
        for name, key in (("w", "ws"), ("iw", "iws"), ("G_imp", "G_imp"),
                          ("G_imp_iw", "G_imp_iw"), ("A_imp", "A_imp")):
            dataset = data.create_dataset(name, data=result[key])
            if name in ("G_imp", "G_imp_iw"):
                dataset.attrs["basis"] = "CLIC impurity basis (AlphaFirst)"
                dataset.attrs["axis_order"] = "frequency,orbital,orbital"
                dataset.attrs["mesh"] = "/data/iw" if name.endswith("iw") else "/data/w"
        thermal = archive.create_group("thermal_avgs")
        for name, value in result["thermal_avgs"].items():
            if name == "state_stats":
                states = thermal.create_group(name)
                for index, stats in enumerate(value):
                    state = states.create_group(str(index))
                    for key, stat in stats.items():
                        if stat is not None:
                            state.create_dataset(key, data=stat)
            elif value is not None:
                thermal.create_dataset(name, data=value)


def aimsolve(ws, h_imp, U_imp, bath=None, iws=None, rspt_clic_params=None,
             eim=None, impurity_to_spherical=None, impurity_angular_operators=None,
             archive_path=None, *, solver_params=None, input_path="input.toml",
             plot_sf=False):
    """Return impurity Green functions for a bath specified by energies and V.

    ``bath = {"energies": eps, "V": V}`` uses spin-full AlphaFirst ordering
    (all up, then all down) separately for impurity and bath. ``eps`` has
    even length B and ``V`` has shape (len(h_imp), B), with Hamiltonian term
    V[i,b] d_i^dagger a_b + conjugate. Up/down partners may differ. Bath
    orbitals are noninteracting; ``None`` means an isolated impurity.
    U_imp uses exactly the same tensor convention as dmft_step.

    All energies, ws, eim and real Matsubara frequencies iws are in Ry,
    relative to the same chemical potential; temperature is in K. Omitted
    iws returns empty Matsubara arrays. eim overrides configured eta.

    Settings precedence: defaults < RSPT parameters < input_path TOML
    < solver_params < explicit model/mesh arguments. input_path=None skips
    TOML. solver_params accepts ClicVars options. Fit settings (including
    nb, freeze_bath and spin_avg_sigma) do not alter the supplied model.
    The rhf basis preparation is unsupported because it mixes impurity and
    bath targets; use basis_prep_method='none' for the supplied star basis.

    Returns a dict with ws, iws, G_imp, G_imp_iw, A_imp, thermal_avgs.
    G arrays use (frequency, impurity orbital, impurity orbital) ordering;
    A_imp contains diagonal spectral densities. Thermal-state selection and
    pruning follow dmft_step. Existing solver text outputs are retained;
    plotting is optional. archive_path creates a NEW HDF5 archive containing
    inputs, resolved settings and results; existing files are never replaced.
    """
    h0, h_imp, U_imp, eps, V = _assemble_aim(h_imp, U_imp, bath)
    if input_path is None:
        settings = ClicVars()
        settings._apply_overrides(settings._translate_rspt_params(rspt_clic_params),
                                  "rspt_clic_params")
    else:
        settings = ClicVars.from_sources(input_path, rspt_clic_params)
    if solver_params is not None:
        settings._apply_overrides(solver_params, "solver_params")
    if settings.basis_prep_method == "rhf":
        raise ValueError(
            "aimsolve does not support basis_prep_method='rhf': it mixes impurity "
            "and bath orbitals, while Green-function targets must remain in the "
            "input impurity basis. Use basis_prep_method='none'."
        )
    settings.ws = _mesh(ws, "ws")
    settings.iws = _mesh([] if iws is None else iws, "iws", allow_empty=True)
    if eim is not None:
        settings.eta = eim
    if (not np.isscalar(settings.eta) or not np.isreal(settings.eta)
            or not np.isfinite(settings.eta) or np.real(settings.eta) <= 0):
        raise ValueError("eim/eta must be finite, real and positive.")
    settings.eta = float(np.real(settings.eta))
    settings.M_imp = len(h_imp) // 2
    settings.M_spatial = len(h0) // 2
    settings.NF = len(h0)
    settings.is_impurity_model = True
    settings.imp_indices_spatial = list(range(settings.M_imp))
    settings.imp_indices_spinfull = (settings.imp_indices_spatial
        + [i + settings.M_spatial for i in settings.imp_indices_spatial])
    settings.green_block_indices = settings.imp_indices_spinfull.copy()
    settings.impurity_to_spherical = impurity_to_spherical
    settings.impurity_angular_operators = impurity_angular_operators
    if archive_path is not None:
        if Path(archive_path).exists():
            raise FileExistsError(f"Refusing to replace existing archive: {archive_path}")
    # Snapshot before solver routines can mutate their working settings.
    resolved = deepcopy(vars(settings))
    result = _solve_aim_green(h0, U_imp, settings, plot_sf=plot_sf, full_block=True)
    if archive_path is not None:
        _save_archive(archive_path, result, h_imp, U_imp, eps, V, resolved)
    return result
