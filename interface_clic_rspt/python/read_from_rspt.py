"""Load an RSPT debug archive and reconstruct the corresponding CLIC call."""

from ast import literal_eval
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from clic.basis.basis_1p import basis_change_U, basis_change_h0


_METADATA_TYPES = {
    "label": str,
    "solver_param": str,
    "dc_param": str,
    "dc_flag": int,
    "n_orb": int,
    "n_rot": int,
    "n_orb_full": int,
    "n_iw": int,
    "n_w": int,
    "eim": float,
    "tau": float,
    "verbosity": int,
}

_DATASETS = (
    "U_mat",
    "hyb",
    "h_dft",
    "sig",
    "sig_real",
    "sig_static",
    "sig_dc",
    "iw",
    "w",
    "corr_to_spherical",
    "corr_to_cf",
    "observable_Lz",
    "observable_Lplus",
    "observable_Sz",
    "observable_Splus",
)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_metadata(attributes):
    metadata = {}
    missing = [name for name in _METADATA_TYPES if name not in attributes]
    if missing:
        raise ValueError("Archive is missing metadata: " + ", ".join(missing))

    for name, value_type in _METADATA_TYPES.items():
        value = _decode(attributes[name])
        metadata[name] = value_type(value)
    return metadata


def _read_datasets(group):
    missing = [name for name in _DATASETS if name not in group]
    if missing:
        raise ValueError("Archive is missing datasets: " + ", ".join(missing))
    names = _DATASETS + tuple(
        name for name in ("hyb_fit", "hyb_fit_iw") if name in group
    )
    return {name: np.asarray(group[name][()]) for name in names}


def _parse_solver_parameters(value):
    """Parse the list representation written by dummy_python_solver."""
    value = value.strip()
    try:
        parsed = literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value.split()

    if isinstance(parsed, str):
        parsed = parsed.split()
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Invalid solver_param metadata: {value!r}")

    parameters = [str(item) for item in parsed]
    if len(parameters) < 2:
        raise ValueError(
            "solver_param must contain at least n_bath_poles and Nelec_imp"
        )
    return parameters


def _make_clic_params(label, solver_parameters):
    return {
        "label": label,
        "n_bath_poles": int(solver_parameters[0]),
        "Nelec_imp": int(solver_parameters[1]),
        "num_roots": int(solver_parameters[2]) if len(solver_parameters) > 2 else 4,
        "temperature": (
            float(solver_parameters[3]) if len(solver_parameters) > 3 else 5.0
        ),
        "NappH": int(solver_parameters[4]) if len(solver_parameters) > 4 else 1,
        "conv_tol": (
            float(solver_parameters[5]) if len(solver_parameters) > 5 else 5e-4
        ),
        "Nmul": float(solver_parameters[6]) if len(solver_parameters) > 6 else None,
        "lanczos_thr": (
            float(solver_parameters[7]) if len(solver_parameters) > 7 else 1e-5
        ),
    }


def _complete_corr_to_cf(corr_to_cf, n_orb, n_rot):
    if corr_to_cf.shape != (n_orb, n_rot):
        raise ValueError(
            f"corr_to_cf has shape {corr_to_cf.shape}, expected {(n_orb, n_rot)}"
        )
    if n_rot == n_orb:
        return np.ascontiguousarray(corr_to_cf)

    completed = np.empty((n_orb, n_orb), dtype=np.complex128)
    completed[:, :n_rot] = corr_to_cf
    completed[:, n_rot:] = np.roll(corr_to_cf, n_rot, axis=0)
    return completed


def _validate_shapes(data):
    n_orb = data["n_orb"]
    expected = {
        "U_mat": (n_orb,) * 4,
        "hyb": (n_orb, n_orb, data["n_hyb"]),
        "h_dft": (n_orb, n_orb),
        "sig": (n_orb, n_orb, data["n_iw"]),
        "sig_real": (n_orb, n_orb, data["n_w"]),
        "sig_static": (n_orb, n_orb),
        "sig_dc": (n_orb, n_orb),
        "iw": (data["n_iw"],),
        "w": (data["n_w"],),
    }
    for name in ("observable_Lz", "observable_Lplus", "observable_Sz", "observable_Splus"):
        expected[name] = (n_orb, n_orb)

    mismatches = [
        f"{name}: got {data[name].shape}, expected {shape}"
        for name, shape in expected.items()
        if data[name].shape != shape
    ]
    if mismatches:
        raise ValueError("Invalid archive shapes: " + "; ".join(mismatches))


def load_solver_data(filename):
    """Load raw RSPT data and reconstruct the arguments passed to ``dmft_step``.

    The returned namespace contains every metadata attribute and dataset from the
    archive.  It additionally provides CLIC-basis inputs and ``dmft_args`` /
    ``dmft_kwargs`` for directly repeating the solver call.
    """
    filename = Path(filename)
    with h5py.File(filename, "r") as archive:
        if "metadata" not in archive or "data" not in archive:
            raise ValueError("Not a dummy_python_solver debug archive")
        loaded = _read_metadata(archive["metadata"].attrs)
        loaded.update(_read_datasets(archive["data"]))

    loaded["filename"] = filename
    loaded["n_hyb"] = loaded["hyb"].shape[-1]
    _validate_shapes(loaded)

    solver_parameters = _parse_solver_parameters(loaded["solver_param"])
    clic_params = _make_clic_params(loaded["label"], solver_parameters)
    angular_operators_corr = {
        name: loaded[f"observable_{name}"]
        for name in ("Lz", "Lplus", "Sz", "Splus")
    }

    corr_to_cf_full = _complete_corr_to_cf(
        loaded["corr_to_cf"], loaded["n_orb"], loaded["n_rot"]
    )
    h_imp_clic = np.ascontiguousarray(
        basis_change_h0(loaded["h_dft"], corr_to_cf_full)
    )
    hyb_clic = np.ascontiguousarray(
        basis_change_h0(np.moveaxis(loaded["hyb"], -1, 0), corr_to_cf_full)
    )
    U_imp_clic = np.ascontiguousarray(
        basis_change_U(loaded["U_mat"], corr_to_cf_full)
    )
    angular_operators_cf = {
        name: np.ascontiguousarray(basis_change_h0(op, corr_to_cf_full))
        for name, op in angular_operators_corr.items()
    }

    loaded.update(
        solver_parameters=solver_parameters,
        clic_params=clic_params,
        angular_operators_corr=angular_operators_corr,
        corr_to_cf_full=corr_to_cf_full,
        hyb_clic=hyb_clic,
        h_imp_clic=h_imp_clic,
        U_imp_clic=U_imp_clic,
        angular_operators_cf=angular_operators_cf,
    )
    loaded["dmft_args"] = (
        loaded["w"],
        loaded["iw"],
        hyb_clic,
        h_imp_clic,
        U_imp_clic,
        clic_params,
    )
    loaded["dmft_kwargs"] = {
        "eim": loaded["eim"],
        "impurity_angular_operators": angular_operators_cf,
        "archive_path": filename,
    }
    return SimpleNamespace(**loaded)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="debug_solver_<label>.h5 archive")
    args = parser.parse_args()
    solver_data = load_solver_data(args.archive)
    print(f"Loaded {solver_data.label!r} from {solver_data.filename}")
    print(f"hyb_clic shape: {solver_data.hyb_clic.shape}")
