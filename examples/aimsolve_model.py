"""Run with: python examples/aimsolve_model.py [new_output.h5].

All energies are in Ry. Replace h_imp and U_imp with your model tensors.
The bath consists of two spatial orbitals (four spin-orbitals).
"""
import sys
import numpy as np
from clic import aimsolve


def main():
    interaction = 0.4
    h_imp = -interaction / 2 * np.eye(2)
    U_imp = np.zeros((2, 2, 2, 2), dtype=complex)
    U_imp[0, 1, 0, 1] = U_imp[1, 0, 1, 0] = interaction
    bath = {
        "energies": [-0.2, 0.3, -0.2, 0.3],
        "V": [[0.05, 0.08, 0., 0.], [0., 0., 0.05, 0.08]],
    }
    result = aimsolve(
        ws=np.linspace(-1., 1., 1001),
        h_imp=h_imp, U_imp=U_imp, bath=bath,
        eim=0.005,
        input_path=None,  # use only the explicit settings below
        solver_params={
            "ci_type": "fci",
            "basis_prep_method": "none",
            "nelec_range": 3,  # total number of impurity + bath electrons
            "num_roots": 4,
            "temperature": 5.,
            "NappH": 8,
            "L_lanczos": 100,
        },
        archive_path=sys.argv[1] if len(sys.argv) > 1 else None,
    )
    print("G_imp shape:", result["G_imp"].shape)
    print("Impurity occupation:", result["thermal_avgs"]["avg_occ"])


if __name__ == "__main__":
    main()
