#!/usr/bin/env python3
import numpy as np
from pathlib import Path

# Directory where the A_i files live
dump_dir = Path("dump")

# List of indices to use, as strings
indices = ["9", "19", "29"]  # extend as needed

# Build file list
files = [dump_dir / f"A_{i}" for i in indices]

# Storage for spectra
omegas = None
spectra = []

# Read all A_i(ω) files
for f in files:
    data = np.loadtxt(f)
    w = data[:, 0]
    Aii = np.sum(data[:, 1:],axis=1)  # second column

    if omegas is None:
        omegas = w
    else:
        # Simple safety check: grids should match
        if not np.allclose(omegas, w):
            raise ValueError(f"Frequency grid mismatch in file {f}")

    spectra.append(Aii)

spectra = np.array(spectra)  # shape (n_indices, n_ω)

# 1) Sum all selected A_ii(ω) to get the DOS
DOS = spectra[0,:]

# Save the total DOS
dos_test = dump_dir / "DOS_test.dat"
np.savetxt(dos_test, np.column_stack([omegas, DOS]),
           header="omega  DOS(omega) = sum_i A_ii(omega)")
print(f"Total DOS written to {dos_test}")

# 2) Convergence check between successive A_i
#    Be careful: ∫(A_i - A_{i-1}) dω is just the difference in total weight.
#    For a shape-difference metric, L1 or L2 norms are more meaningful.
print("\nConvergence metrics between successive spectra:")
print("pair  signed_int   L1_int       L2_int")
for k in range(1, len(spectra)):
    A_prev = spectra[k - 1]
    A_curr = spectra[k]
    diff = A_curr - A_prev

    # Signed integral  ∫ (A_i - A_{i-1}) dω
    signed_int = np.trapezoid(diff, omegas)

    # L1 norm  ∫ |A_i - A_{i-1}| dω  (good for a convergence measure)
    L1_int = np.trapezoid(np.abs(diff), omegas)

    # L2 norm  ∫ (A_i - A_{i-1})^2 dω
    L2_int = np.trapezoid(diff**2, omegas)

    print(f"{indices[k-1]}->{indices[k]}  {signed_int: .3e}  {L1_int: .3e}  {L2_int: .3e}")
