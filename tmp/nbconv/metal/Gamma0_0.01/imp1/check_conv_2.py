#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# Directory where the A_nb files live
dump_dir = Path("dump")

# List of nb values, as strings (corresponding to file names A_5, A_9, ...)
indices = ["9", "19", "29", "39"]  # extend as needed

# Build file list
files = [dump_dir / f"A_{i}" for i in indices]

# Storage
omegas = None
DOS_list = []  # will store DOS for each nb, shape -> (n_nb, n_ω)

# --- Read all A_nb files and build DOS(nb)(ω) ---
for f in files:
    data = np.loadtxt(f)
    w = data[:, 0]
    A_mat = data[:, 1:]  # all diagonal components A_ii(ω) for this nb

    if omegas is None:
        omegas = w
    else:
        # Check that ω grid is the same in all files
        if not np.allclose(omegas, w):
            raise ValueError(f"Frequency grid mismatch in file {f}")

    # Total DOS for this nb: sum over all diagonal components
    DOS_nb = np.sum(A_mat, axis=1)  # sum over columns, keep ω index
    DOS_list.append(DOS_nb)

DOS_array = np.array(DOS_list)  # shape (n_nb, n_ω)

# --- Save all DOS(nb) together in one file ---
# Columns: ω, DOS_nb5, DOS_nb9, ...
header_cols = ["omega"] + [f"DOS_nb{idx}" for idx in indices]
header = " ".join(header_cols)

out_data = np.column_stack([omegas] + [DOS_array[i, :] for i in range(len(indices))])
dos_out = dump_dir / "DOS_all_nb.dat"
np.savetxt(dos_out, out_data, header=header)
print(f"All DOS(nb) written to {dos_out}")
print(f"Columns: {header}")

# --- Convergence check between successive nb using DOS(nb)(ω) ---
print("\nConvergence metrics between successive nb (using total DOS):")
print("pair     L1_int      L2_int")

for k in range(1, len(DOS_array)):
    DOS_prev = DOS_array[k - 1]
    DOS_curr = DOS_array[k]
    diff = DOS_curr - DOS_prev


    # ∫ |DOS_nb(k) - DOS_nb(k-1)| dω
    L1_int = np.trapezoid(np.abs(diff), omegas)

    # ∫ (DOS_nb(k) - DOS_nb(k-1))^2 dω
    L2_int = np.trapezoid(diff**2, omegas)

    print(f"{indices[k-1]}->{indices[k]}  {L1_int: .3e}   {L2_int: .3e}")


# --- Optional: Lorentzian broadening tool (per DOS) ---

def lorentzian_broaden_fft(omegas, A, gamma):
    """
    Convolve A(ω) with a Lorentzian of HWHM = gamma using FFT.

    Assumes uniform ω grid.
    """
    dw = np.diff(omegas)
    if not np.allclose(dw, dw[0]):
        raise ValueError("ω grid is not uniform; FFT broadening is unsafe.")

    dw = dw[0]
    N = len(omegas)

    # k grid corresponding to FFT frequencies
    k = 2 * np.pi * np.fft.fftfreq(N, d=dw)

    A_k = np.fft.fft(A)
    A_broadened_k = A_k * np.exp(-gamma * np.abs(k))
    A_broadened = np.fft.ifft(A_broadened_k).real

    return A_broadened


# Example usage for convergence with broadening:
if True:
    gammas = [0.01, 0.02, 0.05]
    for gamma in gammas:
        DOS_b = np.array([lorentzian_broaden_fft(omegas, DOS_array[i], gamma)
                          for i in range(len(indices))])

        print(f"\nConvergence metrics with Lorentzian broadening, gamma = {gamma}:")
        print("pair     L1_int      L2_int")
        for k in range(1, len(DOS_b)):
            plt.plot(DOS_b[k])
            plt.show()
            diff = DOS_b[k] - DOS_b[k - 1]
            L1_int = np.trapezoid(np.abs(diff), omegas)
            L2_int = np.trapezoid(diff**2, omegas)
            print(f"{indices[k-1]}->{indices[k]}   {L1_int: .3e}   {L2_int: .3e}")