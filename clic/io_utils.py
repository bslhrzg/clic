import os
import numpy as np
from typing import Union

def dump(
    F: Union[np.ndarray, float, int, complex],
    x: np.ndarray,
    filename: str,
    output_dir: str = "dump",
    float_fmt: str = "%.5f",
    header_comment: str = "#"
) -> None:
    """
    Writes the array `x` and array `F` to a file.

    This is a direct Python port of the provided Julia function. The key
    difference is that the dependent axis in `F` is the FIRST axis (F.shape[0]).

    - `x`: 1D array to be the first column in the output.
    - `F`: Array that can be a scalar, 1D, 2D, or 3D.
    - `filename`: String specifying the path to the output file.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    full_path = os.path.join(output_dir, filename)

    # --- Prepare data based on F's dimensions ---
    header = ""
    F_processed = None

    if not isinstance(F, np.ndarray):  # Scalar case
        if not np.isscalar(F):
             raise TypeError("F must be a scalar or a numpy array.")
        # Create a column vector by repeating the scalar F
        F_processed = np.full((len(x), 1), F)
    else:
        nd = F.ndim
        # Python convention: first dimension must match x
        if F.shape[0] != len(x):
            raise ValueError(
                f"The first dimension of F (shape: {F.shape}) must match the "
                f"length of x ({len(x)})"
            )

        if nd == 1:
            F_processed = F.reshape(-1, 1)
        elif nd == 2:
            F_processed = F
        elif nd == 3:
            p, q = F.shape[1], F.shape[2]
            F_processed = F.reshape(len(x), p * q)
            # Generate the exact header from the Julia function
            header_lines = [f"{header_comment} Column index map to original (p, q) indices:"]
            counter = 2  # Column 1 is 'x'
            for i in range(p):
                line = " ".join([f"{counter + j:<4d}" for j in range(q)])
                header_lines.append(f"{header_comment} {line}")
                counter += q
            header = "\n".join(header_lines)
        else:
            raise ValueError(f"Unsupported number of dimensions for F: {nd}")

    # Combine x and the processed F into a single array
    # Note: the data array can have a mix of types if F is complex
    data_to_save = np.c_[x, F_processed]

    # --- Manual file writing to replicate Julia's sprintf behavior ---
    with open(full_path, "w") as f:
        if header:
            f.write(header + "\n")

        for row in data_to_save:
            formatted_parts = []
            for item in row:
                # Format floats with the specified format string,
                # otherwise, convert to a standard string (handles complex numbers).
                if isinstance(item, (float, np.floating)):
                    formatted_parts.append(format(item, float_fmt.lstrip('%')))
                else:
                    formatted_parts.append(str(item))
            f.write(" ".join(formatted_parts) + "\n")

    print(f"Data saved to '{full_path}'")