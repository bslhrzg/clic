import os
import numpy as np
import atexit
from typing import Dict, TextIO, Union
import sys
 
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


def print_header(str):
    print("\n")
    print("*"*62)
    print(str)
    print("*"*62)

def print_subheader(str):
    print("\n")
    print("-"*62)
    print(str)
    print("-"*62)

_VERBOSE_LEVEL = 3

# 2. Global dictionary to hold open file handles.
#    Key: filename (str), Value: file object (TextIO)
_OPEN_FILES: Dict[str, TextIO] = {}


def _close_all_files():
    """A cleanup function to close all managed files."""
    # This is called automatically when the program exits.
    for f in _OPEN_FILES.values():
        f.close()
    _OPEN_FILES.clear()

# 3. Register the cleanup function to run on program exit.
#    This is the crucial part for resource safety.
atexit.register(_close_all_files)

def set_verbosity(level: int):
    """Sets the global verbosity level for vprint."""
    global _VERBOSE_LEVEL
    _VERBOSE_LEVEL = level
    if _VERBOSE_LEVEL > 0:
        print(f"[System] Verbosity level set to {_VERBOSE_LEVEL}")

def vprint(required_level: int, *args, filename: str = None, **kwargs):
    """
    Prints to console or a file based on verbosity and the 'filename' argument.

    - If filename is None, prints to the console (stdout).
    - If filename is provided, appends the message to that file.
    - Manages file handles automatically for performance.
    """
    # Early exit if the message is not important enough to be printed
    if _VERBOSE_LEVEL < required_level:
        return

    # Prepare the message content
    prefix = f"[V{required_level}]"
    full_message = f"{prefix} {' '.join(map(str, args))}"

    if filename is None:
        # Case 1: Print to console (standard output)
        print(full_message, **kwargs)
    else:
        # Case 2: Print to a file
        try:
            if filename not in _OPEN_FILES:
                # If file is not yet open, open it in append mode ('a')
                # and store the handle in our global dictionary.
                _OPEN_FILES[filename] = open(filename, 'a', encoding='utf-8')
            
            # Get the file handle and write to it
            file_handle = _OPEN_FILES[filename]
            print(full_message, file=file_handle, **kwargs)
            
            # Optional: Immediately flush to ensure it's written to disk
            # Useful for long-running processes or debugging.
            file_handle.flush()

        except IOError as e:
            # Handle potential errors like permission denied
            print(f"[vprint Error] Could not write to file '{filename}': {e}", file=sys.stderr)