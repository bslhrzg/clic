# hybfit/__init__.py

# Expose the main high-level functions
from .hybfit import fit, analyze_fit

# Expose the most common utility functions directly
from .utils import (
    load_delta_from_files,
    create_dummy_delta,
    delta_from_poles
)

# You can also import the entire utils module if needed for less common functions
from . import utils