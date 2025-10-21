# clic/__init__.py

"""
CLIC: Configuration interaction and Lanczos for Impurity Calculations
"""

# 1. Import the compiled C++ extension module.
#    The name 'clic_clib' comes from CMakeLists.txt
from . import clic_clib

# 2. Expose the most important C++ classes to the top level of the package.
#    This allows users to write `clic.Wavefunction` instead of
#    `clic.clic_clib.Wavefunction`.
from .clic_clib import (
   
    SlaterDeterminant,
    Wavefunction,
    Spin,
    SpinOrbitalOrder,

    apply_creation,
    apply_annihilation,
    get_creation_operator,
    get_annihilation_operator,

    apply_one_body_operator,
    apply_two_body_operator,

    get_connections_one_body,
    get_connections_two_body,


    build_hamiltonian_openmp,
)

# 3. Import your Python helper functions from the tools submodule.
from . import basis_1p,basis_Np,gfs,hamiltonians

# 4. Expose the most important Python functions to the top level.
from .hamiltonians import (
    get_impurity_integrals,
    create_hubbard_V,
    get_one_body_terms,
    get_two_body_terms
)

from .basis_1p import transform_integrals_interleaved_to_alphafirst

from .basis_Np import get_fci_basis

from .gfs import(
    wf_to_vec,
    expand_basis_by_H,
    green_function_block_lanczos_fixed_basis
)
