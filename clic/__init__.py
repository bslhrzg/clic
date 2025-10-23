# clic/__init__.py

"""
CLIC: Configuration interaction and Lanczos for Impurity Calculations
"""


# Expose the most important C++ classes to the top level of the package.
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


# Expose the most important Python functions to the top level.
from .hamiltonians import (
    get_impurity_integrals,
    create_hubbard_V,
)

from .basis_1p import (
    transform_integrals_interleaved_to_alphafirst,
    umo2so,
    double_h,
    basis_change_h0_U
)

from .basis_transforms import *

from .basis_Np import (
    get_fci_basis,
    partition_by_Sz,
    subbasis_by_Sz,
    get_starting_basis
)

from .ops import (
    one_rdm, 
    get_ham,
    get_one_body_terms,
    get_two_body_terms
)
from .gfs import *

from .mf import mfscf
from .basis_transforms import *

from .sci import selective_ci, hamiltonian_generator,   cipsi_one_iter