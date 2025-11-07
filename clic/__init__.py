# clic/__init__.py

"""
CLIC: Configuration interaction and Lanczos for Impurity Calculations
"""

from clic.symmetries import symmetries
from clic.lanczos.block_lanczos import *
from clic.lanczos.scalar_lanczos import *

from clic.mf.mf import mfscf, block_electron_counts, group_electron_counts

from clic.model.double_chains import double_chain_by_blocks,get_double_chain_transform_multi