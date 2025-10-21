
from . import clic_clib as cc
from itertools import combinations


def get_fci_basis(num_spatial, num_electrons):
    num_spin_orbitals = 2 * num_spatial
    basis_dets = []
    for occupied_indices in combinations(range(num_spin_orbitals), num_electrons):
        occ_a = [i for i in occupied_indices if i < num_spatial]
        occ_b = [i - num_spatial for i in occupied_indices if i >= num_spatial]
        det = cc.SlaterDeterminant(num_spatial, occ_a, occ_b)
        basis_dets.append(det)
    return sorted(basis_dets)