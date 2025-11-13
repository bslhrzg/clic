

import numpy as np
from scipy.linalg import eigh, block_diag, eig
from scipy.sparse.linalg import eigsh
import numpy as np
from typing import Tuple, Literal
import matplotlib.pyplot as plt

from clic import * 


def greedy_gauge(basis, amps, n_orb, K=100, max_sweeps=5):
    """
    Heuristic Z2 gauge optimisation.

    basis : list of SlaterDeterminants
    amps  : array of CI amplitudes c_i (can be complex, we use real part)
    n_orb : number of spin orbitals
    K     : number of most important determinants to use in the objective
    max_sweeps : how many full passes over orbitals

    Returns
    -------
    sigma : array of shape (n_orb,) with entries ±1
    """

    amps = np.asarray(amps)

    # select top K determinants by weight
    idx = np.argsort(-np.abs(amps))[:K]
    important_basis = [basis[i] for i in idx]
    important_amps  = amps[idx]

    # use weights w_i = |c_i|
    w = np.abs(important_amps)
    # take real part for objective (imaginary parts are usually small noise)
    c_real = np.real(important_amps)

    K_eff = len(important_basis)

    # precompute occ lists and orbital -> list of determinants map
    occ = []
    orb_to_dets = [[] for _ in range(n_orb)]
    for k, det in enumerate(important_basis):
        occ_k = list(det.alpha_occupied_indices()) + list(det.beta_occupied_indices())
        occ.append(occ_k)
        for p in occ_k:
            orb_to_dets[p].append(k)

    # initial gauge: all +1
    sigma = np.ones(n_orb, dtype=int)
    # determinant sign factor under current sigma
    s_det = np.ones(K_eff, dtype=int)

    # objective F = sum_i w_i * s_i * c_i
    F = np.dot(w, s_det * c_real)

    for sweep in range(max_sweeps):
        improved = False
        for p in range(n_orb):
            det_list = orb_to_dets[p]
            if not det_list:
                continue

            # change in F if we flip sigma[p]
            # for each determinant containing p, s_i -> -s_i
            # contribution delta F_i = w_i * (-s_i c_i) - w_i * (s_i c_i) = -2 w_i s_i c_i
            delta = 0.0
            for k in det_list:
                delta += -2.0 * w[k] * s_det[k] * c_real[k]

            if delta > 0:
                # accept flip
                sigma[p] *= -1
                for k in det_list:
                    s_det[k] *= -1
                F += delta
                improved = True

        if not improved:
            break

    return sigma



# 1. System Parameters
nb = 11
M = 1 + nb
u = 0.1
mu = u / 2
Nelec = M 
Nelec_imp = 1 
imp_indices = [0]
Nelec_half = M // 2
e_bath = np.linspace(-0.2, 0.2, nb)
if nb == 1: e_bath = [0.0]
V_bath = np.full(nb, 0.1)


    
h0_0, U_0 = get_impurity_integrals(M, u, e_bath, V_bath, mu)

print("h0_0 size: ",np.shape(h0_0))


method = "none"

if method == "dbl_chain":

    h0_spin = np.real(h0_0[:M, :M])
    Nelec_half = Nelec // 2 

    h_final_matrix, C_spin = get_double_chain_transform(
        h0_spin, u, Nelec_half
    )



    h_final_matrix[0, 0] = -u / 2

    C = block_diag(C_spin, C_spin)
    h0_0 = C.conj().T @ h0_0 @ C


random_signs = np.random.choice([1,-1],size=2*M)
random_signs[:M] = random_signs[M:]

#print(f"random signs: {random_signs}")
C = np.diag(random_signs)
#h0_0,U_0 = basis_change_h0_U(h0_0,U_0,C)

sb = get_imp_starting_basis(np.real(h0_0), Nelec, Nelec_imp, imp_indices)
print(f"sb = {sb}")
cipsi_max_iter = 3
res = selective_ci(
    h0_0, U_0, C,
    M, Nelec,
    sb,
    generator=hamiltonian_generator, 
    selector=cipsi_select,
    num_roots=1,
    one_bh=None,
    two_bh=None,
    max_iter=cipsi_max_iter,
    conv_tol=1e-6,
    prune_thr=1e-6,
    Nmul = None,
    min_size=513,
    max_size=1e5,
    verbose=True)

energies = res.energies 
psis = res.wavefunctions
basis = res.basis 

print(f"energies = {energies}")
print(f"len basis = {len(basis)}")
print(f"len(basis(psis0)) = {len((psis[0]).get_basis())}")


amps = psis[0].get_amplitudes()

sortamps = np.argsort(np.abs(amps))[::-1]
basis = [basis[s] for s in sortamps]
amps = amps[sortamps]

for i in range(10):
    print(f"{i}: {basis[i]} {amps[i]}")

sigmas =  greedy_gauge(basis, amps, 2*M, K=100, max_sweeps=1000)
#sigmas = np.random.choice([1,-1],size=2*M)
#sigmas[:M] = sigmas[M:]
print(f"sigmas = {sigmas}")

C = np.diag(sigmas)
h0,U_0 = basis_change_h0_U(h0_0,U_0,C)

res = selective_ci(
    h0, U_0, C,
    M, Nelec,
    sb,
    generator=hamiltonian_generator, 
    selector=cipsi_select,
    num_roots=1,
    one_bh=None,
    two_bh=None,
    max_iter=cipsi_max_iter,
    conv_tol=1e-6,
    prune_thr=1e-6,
    Nmul = None,
    min_size=513,
    max_size=1e5,
    verbose=True)

energies = res.energies 
psis = res.wavefunctions
basis = res.basis 

print(f"energies = {energies}")
print(f"len basis = {len(basis)}")
print(f"len(basis(psis0)) = {len((psis[0]).get_basis())}")


amps = psis[0].get_amplitudes()

sortamps = np.argsort(np.abs(amps))[::-1]
basis = [basis[s] for s in sortamps]
amps = amps[sortamps]

for i in range(10):
    print(f"{i}: {basis[i]} {amps[i]}")