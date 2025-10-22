import numpy as np
from scipy.linalg import eigh, block_diag, eig
from scipy.sparse.linalg import eigsh
from clic import * 
import numpy as np
from typing import Tuple, Literal



# --- Main script ---

np.set_printoptions(precision=4, suppress=True)


# 1. System Parameters
nb = 9
M = 1 + nb
u = 4.0
mu = u / 2
Nelec = M 
Nelec_half = M // 2
e_bath = np.linspace(-2.0, 2.0, nb)
if nb == 1: e_bath = [0.0]
V_bath = np.full(nb, 0.5)


    
h0, U_mat = get_impurity_integrals(M, u, e_bath, V_bath, mu)


# ----------------------

# --- Find Ground State ---
basis = get_fci_basis(M, Nelec)
#inds, blocks = partition_by_Sz(basis)    # lists of indices + Sz values
basis, idxs0 = subbasis_by_Sz(basis, 0.0)  # S_z = 0 sector
print(f"basis size = {len(basis)}")


H_sparse = build_hamiltonian_openmp(basis, h0, U_mat)
eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
e0 = eigvals[0]
print(f"Impurity Model Ground State Energy: {e0:.6f}")
psi0star = eigvecs[:, 0]


# -----------------------

hmf, es, Vs, rho = mfscf(h0,U_mat,M)
h0,U_mat = basis_change_h0_U(h0,U_mat,Vs)

basis0 = get_starting_basis(np.real(h0), M)
H = get_ham(basis0,h0,U_mat)
eigvals, eigvecs = eig(H.toarray())

e0 = eigvals[0]
psi0 = Wavefunction(M, basis0, eigvecs[:,0])
print(f"e0 = {e0}")

one_bh = get_one_body_terms(h0, M)
two_bh = get_two_body_terms(U_mat, M)

res_cisd = selective_ci(h0,U_mat,M,Nelec,cipsi_one_iter,max_iter=1,prune_thr=1e-12)
print("CISD E0 =", res_cisd["energy"])
print("CISD dim =", len(res_cisd["basis"]))

do_no = True 
if do_no:
    psi_cisd = res_cisd["wavefunction"]
    rdm1 = one_rdm(psi_cisd,M)
    eno,nos = eig(rdm1[:M,:M])
    nos = double_h(nos,M)
    h0,U_mat = basis_change_h0_U(h0,U_mat,nos)


cipsi_max_iter = 15
res = selective_ci(h0,U_mat,M,Nelec,cipsi_one_iter,Nmul=1.0,max_iter=cipsi_max_iter)

print("Final E0 =", res["energy"])
print("Final dim =", len(res["basis"]))



