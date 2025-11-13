

import numpy as np
from scipy.linalg import eigh, block_diag, eig
from scipy.sparse.linalg import eigsh
import numpy as np
from typing import Tuple, Literal
import matplotlib.pyplot as plt

from clic import * 


np.set_printoptions(precision=3, suppress=True, linewidth=300)


print("*"*42)
Nimp = 2
scheme = ["metal","metal"]
lambda_soc = 0.0
Nb = 5
h0 = build_impurity_model(Nimp,Nb,scheme,[0.0,0.03],lambda_soc)

NF = np.shape(h0)[0]
M = np.shape(h0)[0] // 2

print("*"*42)
print(f"NF = {NF}, M = {M}, Nimp = {Nimp}")
print("*"*42)


print(np.real(h0))
print("*"*12)
ia=[i for i in range(np.shape(h0)[0]) if i%2 == 0]
ia = np.ix_(ia,ia)
print(np.real(h0[ia]))

kanamori = build_kanamori_params(Nimp, U=5.0, J=0.7)
U_imp = build_U_matrix_kanamori(Nimp, **{k:v for k,v in kanamori.items()
                                if k in ("U","J")})
print(f"kanamori = {kanamori}")
print("U_imp shape: ",np.shape(U_imp))

U_imp = umo2so(U_imp,Nimp)

M_total_spinfull = NF
M_total_spatial = M

# Pad the U matrix to the full size of the new hamiltonian
# Assumes U_imp is dense (4-index tensor)
U_0 = np.zeros((M_total_spinfull,)*4, dtype=U_imp.dtype)
imp_size = U_imp.shape[0]
halfimp = imp_size // 2

impindex = [i for i in range(halfimp)] + [i for i in range(M_total_spatial, M_total_spatial+halfimp)]
U_0[np.ix_(impindex, impindex, impindex, impindex)] = U_imp



h0_0,U_0 = transform_integrals_interleaved_to_alphafirst(h0, U_0, M)
C = None

print("h0_0 size: ",np.shape(h0_0))

#--------------------------------------
Nelec = M
Nelec_imp = Nimp
imp_indices = [i for i in range(Nimp)]

# Define impurity indices (now expecting global SPATIAL indices)
imp_indices_spatial = imp_indices

    
# Get the mean field h to get impurity occupation not crazy
hmf,_,_,rho_mf = mfscf(h0_0,U_0,Nelec)

# Double chain expects a interleaved convention
hmf_ab = transform_h0_alphafirst_to_interleaved(hmf)
rhomf_ab = transform_h0_alphafirst_to_interleaved(rho_mf)

Nimp_sf = len(imp_indices) * 2
print(f"DEBUG, Nimp = {Nimp}")
hdc_ab, C_ab, meta = double_chain_by_blocks(hmf_ab,rhomf_ab,
                                                    Nimp_sf,Nelec,
                    analyze_symmetries, get_double_chain_transform_multi)

hdc = transform_integrals_interleaved_to_alphafirst(hdc_ab)
C = transform_integrals_interleaved_to_alphafirst(C_ab)

h0 = C.conj().T @ h0 @ C 
transformation_matrix = C


print("*"*42)




sb = get_imp_starting_basis(np.real(h0_0), Nelec, Nelec_imp, imp_indices)
print(f"sb = {sb}")
cipsi_max_iter = 10


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

