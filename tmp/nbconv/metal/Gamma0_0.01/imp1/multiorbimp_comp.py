

import numpy as np
from scipy.linalg import eigh, block_diag, eig
from scipy.sparse.linalg import eigsh
import numpy as np
from typing import Tuple, Literal
import matplotlib.pyplot as plt

from clic import * 


np.set_printoptions(precision=3, suppress=True, linewidth=300)


print("*"*42)
Nimp = 1

scheme = ["metal" for _ in range(Nimp)]
lambda_soc = 0.0
Nb = 49
U_kanamori = 0.5 # ~6.8 eV
J_kanamori = 0.0
nimp = 1
dc = U_kanamori * (nimp - 0.5) - 0.5 * J_kanamori * (nimp-1)
eps = [-dc for i in range(Nimp)]

D=1
Gamma0=0.01
gap=1.0
h0_0 = build_impurity_model(Nimp,Nb,scheme,eps,lambda_soc,
                           D=D, Gamma0=Gamma0, gap=gap)

NF = np.shape(h0_0)[0]
M = np.shape(h0_0)[0] // 2

print("*"*42)
print(f"NF = {NF}, M = {M}, Nimp = {Nimp}")
print("*"*42)


print(np.real(h0_0))
print("*"*12)
ia=[i for i in range(np.shape(h0_0)[0]) if i%2 == 0]
ia = np.ix_(ia,ia)
print(np.real(h0_0[ia]))



U_0 = build_U_tensor(Nimp, M,  U=U_kanamori, J=J_kanamori)
h0_0,U_ = transform_integrals_interleaved_to_alphafirst(h0_0, U_0, M)
C = None

h0_legacy = h0_0.copy()

if False:
    for i in range(NF):
        for j in range(NF):
            for k in range(NF):
                for l in range(NF):
                    if U_0[i,j,k,l] != 0:
                        print(f"U({i,j,k,l}) = {U_0[i,j,k,l]}")
                        if i==j and i==k and i==l : 
                            print("setting to 0")
                            U_0[i,j,k,l] = 0

print("h0_0 size: ",np.shape(h0_0))
print("U_0 size: ",np.shape(U_0))

#--------------------------------------

nelec_bath = calculate_bath_filling(h0_0, Nimp)
Nelec = nimp + nelec_bath

Nelec_imp = nimp
print(f"Going with Nelec_imp = {nimp}, nbath = {nelec_bath}, Nelec = {Nelec}")

imp_indices_spatial = [i for i in range(Nimp)]
imp_indices_full = imp_indices_spatial + [e + M for e in imp_indices_spatial]
print(f"imp_indices_full = {imp_indices_full}")
# Define impurity indices (now expecting global SPATIAL indices)


dodblchain=True
if dodblchain:  
    print("h0_0 = ")
    print(np.real(h0_0))
    # Get the mean field h to get impurity occupation not crazy
    hmf,_,_,rho_mf = mfscf(h0_0,U_0,Nelec)
    print("hmf = ")
    print(np.real(hmf))
    #hmf = h0_0
    #rho_mf = 

    # Double chain expects a interleaved convention
    hmf_ab = transform_h0_alphafirst_to_interleaved(hmf)
    rhomf_ab = transform_h0_alphafirst_to_interleaved(rho_mf)

    Nimp_sf = len(imp_indices_full)
    print(f"DEBUG, Nimp_sf = {Nimp_sf}")
    print("hmf_ab = ")
    print(hmf_ab)
    hdc_ab, C_ab, meta = double_chain_by_blocks(hmf_ab,rhomf_ab,
                                                        Nimp_sf,Nelec,
                        analyze_symmetries, get_double_chain_transform_multi,
                        verbose=True)

    print("hdc_ab = ")
    print(np.real(hdc_ab))

    print("C_ab = ")
    print(np.real(C_ab))
    
    hdc = transform_integrals_interleaved_to_alphafirst(hdc_ab)
    C = transform_integrals_interleaved_to_alphafirst(C_ab)


    print("C = ")
    print(np.real(C))

    h0_0 = C.conj().T @ h0_0 @ C 
    transformation_matrix = C
else : 
    C = None


print("*"*42)
print("h0_0 = ")
print(np.real(h0_0))

print("h0_legacy = ")
print(np.real(h0_legacy))


#assert 1 == 0 

sb = get_imp_starting_basis(np.real(h0_0), Nelec, Nelec_imp, imp_indices_spatial)
print(f"sb = {sb}")
cipsi_max_iter = 12

#print("h0 = ")
#print(h0_0)



sci=True
if sci :
    res = selective_ci(
        h0_0, U_0, C,
        M, Nelec,
        sb,
        generator=hamiltonian_generator, 
        selector=cipsi_select,
        num_roots=2,
        one_bh=None,
        two_bh=None,
        max_iter=cipsi_max_iter,
        conv_tol=1e-5,
        prune_thr=1e-6,
        Nmul = 2,
        min_size=513,
        max_size=1e5,
        verbose=True)
else : 
    res=do_fci(h0_0,U_0,M,Nelec,num_roots=2,Sz=0,verbose=True)

energies = res.energies 
psis = res.wavefunctions
basis = res.basis 

e0 = energies[0]
psi0 = psis[0]


print(f"energies = {energies}, e0 = {e0}")
print(f"len basis = {len(basis)}")
print(f"len(basis(psis0)) = {len((psis[0]).get_basis())}")


rdm = one_rdm(psi0, M)
print(f"rdm : ")
print(rdm)

imp_occ = 0 
for i in imp_indices_full:
    imp_occ += rdm[i,i]

print(f"impurity occupation: {imp_occ}")
print(f"total occupation : {np.trace(rdm)}")
NappH = 1
eta = 0.02
ws = np.linspace(-2,2,1001)
target_indices = imp_indices_full
gfmeth = "scalar_continued_fraction"
one_bh_n = get_one_body_terms(h0_0, M)
two_bh_n = get_two_body_terms(U_0, M)
coeff_thresh = 1e-6
L = 200

print(f"target_indices = {target_indices}")
G_sub_block_n = get_green_block(M, psi0, e0, NappH, eta, 
                                                      h0_0, U_0, ws,target_indices, gfmeth, 
                                                      one_bh_n,two_bh_n, coeff_thresh, L
                                                      )


weight = 1.0
G_total_diag = np.zeros((len(ws), len(target_indices)), dtype=np.complex128)
for i in range(len(target_indices)):
    G_total_diag[:, i] += weight * G_sub_block_n[:, i, i]

    
print("\nThermally-averaged calculation finished.")
A_w_total = -(1 / np.pi) * np.imag(G_total_diag)

# Save and plot the final, thermally-averaged results
#if self.settings.output.gf_diag_txt_file:
dodump=True 
if dodump:
    dump(
        A_w_total,
        ws,
        'A_'+str(Nb),
    )