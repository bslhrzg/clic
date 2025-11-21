import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from clic_clib import get_creation_operator, get_annihilation_operator
import clic_clib as cc
from clic import * 
import matplotlib.pyplot as plt 
from utils import *

np.set_printoptions(precision=3, suppress=True, linewidth=300)


print("*"*42)
#----------------------------------------------------------
# SIAM 
#----------------------------------------------------------

Nb=5
u = 0.5

h0_0,U_0 = h0U_siam_half_filling(Nb,u)
C = None

NF = h0_0.shape[0]
M = NF // 2 

print("h0 siam = ")
print(h0_0.real)

#----------------------------------------------------------
# ED 
#----------------------------------------------------------
full_ed_basis = np.arange(2**NF)
Nelec_basis = {ne : np.array([i for i in full_ed_basis if bin(i).count("1") == ne]) for ne in range(NF)}
Nelec=NF//2
fcibasis = full_ed_basis[Nelec_basis[Nelec]]


c_dag = [get_creation_operator(NF, i) for i in range(1, NF + 1)]
c = [get_annihilation_operator(NF, i) for i in range(1, NF + 1)]
ns = [(c_dag[i] @ c[i])[np.ix_(fcibasis,fcibasis)] for i in range(len(c_dag))]

hop = h0_to_hop(h0_0,c,c_dag)
Uop = U_to_Uop(U_0,c,c_dag)

Hed = hop + Uop 
Hed = Hed.toarray()


H = Hed[np.ix_(fcibasis,fcibasis)]

eigenvalues, eigenvectors = eigh(H)

print("*"*42)
print(f"eige : {eigenvalues}")


psi0 = eigenvectors[:,0]
ntot = 0
for i in range(NF):
    ni = ns[i]
    av_ni = get_av_0T(psi0,ni)
    ntot+=av_ni
    print(f"n({i}) = {av_ni}")
print(f"Nelec = {Nelec}, ntot = {ntot}")

#----------------------------------------------------------
# FCI 
#----------------------------------------------------------


res=do_fci(h0_0,U_0,M,Nelec,num_roots=2,Sz=0,verbose=True)
C=None 
Nelec_imp = 1 
imp_indices_spatial = [0]
sb = get_imp_starting_basis(np.real(h0_0), Nelec, Nelec_imp, imp_indices_spatial)
cipsi_max_iter = 12

res=selective_ci(
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
#----------------------------------------------------------
# Add core 
#----------------------------------------------------------
print("*"*42)
print("Add core")
print("*"*42)

e_c = -0.5
U_dc = 0.5
h0,U = add_core_site(h0_0, U_0, e_c, U_dc)

NF = h0.shape[1]
M = NF // 2
print(f"NF = {NF}, M = {M}")


print("h0 with core :")
print(h0.real)

full_ed_basis = np.arange(2**NF)
Nelec_basis = {ne : np.array([i for i in full_ed_basis if bin(i).count("1") == ne]) for ne in range(NF)}

# +2 --> filled core
Nelec = Nelec + 2
print(f"Nelec = {Nelec}")
fcibasis = full_ed_basis[Nelec_basis[Nelec]]
print(f"len(fcibasis) = {len(fcibasis)}")


c_dag = [get_creation_operator(NF, i) for i in range(1, NF + 1)]
c = [get_annihilation_operator(NF, i) for i in range(1, NF + 1)]
ns = [(c_dag[i] @ c[i])[np.ix_(fcibasis,fcibasis)] for i in range(len(c_dag))]


hop = h0_to_hop(h0,c,c_dag)
Uop = U_to_Uop(U,c,c_dag)

Hed = hop + Uop 
Hed = Hed.toarray()


H = Hed[np.ix_(fcibasis,fcibasis)]

eigenvalues, eigenvectors = eigh(H)


print(f"ED:     eige[:10]: {eigenvalues[:10]}")
print(f"max eige = {np.max(eigenvalues)}, spectral window = {np.max(eigenvalues)-np.min(eigenvalues)}")

psi0 = eigenvectors[:,0]
e0 = eigenvalues[0]
ntot = 0
for i in range(NF):
    ni = ns[i]
    av_ni = get_av_0T(psi0,ni)
    ntot+= av_ni
    print(f"n({i}) = {av_ni}")
print(f"Nelec = {Nelec}, ntot = {ntot}")

res=do_fci(h0,U,M,Nelec,num_roots=1,verbose=True)

C=None 
Nelec_imp = 1 
imp_indices_spatial = [0]
sb = get_imp_starting_basis(np.real(h0), Nelec, Nelec_imp, imp_indices_spatial)
cipsi_max_iter = 12

if True:
    res=selective_ci(
    h0, U, C,
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



psi0_ci = res.wavefunctions[0]
basis_ci = psi0_ci.get_basis()
print(f"basis_ci length : {len(basis_ci)}")
###########


#psi0_ci_vec = wf_to_vec(psi0_ci, basis_ci)  # basis_exp is the same used to build H_in_basis_exp
#ov_gs = psi0_ci_vec.conj().T @ psi0
#print("overlap between gs (CI vs ED) =", ov_gs, "|ov| =", abs(ov_gs))

#----------------------------------------------------------
# XAS 
#----------------------------------------------------------
print("*"*42)
print("XAS")
print("*"*42)

valence_index = 0
core_index = M-1 
ws = np.linspace(-10,10,1001)
eta=1e-1
op_xas_alpha = c_dag[valence_index] @ c[core_index]
op_xas_alpha = op_xas_alpha[np.ix_(fcibasis,fcibasis)]

#g_xas = ED_spectra(op_xas_alpha,psi0,e0,H,ws,eta)
#plt.plot(ws,-np.imag(g_xas))
#plt.show()

#----------------------------------------------------------
# Create Pump 
#----------------------------------------------------------
print("*"*42)
print("Pump")
print("*"*42)




pump_xas = XAS_core_val(core_index,valence_index,NF)

bath_indexes = [i for i in range(M) if i not in [0,M-1]]
pump_bath = XAS_core_cond(core_index,bath_indexes,NF)


RY   = 2.179874099e-18   # J
HBAR = 1.05457182e-34    # J s
FS   = 1e-15             # s


code_to_fs = HBAR / RY / FS   # fs per code unit
print(f"fs per code unit : {code_to_fs}")
dt_code = 0.1        # time step in code unit


Ttot_fs_desired = 100
Ttot_code = Ttot_fs_desired / code_to_fs   # 
Nsteps  = int(Ttot_code / dt_code)         #

ts_code = np.arange(Nsteps) * dt_code          # times in code units
ts_fs   = ts_code * code_to_fs                 # same times in fs for plotting

print(f"T_total_fs = {ts_fs[-1]}, Nsteps = {Nsteps}")

E0=50





nvas = []
nvbs = []
ncas = []
ncbs = []


psi = psi0.copy()
psi_t_list = []
nvas, nvbs, ncas, ncbs = [], [], [], []
nvas_ex, ncas_ex = [], []


h0_quench = XAS_core_cond(core_index,bath_indexes,NF)
hop_quench = h0_to_hop(h0_quench, c, c_dag).toarray()
hop_quench = hop_quench[np.ix_(fcibasis, fcibasis)]
H_quench = E0 * hop_quench

psi_quenched =  H_quench @ psi0 
psi_quenched /= np.linalg.norm(psi_quenched)


#########
# In the sci basis
#H_quench = get_ham(basis_ci,E0*h0_quench,U*0,method="1",tables=None)
#psi_quenched_ci = H_quench @ psi0_ci.get_amplitudes()
#psi_quenched_ci = cc.Wavefunction(M,basis_ci,psi_quenched_ci)
#psi_quenched_ci.normalize()

toltables = 0
tables_quench = cc.build_hamiltonian_tables(E0*h0_quench,U*0,toltables)
tol_el = 0
psi_quenched_ci = cc.apply_hamiltonian(psi0_ci, tables_quench,E0*h0_quench,U*0,tol_el)

psi_quenched_ci.normalize()
psi_quenched_ci_basis = psi_quenched_ci.get_basis()
print(f"len(psi_quenched_sci_basis)={len(psi_quenched_ci_basis)}")
# Now psi_quenched_sci = H_quench @ psi_sci, normalized

one_body_terms = get_one_body_terms(h0, M)
two_body_terms = get_two_body_terms(U, M)
NappH=1
basis_exp = expand_basis_by_H(psi_quenched_ci_basis, one_body_terms, two_body_terms, NappH)
#basis_exp = basis_ci
#print(f"len(basis_exp)={len(basis_exp)}")
H_in_basis_exp = get_ham(basis_exp,h0,U,method="1",tables=None)

#diffH = np.sum(np.abs(H_in_basis_exp.toarray() - H))
#rint("diffH = ",diffH)

#psis_t = lanczos_time_evolution(H, psi_quenched, ts_code, L=150, reorth=True)
psi_quenched_in_basis = wf_to_vec(psi_quenched_ci, basis_exp)
psis_t = lanczos_time_evolution(H_in_basis_exp, psi_quenched_in_basis, ts_code, L=150, reorth=True)


#angle_psis = psi_quenched_in_basis.conj().T @ psi_quenched
#print("angle_psis: ",angle_psis)

#ov_quench = psi_quenched_in_basis.conj().T @ psi_quenched
#print("overlap between quench (CI vs ED) =", ov_quench, "|ov| =", abs(ov_quench))

nvas, ncas = [], []
for k, psi_t in enumerate(psis_t):
    wf = cc.Wavefunction(M,basis_exp,psi_t)
    nva = get_n_i_j(wf,M,valence_index,valence_index)
    nvas.append(nva)
    nca = get_n_i_j(wf,M,core_index,core_index)
    ncas.append(nca)



psis_t_ex = []
coeff0 = eigenvectors.conj().T @ psi_quenched   # shape (Nd,)
for t in ts_code:
    phase = np.exp(-1j * eigenvalues * t)    # (Nd,)
    psi_t = eigenvectors @ (coeff0 * phase)             # (Nd,)
    psis_t_ex.append(psi_t)
psis_t_ex = np.array(psis_t_ex)  # shape (Nt, Nd)


for psi_t in psis_t_ex:
    nvas_ex.append(get_av_0T(psi_t, ns[valence_index]))
    ncas_ex.append(get_av_0T(psi_t, ns[core_index]))

plt.close()
plt.figure()

plt.plot(ts_fs,nvas,label="n(imp) lanczos sci")
plt.plot(ts_fs,nvas_ex,label="n(imp) exact")
plt.plot(ts_fs,ncas,label="n(core) lanczos sci")
plt.plot(ts_fs,ncas_ex,label="n(core) exact")

plt.xlabel("t (fs)")
plt.legend()
name = "comp_Nb="+str(Nb)+".png"
plt.savefig(name)
plt.show()

