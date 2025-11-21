import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from clic_clib import get_creation_operator, get_annihilation_operator
from clic import * 
import matplotlib.pyplot as plt 
from utils import *

np.set_printoptions(precision=3, suppress=True, linewidth=300)


print("*"*42)
#----------------------------------------------------------
# SIAM 
#----------------------------------------------------------

Nb=3
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




###########
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

g_xas = ED_spectra(op_xas_alpha,psi0,e0,H,ws,eta)
plt.plot(ws,-np.imag(g_xas))
plt.show()

#assert 1==0
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
dt_code = 1        # time step in code unit


Ttot_fs_desired = 10
Ttot_code = Ttot_fs_desired / code_to_fs   # 
Nsteps  = int(Ttot_code / dt_code)         #

ts_code = np.arange(Nsteps) * dt_code          # times in code units
ts_fs   = ts_code * code_to_fs                 # same times in fs for plotting

print(f"T_total_fs = {ts_fs[-1]}, Nsteps = {Nsteps}")

E0=0.05
omega_pump = 0.5
sigma_pulse = 1 / code_to_fs
t0_pump = Ttot_code / 5
print(f"sigma_pulse = {sigma_pulse}, omega_pump = {omega_pump}")

E0s=[]
pump_envelope_t = []




nvas = []
nvbs = []
ncas = []
ncbs = []


psi = psi0.copy()
psi_t_list = []
nvas, nvbs, ncas, ncbs = [], [], [], []

for n in range(Nsteps-1):

    t_mid = ts_code[n] + 0.5 * dt_code

    #h0_t, Et = h0_pump_t(t_mid, pump_xas, E0, omega_pump, sigma_pulse,t0_pump)
    h0_t, Et = h0_pump_t(t_mid, pump_bath, E0, omega_pump, sigma_pulse,t0_pump)

    pump_envelope_t.append(Et)
    hop_t = h0_to_hop(h0_t, c, c_dag).toarray()
    hop_t = hop_t[np.ix_(fcibasis, fcibasis)]
    Ht = H + hop_t

    # One time step
    U_step = expm(-1j * dt_code * Ht)
    psi = U_step @ psi
    psi /= np.linalg.norm(psi)

    e0t = psi.T.conj() @ (Ht @ psi)
    E0s.append(e0t)

    # Measure
    nvas.append(get_av_0T(psi, ns[valence_index]))
    nvbs.append(get_av_0T(psi, ns[valence_index+M]))
    ncas.append(get_av_0T(psi, ns[core_index]))
    ncbs.append(get_av_0T(psi, ns[core_index+M]))
    psi_t_list.append(psi.copy())

print(f"nvas[:3] = {nvas[:3]}")
print(f"nvas[-3:-1] = {nvas[-3:-1]}")

print(f"nvbs[:3] = {nvbs[:3]}")
print(f"nvbs[-3:-1] = {nvbs[-3:-1]}")

print(f"ncas[:3] = {ncas[:3]}")
print(f"ncas[-3:-1] = {ncas[-3:-1]}")

print(f"ncbs[:3] = {ncbs[:3]}")
print(f"ncbs[-3:-1] = {ncbs[-3:-1]}")


plt.close()
plt.figure()
plt.plot(ts_fs[1:],E0s-E0s[0], label = "E0(t) - E0(0)")
plt.plot(ts_fs[1:],pump_envelope_t,label = "envelope")

plt.plot(ts_fs[1:],nvas,label="nvas")
#plt.plot(ts,nvbs,label="nvbs")
plt.plot(ts_fs[1:],ncas,label="ncas")
#plt.plot(ts,ncbs,label="ncbs")

plt.xlabel("t")
plt.legend()
plt.show()

