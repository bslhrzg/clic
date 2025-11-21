import numpy as np
from scipy.sparse import csr_matrix
from clic import * 


def h0U_siam_half_filling(Nb,u):

    Nimp = 1
    scheme = ["metal"]
    lambda_soc = 0.0
    dc = u/2
    eps = [-dc]

    D=0.1
    Gamma0=0.1
    gap=1.0
    h0_0 = build_impurity_model(Nimp,Nb,scheme,eps,lambda_soc,
                            D=D, Gamma0=Gamma0, gap=gap)

    NF = np.shape(h0_0)[0]
    M = NF // 2


    U_0 = build_U_tensor(Nimp, M,  U=u, J=0)
    h0_0,_ = transform_integrals_interleaved_to_alphafirst(h0_0, U_0, M)

    return h0_0,U_0

def h0_to_hop(h0,c,c_dag):

    NF,_ = h0.shape 
    dim = 2**NF
    hop = csr_matrix((dim, dim))
    for i in range(NF):
        for j in range(NF):
            hij = h0[i,j]
            if abs(hij) > 0 :
                hop += hij * c_dag[i] @ c[j]

    return hop

def U_to_Uop(U,c,c_dag):

    NF,_,_,_ = U.shape 
    dim = 2**NF
    Uop = csr_matrix((dim, dim))
    for i in range(NF):
        for j in range(NF):
            for k in range(NF):
                for l in range(NF):
                    u=U[i,j,k,l]
                    if abs(u) > 0:
                        Uop += u/2 * c_dag[i] @ c_dag[j] @ c[l] @ c[k]

    return Uop

def add_core_site(h0, U, e_c, U_dc):
    """
    Extend (h0, U) by adding one core site with onsite energy e_c and
    density-density interaction U_dc with the impurity site.

    Input: 
        h0 : (NF x NF) one-body matrix, alpha-first ordering
        U  : (NF x NF x NF x NF) two-body tensor, with U[i,j,k,l] giving
             c†_i c†_j c_l c_k term
        e_c : float, core onsite energy (same for alpha/beta)
        U_dc : float, density-density interaction with impurity

    Output:
        h0_new, U_new
    """

    NF_old = h0.shape[0]
    assert NF_old % 2 == 0
    M_old = NF_old // 2

    M_new = M_old + 1
    NF_new = 2*M_new

    # New physical orbital index (alpha, beta)
    core_alpha = M_new-1            # physical orbit index M_old in alpha sector
    core_beta  = core_alpha + M_new # physical orbit index M_old in beta sector after extension

    # Build extended h0
    h0_new = np.zeros((NF_new, NF_new), dtype=h0.dtype)
    
    iold_a = [i for i in range(M_old)]
    iold_b = [i+M_old for i in range(M_old)]

    h0_old_a = h0[np.ix_(iold_a,iold_a)]
    h0_old_b = h0[np.ix_(iold_b,iold_b)]

    iold_in_new_a = [i for i in range(M_old)]
    iold_in_new_b = [i+M_new for i in range(M_old)]

    h0_new[np.ix_(iold_in_new_a,iold_in_new_a)] = h0_old_a
    h0_new[np.ix_(iold_in_new_b,iold_in_new_b)] = h0_old_b


    # onsite core energy (alpha and beta)
    h0_new[core_alpha, core_alpha] = e_c
    h0_new[core_beta, core_beta]   = e_c

    # Build extended U
    U_new = np.zeros((NF_new, NF_new, NF_new, NF_new), dtype=U.dtype)
    old_in_new = iold_in_new_a + iold_in_new_b
    U_new[np.ix_(old_in_new,old_in_new,old_in_new,old_in_new)] = U



    # impurity spin orbitals (first site)
    imp_alpha = 0
    imp_beta  = imp_alpha + M_new  # first beta index

    # Add density-density U_dc * n_dσ n_cσ'
    # Density-density convention: U_dc * c†_dσ c†_cσ' c_cσ' c_dσ   (four terms)

    pairs = [
        (imp_alpha, core_alpha),
        (imp_alpha, core_beta),
        (imp_beta,  core_alpha),
        (imp_beta,  core_beta),
    ]

    for d, c in pairs:
        # (d,c | d,c)
        U_new[d, c, d, c] += U_dc
        U_new[c, d, d, c] += U_dc  # symmetry if needed (depends on your internal symmetrisation policy)

    return h0_new, U_new

def get_av_0T(psi,O):
    Opsi = O @ psi
    av=np.dot(psi.conj().T,Opsi)
    return av




def ED_spectra(op,psi,e,H,ws,eta):

        dim = np.shape(H)[0]
        I = np.eye(dim)
        psi_right = op @ psi
        psi_left = psi_right.conj().T

        g = np.zeros(len(ws),dtype=complex)

        for (i,w) in enumerate(ws):
            resolvent = np.linalg.inv((w + 1j*eta + e)*I - H)
            right = resolvent @ psi_right
            g[i] = psi_left @ right
            
        return g 




def XAS_core_val(core_index,valence_index,NF):
    """
    c^+_{val} c_{core} + c^+_{core} c_{val}
    """

    M = NF // 2
    core_a = core_index
    core_b = core_index + M 
    val_a = valence_index
    val_b = valence_index + M 

    h0 = np.zeros((NF,NF))

    h0[core_a,val_a] = 1 
    h0[val_a,core_a] = 1 
    h0[core_b,val_b] = 1 
    h0[val_b, core_b] = 1

    return h0


def XAS_core_cond(core_index,bath_indexes,NF):
    """
    c^+_{val} c_{core} + c^+_{core} c_{val}
    """

    M = NF // 2
    core_a = core_index
    core_b = core_index + M 

    h0 = np.zeros((NF,NF))

    for bi in bath_indexes:
        bi_a = bi
        bi_b = bi + M 

        h0[core_a,bi_a] = 1 
        h0[bi_a,core_a] = 1 
        h0[core_b,bi_b] = 1 
        h0[bi_b, core_b] = 1
    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    return h0


def random_quench(NF):

    M = NF//2
    rand_a = np.random.random((M,M))
    rand_a = rand_a + rand_a.T

    h0 = np.zeros((NF,NF))
    h0[:M,:M] = rand_a 
    h0[M:,M:] = rand_a

    h0 = np.ascontiguousarray(h0, dtype=np.complex128)
    return h0





def h0_pump_t(t,pump,E0,omega_pump,sigma,t0=0):
    """
    Define a gaussian pulse of freq omega_pump,
    duration sigma, and strength E0 
    the pump is a one particle hamiltonian, eg a XAS operator, 
    return the one particle pump times the pulse at time t-t0
    """

    if (t-t0) >= 0:
        Et = E0 * np.exp(-(t-t0)**2 / (2*(sigma)**2) ) 
        Et *= np.cos(omega_pump * t) 
    else :
        Et = 0

    h_pump_t = Et * pump 
    return h_pump_t,Et



