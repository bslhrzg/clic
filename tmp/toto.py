import numpy as np
import scipy as sc

from clic import * 

def tridiagonalize(eb, tns):
    assert len(eb) == tns.shape[0]
    block_size = tns.shape[1]

    v0, v0_tilde = sc.linalg.qr(tns, mode="economic")

    H = np.diag(eb)
    N = H.shape[0]
    Q = np.zeros((N, N), dtype=complex)
    q = np.zeros((2, N, block_size), dtype=complex)
    q[1, :, :block_size] = v0
    alphas = np.empty((N // block_size, block_size, block_size), dtype=complex)
    betas = np.zeros((N // block_size, block_size, block_size), dtype=complex)

    for i in range(N // block_size):
        wp = H @ q[1]
        alphas[i] = np.conj(q[1].T) @ wp
        wp -= q[1] @ alphas[i] + q[0] @ np.conj(betas[i - 1].T)
        wp -= Q @ np.conj(Q.T) @ wp
        Q[:, i * block_size : (i + 1) * block_size] = q[1]
        q[0] = q[1]
        q[1], betas[i] = np.linalg.qr(wp)

    return alphas, betas, v0_tilde


def edchains(vs, ebs):
    """
    Transform the bath geometry from a star into one or two auxilliary chains.
    The two chains correspond to the occupied and unoccupied parts of the spectra respectively.
    Returns the Hopping term from the impurity onto the chains and the chain bath Hamiltonian.
    Parameters:
    vs: np.ndarray((Neb, block_size)) - Hopping parameters for star geometry.
    ebs: np.ndarray((Neb)) - Bath energies for star geometry
    Returns:
    chain_v, H_bath_chain
    chain_v: np.ndarray((Neb_chain, block_size)) - Hopping parameters for chain geometry.
    H_bath_chain: np.ndarray((Neb_chain, Neb_chain)) - Hamiltonian describind the bath in chain geometry.
    """
    n_block_orb = vs.shape[1]
    n = sum(ebs < 0)
    sorted_indices = np.argsort(ebs)
    ebs = ebs[sorted_indices]
    vs = vs[sorted_indices]
    ebs[:n] = ebs[:n][::-1]
    vs[:n] = vs[:n][::-1]
    chain_eb, chain_v, v0_tilde = tridiagonalize(ebs[:n], vs[:n])
    chain_v_occ = np.zeros((len(chain_eb) * n_block_orb, n_block_orb), dtype=complex)
    H_bath_occ = np.zeros((len(chain_eb) * n_block_orb, len(chain_eb) * n_block_orb), dtype=complex)
    chain_v_occ[0:n_block_orb] = v0_tilde
    for i in range(0, len(chain_eb) - 1):
        H_bath_occ[i * n_block_orb : (i + 1) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb] = chain_eb[i]
        H_bath_occ[(i + 1) * n_block_orb : (i + 2) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb] = chain_v[i]
        H_bath_occ[i * n_block_orb : (i + 1) * n_block_orb, (i + 1) * n_block_orb : (i + 2) * n_block_orb] = np.conj(
            chain_v[i].T
        )
    H_bath_occ[-n_block_orb:, -n_block_orb:] = chain_eb[-1]
    if n < len(ebs):
        chain_eb, chain_v, v0_tilde = tridiagonalize(ebs[n:], vs[n:])
        chain_v_unocc = np.zeros((len(chain_eb) * n_block_orb, n_block_orb), dtype=complex)
        H_bath_unocc = np.zeros((len(chain_eb) * n_block_orb, len(chain_eb) * n_block_orb), dtype=complex)
        chain_v_unocc[0:n_block_orb] = v0_tilde
        for i in range(0, len(chain_eb) - 1):
            H_bath_unocc[i * n_block_orb : (i + 1) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb] = chain_eb[i]
            H_bath_unocc[(i + 1) * n_block_orb : (i + 2) * n_block_orb, i * n_block_orb : (i + 1) * n_block_orb] = (
                chain_v[i]
            )
            H_bath_unocc[i * n_block_orb : (i + 1) * n_block_orb, (i + 1) * n_block_orb : (i + 2) * n_block_orb] = (
                np.conj(chain_v[i].T)
            )
        H_bath_unocc[-n_block_orb:, -n_block_orb:] = chain_eb[-1]
    else:
        chain_v_unocc = np.zeros((0 * n_block_orb, n_block_orb), dtype=complex)
        H_bath_unocc = np.zeros((0 * n_block_orb, 0 * n_block_orb), dtype=complex)
    return (H_bath_occ[::-1, ::-1].copy(), chain_v_occ[::-1].copy()), (H_bath_unocc, chain_v_unocc)


nb = 5
M = 1 + nb
u = 1
mu = u/2
Nelec = M

e_bath = np.linspace(-2.0, 2.0, nb).astype(float)
if nb == 1:
    e_bath = np.array([0.0], dtype=float)

# shape must be (nb, block_size). For one orbital, block_size=1
V_bath = np.full((nb, 1), 0.1, dtype=complex)

h0, U_mat = get_impurity_integrals(M, u, e_bath, V_bath.ravel(), mu)

#h_new, C = get_double_chain_transform(np.real(h0[:M, :M]), u, Nelec//2)
h_new, C = get_natural_orbital_transform(np.real(h0[:M, :M]), u, Nelec//2)
print("h_new =")
print(np.round(np.real(C.conj().T @ h0[:M,:M] @ C), 3))

# call edchains with the 2D V_bath
(H_bath_occ, chain_v_occ), (H_bath_unocc, chain_v_unocc) = edchains(V_bath, e_bath)

print("occ chain shapes:", H_bath_occ.shape, chain_v_occ.shape)
print("unocc chain shapes:", H_bath_unocc.shape, chain_v_unocc.shape)

print(f"H_bath_occ = {H_bath_occ.real}")
print(f"chain_v_occ = {chain_v_occ.real}")
print(f"H_bath_unocc = {H_bath_unocc.real}")
print(f"chain_v_unocc = {chain_v_unocc.real}")