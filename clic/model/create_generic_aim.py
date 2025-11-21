# clic/create_generic_aim.py

import numpy as np
import matplotlib.pyplot as plt
from clic.basis.basis_1p import umo2so
# The idea here is to define a generic anderson impurity model 
# with some constraints: the hybridization is always diagonal
#

# ------------------------------------------------------------------
# Here we define two possibilities for the bath :
# semicircle_bath for metal like hybridization
# mott_two_semicircle_bath for mott like hybridization 



def semicircle_bath(nb, D=1.0, Gamma0=0.2, center=0.0,
                    plot=False, ax=None, n_omega=401):
    """
    Discretize a semicircular hybridization:
        Gamma(omega) = Gamma0 * sqrt(1 - ((omega - center)/D)**2)
    on nb equally spaced bath levels within the support.

    Parameters
    ----------
    nb : int
        Number of bath levels.
    D : float
        Half-bandwidth.
    Gamma0 : float
        Overall hybridization scale.
    center : float
        Center of the semicircle.
    plot : bool, optional
        If True, plot the continuous Gamma(omega) together
        with the discrete bath representation.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None and plot=True, a new figure is created.
    n_omega : int, optional
        Number of points for the continuous Gamma(omega) curve.

    Returns
    -------
    e_bath : (nb,) array
        Bath on-site energies.
    V_bath : (nb,) array
        Hybridization amplitudes to approximate Gamma(omega).
    """
    if nb <= 0:
        return np.array([]), np.array([])

    # linear mesh strictly inside [center - D, center + D]
    e_bath = np.linspace(center - D, center + D, nb, endpoint=False) + D / nb
    delta_eps = 2 * D / nb   # uniform spacing

    x = (e_bath - center) / D
    Gamma = Gamma0 * np.sqrt(np.clip(1.0 - x**2, 0.0, None))

    # Discrete hybridization amplitudes
    V_bath = np.sqrt(Gamma * delta_eps / np.pi)

    if plot:
        if ax is None:
            fig, ax = plt.subplots()

        # Continuous hybridization
        omega = np.linspace(center - D, center + D, n_omega)
        xw = (omega - center) / D
        Gamma_exact = Gamma0 * np.sqrt(np.clip(1.0 - xw**2, 0.0, None))

        ax.plot(omega, Gamma_exact, label=r"$\Gamma(\omega)$ (continuous)")

        # Discrete representation: in the continuum limit
        # Gamma(omega) ≈ π * Σ_i V_i^2 δ(ω-ε_i) / Δε
        Gamma_disc = np.pi * V_bath**2 / delta_eps

        # Sticks for each bath level
        markerline, stemlines, baseline = ax.stem(
            e_bath, Gamma_disc, label="discrete bath"
        )

        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$\Gamma(\omega)$")
        ax.set_xlim(center - D * 1.05, center + D * 1.05)
        ax.legend()
        ax.set_title("Semicircular hybridization and discrete bath")
        plt.savefig("hyb.png")

    return e_bath, V_bath

def mott_two_semicircle_bath(nb, D=1.0, Gamma0=0.2, gap=1.0):
    """
    Two semicircles separated by a gap around the Fermi level (0).

    Lower band: center at -(gap/2 + D)
    Upper band: center at +(gap/2 + D)
    """
    if nb <= 0:
        return np.array([]), np.array([])

    nb1 = nb // 2
    nb2 = nb - nb1

    center_lower = -(gap / 2 + D)
    center_upper = +(gap / 2 + D)

    e1, V1 = semicircle_bath(nb1, D=D, Gamma0=Gamma0, center=center_lower)
    e2, V2 = semicircle_bath(nb2, D=D, Gamma0=Gamma0, center=center_upper)

    e_bath = np.concatenate([e1, e2])
    V_bath = np.concatenate([V1, V2])

    # sort, just to keep levels ordered in energy
    idx = np.argsort(e_bath)
    return e_bath[idx], V_bath[idx]


def get_siam_h(nb, bath_scheme="metal",
               D=1.0, Gamma0=0.2, gap=1.0):
    """
    Constructs the single particle Hamiltonian matrix for one spin channel.

    bath_scheme: "metal" (single semicircle) or "mott" (two semicircles with a gap)
    """
    M = nb + 1
    h = np.zeros((M, M))

    if bath_scheme == "metal":
        e_bath, V_bath = semicircle_bath(nb, D=D, Gamma0=Gamma0, center=0.0)
    elif bath_scheme == "mott":
        e_bath, V_bath = mott_two_semicircle_bath(nb, D=D, Gamma0=Gamma0, gap=gap)
    else:
        raise ValueError(f"Unknown bath_scheme: {bath_scheme}")

    if nb > 0:
        np.fill_diagonal(h[1:, 1:], e_bath)
        h[0, 1:] = V_bath
        h[1:, 0] = V_bath

    return h, e_bath, V_bath

# this only define the coupling and bath hamiltonian given a list of bath_scheme
# so we can have one impurity orbital coupling to a metal like hybridization 
# and another one coupling to a mott like one
# the impurity block is left to be 0
# Note : this is spinless
def get_multiorb_diag_imp(Nimp, Nb, bath_scheme=None,
                          D=1.0, Gamma0=0.2, gap=1.0):

    M = Nimp * (1 + Nb)
    himp = np.zeros((M, M))

    e_d = e_bath = V_bath = None

    for i in range(Nimp):
        if bath_scheme is not None and len(bath_scheme) != Nimp:
            raise ValueError("len(bath_scheme) must equal Nimp")            
        else :
            i_scheme = bath_scheme[i]

        h, e_bath, V_bath = get_siam_h(
            Nb, bath_scheme=i_scheme,
            D=D, Gamma0=Gamma0, gap=gap
        )

        indb = np.arange(Nimp + i * Nb, Nimp + (i + 1) * Nb)
        himp[i, indb] = V_bath
        himp[indb, i] = V_bath
        himp[np.ix_(indb, indb)] = np.diag(e_bath)

    return himp

# ------------------------------------------------------------------
# Here we get the kanamori hamiltonian parameters
def build_kanamori_params(Nimp, U, J,
                          rotational_invariant=True,
                          Uprime=None, Jpair=None):
    """
    Return a dict with Kanamori parameters for an Nimp orbital impurity.

    H_int = U * n_a↑ n_a↓
          + U' * n_a↑ n_b↓
          + (U' - J) * n_aσ n_bσ
          - J * spin flip
          + Jpair * pair hopping

    with a != b implied in the sums.

    If rotational_invariant is True, use the standard relations
    U' = U - 2 J and Jpair = J.
    """
    if rotational_invariant:
        Uprime = U - 2.0 * J
        Jpair = J
    else:
        if Uprime is None or Jpair is None:
            raise ValueError("Need Uprime and Jpair when rotational_invariant=False")

    params = dict(
        Nimp=int(Nimp),
        U=float(U),
        J=float(J),
        Uprime=float(Uprime),
        Jpair=float(Jpair),
    )
    return params

# We construct U from above
def build_U_matrix_kanamori(Nimp, U, J,
                            rotational_invariant=True,
                            Uprime=None, Jpair=None,
                            symmetrize=True):
    """
    Build the 4-index orbital Coulomb tensor U_{a b c d} that reproduces
    a Kanamori interaction with parameters (U, J, U', Jpair).

    H_int = 1/2 sum_{a b c d} sum_{σ σ'}
              U_{a b c d} c†_{aσ} c†_{bσ'} c_{dσ'} c_{cσ}

    For real integrals this yields:
      U_{a a a a} = U            (intraorbital)
      U_{a b a b} = U'           (interorbital direct)
      U_{a b b a} = J            (exchange)
      U_{a a b b} = Jpair        (pair hopping)
    for all a != b, plus symmetry-related copies.

    Parameters
    ----------
    Nimp : int
        Number of impurity orbitals.
    U, J : float
        Kanamori parameters.
    rotational_invariant : bool
        If True, set U' = U - 2 J and Jpair = J.
    Uprime, Jpair : float or None
        If rotational_invariant is False, you must provide these.
    symmetrize : bool
        If True, enforce U_{a b c d} = U_{b a d c} and U_{a b c d} = U_{c d a b}.
        For small Nimp this is cheap and gives a clean tensor.

    Returns
    -------
    Umat : ndarray, shape (Nimp, Nimp, Nimp, Nimp)
        Real 4-index tensor U_{a b c d}.
    """
    Nimp = int(Nimp)
    U = float(U)
    J = float(J)

    if rotational_invariant:
        Uprime = U - 2.0 * J
        Jpair = J
    else:
        if Uprime is None or Jpair is None:
            raise ValueError("Provide Uprime and Jpair if rotational_invariant=False")
        Uprime = float(Uprime)
        Jpair = float(Jpair)

    Umat = np.zeros((Nimp, Nimp, Nimp, Nimp), dtype=float)

    # intraorbital U
    for a in range(Nimp):
        Umat[a, a, a, a] = U

    # interorbital pieces
    for a in range(Nimp):
        for b in range(Nimp):
            if a == b:
                continue

            # direct interorbital U' (density-density)
            Umat[a, b, a, b] = Uprime

            # exchange J (spin flip type)
            Umat[a, b, b, a] = J

            # pair hopping Jpair
            Umat[a, a, b, b] = Jpair

    if symmetrize:
        # enforce basic symmetries:
        # U_{a b c d} = U_{b a d c} = U_{c d a b}
        Usym = np.zeros_like(Umat)
        for a in range(Nimp):
            for b in range(Nimp):
                for c in range(Nimp):
                    for d in range(Nimp):
                        val = Umat[a, b, c, d]
                        val += Umat[b, a, d, c]
                        val += Umat[c, d, a, b]
                        val += Umat[d, c, b, a]
                        Usym[a, b, c, d] = val / 4.0
        Umat = Usym

    return Umat


# ------------------------------------------------------------------
# here we can construct an spin-orbit like hamiltonian
# internal helper: build effective L matrices in orbital space
def _leff_matrices(Nimp, mode):
    """
    Return (Lx, Ly, Lz) as Nimp x Nimp complex matrices.

    mode:
      - "l_eff_1_2":  Nimp must be 2, use j=1/2 Pauli representation
      - "l_eff_1":    Nimp must be 3, use j=1 representation (m=+1,0,-1)
      - "diag_lz":    generic fallback, Lz diagonal, Lx=Ly=0
    """
    Nimp = int(Nimp)

    if mode == "l_eff_1_2":
        if Nimp != 2:
            raise ValueError("l_eff_1_2 requires Nimp=2")
        # L = 1/2 * Pauli matrices
        Lx = 0.5 * np.array([[0, 1],
                             [1, 0]], dtype=complex)
        Ly = 0.5 * np.array([[0, -1j],
                             [1j, 0]], dtype=complex)
        Lz = 0.5 * np.array([[1, 0],
                             [0, -1]], dtype=complex)
        return Lx, Ly, Lz

    if mode == "l_eff_1":
        if Nimp != 3:
            raise ValueError("l_eff_1 requires Nimp=3")
        # standard l=1 representation, basis ordered as m=+1,0,-1
        l = 1.0
        Lp = np.zeros((3, 3), dtype=complex)

        # <m+1|L+|m> = sqrt(l(l+1) - m(m+1))
        # m = -1 -> 0 (index 2->1)
        m = -1.0
        val = np.sqrt(l * (l + 1.0) - m * (m + 1.0))
        Lp[1, 2] = val

        # m = 0 -> 1 (index 1->0)
        m = 0.0
        val = np.sqrt(l * (l + 1.0) - m * (m + 1.0))
        Lp[0, 1] = val

        Lm = Lp.conj().T
        Lx = 0.5 * (Lp + Lm)
        Ly = -0.5j * (Lp - Lm)
        Lz = np.diag([1.0, 0.0, -1.0])
        return Lx, Ly, Lz

    # fallback: simple diagonal Lz, Lx=Ly=0
    # assign "m" values symmetrically spaced
    m_vals = np.linspace((Nimp - 1) / 2.0, -(Nimp - 1) / 2.0, Nimp)
    Lz = np.diag(m_vals.astype(float)).astype(complex)
    Lx = np.zeros((Nimp, Nimp), dtype=complex)
    Ly = np.zeros((Nimp, Nimp), dtype=complex)
    return Lx, Ly, Lz


def build_effective_soc_matrix(Nimp, lambda_soc,
                               mode="auto"):
    """
    Build the local SOC matrix in the single particle basis |a,σ>.

    Dimension: 2*Nimp x 2*Nimp, ordered as
      i = 2*a + s,  a=0..Nimp-1, s=0 (up), 1 (down)

    H_SOC = λ sum_i (L_i ⊗ S_i)

    mode:
      - "none":      return zero matrix
      - "auto":      pick l_eff_1_2 for Nimp=2, l_eff_1 for Nimp=3,
                     else "diag_lz"
      - "l_eff_1_2": force 2 orbital pseudo l=1/2
      - "l_eff_1":   force 3 orbital pseudo l=1
      - "diag_lz":   only Lz nonzero
    """
    Nimp = int(Nimp)
    lambda_soc = float(lambda_soc)

    dim = 2 * Nimp
    H = np.zeros((dim, dim), dtype=complex)

    if lambda_soc == 0.0 or mode == "none":
        return H

    if mode == "auto":
        if Nimp == 2:
            mode_eff = "l_eff_1_2"
        elif Nimp == 3:
            mode_eff = "l_eff_1"
        else:
            mode_eff = "diag_lz"
    else:
        mode_eff = mode

    Lx, Ly, Lz = _leff_matrices(Nimp, mode_eff)

    # spin 1/2 matrices
    Sx = 0.5 * np.array([[0, 1],
                         [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j],
                         [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0],
                         [0, -1]], dtype=complex)

    # full SOC: λ sum_i L_i ⊗ S_i
    H += lambda_soc * np.kron(Lx, Sx)
    H += lambda_soc * np.kron(Ly, Sy)
    H += lambda_soc * np.kron(Lz, Sz)

    return H


# ------------------------------------------------------------------
# here we get the full impurity block 
# eps is to be given and control the degeneracy
def build_local_impurity_block(Nimp, eps,
                               lambda_soc=0.0,
                               soc_mode="auto"):
    """
    Build the local one body impurity block h_imp of size (2*Nimp, 2*Nimp).

    eps: array_like of length Nimp, orbital onsite energies ε_a.
         These are copied to both spins (spin independent CF).

    lambda_soc: strength of SOC (same λ as in build_effective_soc_matrix)
    soc_mode:   passed to build_effective_soc_matrix
    """
    Nimp = int(Nimp)
    eps = np.asarray(eps, dtype=float)
    if eps.shape != (Nimp,):
        raise ValueError(f"eps must have shape ({Nimp},), got {eps.shape}")

    dim = 2 * Nimp
    h_cf = np.zeros((dim, dim), dtype=complex)

    # diagonal crystal field, spin independent
    for a in range(Nimp):
        for s in range(2):
            i = 2 * a + s
            h_cf[i, i] = eps[a]

    H_soc = build_effective_soc_matrix(Nimp, lambda_soc, mode=soc_mode)

    return h_cf + H_soc


# ------------------------------------------------------------------
# The full constructor 
def build_impurity_model(Nimp_orb, Nb, bath_scheme, eps_imp, lambda_soc,
                         D=1.0, Gamma0=0.2, gap=1.0):

    # spinless multi-orbital + bath Hamiltonian
    h0_orb = get_multiorb_diag_imp(Nimp_orb, Nb,
                                   bath_scheme=bath_scheme,
                                   D=D, Gamma0=Gamma0, gap=gap)

    M_orb = h0_orb.shape[0]

    # spinful lifting with interleaved spin:
    # basis |α,σ> with index i = 2*α + s
    I2 = np.eye(2, dtype=complex)
    h0 = np.kron(h0_orb.astype(complex), I2)  # shape (2*M_orb, 2*M_orb)

    # local impurity block (CF + SOC) on the impurity orbitals only
    # size: 2 * Nimp_orb × 2 * Nimp_orb, in basis |a,σ>, a=0..Nimp_orb-1
    h_imp = build_local_impurity_block(Nimp_orb, eps_imp,
                                       lambda_soc=lambda_soc,
                                       soc_mode="auto")

    nloc = 2 * Nimp_orb
    # the impurity block in h0_orb is zero on-site (only hybridization lives outside),
    # so we can just add the local part here
    h0[:nloc, :nloc] += h_imp

    h0 = np.ascontiguousarray(h0, dtype=np.complex128)

    return h0


def build_U_tensor(Nimp, M_total_spatial,  U, J, Uprime=None, Jpair=None, symmetrize=True):


    U_imp_spatial = build_U_matrix_kanamori(Nimp, U, J,
                            rotational_invariant=True,
                            Uprime=Uprime, Jpair=Jpair,
                            symmetrize=symmetrize)
    
    U_imp = umo2so(U_imp_spatial,Nimp)

    M_total_spinfull = M_total_spatial*2

    # Pad the U matrix to the full size of the new hamiltonian
    # Assumes U_imp is dense (4-index tensor)
    U_0 = np.zeros((M_total_spinfull,)*4, dtype=U_imp.dtype)
    imp_size = U_imp.shape[0]
    halfimp = imp_size // 2

    impindex = [i for i in range(halfimp)] + [i for i in range(M_total_spatial, M_total_spatial+halfimp)]
    U_0[np.ix_(impindex, impindex, impindex, impindex)] = U_imp

    U_0 = np.ascontiguousarray(U_0, dtype=np.complex128)

    return U_0





test=False 

if test:
    Nimp = 2 
    Nb = 3
    scheme = ["metal", "mott"]
    h0 = get_multiorb_diag_imp(Nimp,Nb,bath_scheme=scheme)

    np.set_printoptions(precision=3, suppress=True, linewidth=300)

    print(h0)


    # 2 orbital model, degenerate, with SOC
    Nimp = 2
    eps = [0.0, 0.0]
    lambda_soc = 0.2

    h_imp_2 = build_local_impurity_block(Nimp, eps,
                                        lambda_soc=lambda_soc,
                                        soc_mode="auto")
    print(f"h_imp_2 = ")
    print(h_imp_2)


    kanamori_2 = build_kanamori_params(Nimp, U=4.0, J=0.8)
    U2 = build_U_matrix_kanamori(2, **{k:v for k,v in kanamori_2.items()
                                    if k in ("U","J")})
    print(f"kanamori_2={kanamori_2}")

    # 3 orbital model, t2g-like (l_eff = 1), no CF splitting yet
    Nimp = 3
    eps = [0.0, 0.0, 0.0]
    lambda_soc = 0.1

    h_imp_3 = build_local_impurity_block(Nimp, eps,
                                        lambda_soc=lambda_soc,
                                        soc_mode="auto")

    print(f"h_imp_3 = ")
    print(h_imp_3)

    kanamori_3 = build_kanamori_params(Nimp, U=5.0, J=0.7)
    U3 = build_U_matrix_kanamori(3, **{k:v for k,v in kanamori_3.items()
                                    if k in ("U","J")})
    print(f"kanamori_3={kanamori_3}")


    print("*"*42)
    Nimp = 3
    scheme = ["metal","metal","mott"]
    lambda_soc = 0.0
    Nb = 3
    h = build_impurity_model(Nimp,Nb,scheme,[0.0,-0.2,0.03],lambda_soc)
    print(f"h shape: {np.shape(h)}")
    print(np.real(h))
    print("*"*12)
    ia=[i for i in range(np.shape(h)[0]) if i%2 == 0]
    ia = np.ix_(ia,ia)
    print(np.real(h[ia]))