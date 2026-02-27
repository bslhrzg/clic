

import numpy as np
from math import sqrt, pi
from sympy.physics.wigner import gaunt

def cart_to_sph_pol(n):
    """
    Cartesian polarization vector -> spherical components ntilde[q].
    Convention:
      ntilde[+1] = (-nx + i ny)/sqrt(2)
      ntilde[ 0] =  nz
      ntilde[-1] = ( nx + i ny)/sqrt(2)
    """
    nx, ny, nz = n
    nx = complex(nx); ny = complex(ny); nz = complex(nz)
    return {
        +1: (-nx + 1j*ny)/sqrt(2.0),
         0: nz,
        -1: ( nx + 1j*ny)/sqrt(2.0),
    }

def gaunt_C_racah(l2, m2, q, l1, m1, prec=32):
    """
    Angular integral <l2 m2 | C^{(1)}_q | l1 m1> where
    C^{(k)}_q = sqrt(4*pi/(2k+1)) Y_{kq}.
    Uses Y*_{lm} = (-1)^m Y_{l,-m}.
    """
    k = 1
    # integral: ∫ dΩ Y*_{l2 m2} C^{(1)}_q Y_{l1 m1}
    # = sqrt(4π/3) (-1)^{m2} ∫ dΩ Y_{l2,-m2} Y_{1,q} Y_{l1,m1}
    val = sqrt(4*pi/(2*k+1)) * ((-1)**m2) * gaunt(l2, k, l1, -m2, q, m1, prec=prec)
    return complex(val.evalf(prec))

def dipole_elements_gaunt(l1, l2, pol, pol_basis="cart", prec=32, include_spin=False):
    """
    Return dipole angular matrix elements D[m2,m1] for l1 -> l2:
        D_{m2,m1} = sum_q ntilde[q] <l2 m2 | C^{(1)}_q | l1 m1>
    as a dict {(m2,m1): complex}.
    
    pol:
      - if pol_basis="cart": (nx,ny,nz) (real or complex)
      - if pol_basis="sph":  dict {+1:..., 0:..., -1:...} giving ntilde[q]
    """
    # selection rule in l:
    if abs(l2 - l1) != 1:
        return {}  # dipole forbidden
    
    if pol_basis == "cart":
        ntilde = cart_to_sph_pol(pol)
    elif pol_basis == "sph":
        ntilde = {+1: complex(pol.get(+1, 0.0)),
                  0: complex(pol.get(0, 0.0)),
                 -1: complex(pol.get(-1, 0.0))}
    else:
        raise ValueError("pol_basis must be 'cart' or 'sph'")

    D = {}
    for m2 in range(-l2, l2+1):
        for m1 in range(-l1, l1+1):
            # Δm = q constraint already enforced by gaunt vanishing, but cheap to skip
            amp = 0.0 + 0.0j
            for q in (-1, 0, +1):
                if m2 != m1 + q:
                    continue
                amp += ntilde[q] * gaunt_C_racah(l2, m2, q, l1, m1, prec=prec)
            if amp != 0.0j:
                D[(m2, m1)] = amp

    if include_spin:
        # make it spin-diagonal blocks: key (m2, s2, m1, s1)
        Dspin = {}
        for (m2, m1), v in D.items():
            for s in (0, 1):
                Dspin[(m2, s, m1, s)] = v
        return Dspin

    return D






#######################################################

def gauntC(k, l, m, lp, mp, prec=16):
    """
    return "nonvanishing" Gaunt coefficients of
    Coulomb interaction expansion.
    """
    c = sqrt(4 * pi / (2 * k + 1)) * (-1) ** m * gaunt(l, k, lp, -m, m - mp, mp, prec=prec)
    return float(c)


def getDipoleOperator(l1, l2, n):
    r"""
    Return dipole transition operator :math:`\hat{T}`.

    Transition between states of different angular momentum,
    defined by the keys in the nBaths dictionary.

    Parameters
    ----------
    nBaths : Ordered dict
        int : int,
        where the keys are angular momenta and values are number of bath states.
    n : list
        polarization vector n = [nx,ny,nz]

    """
    tOp = {}
    nDict = {-1: (n[0] + 1j * n[1]) / sqrt(2), 0: n[2], 1: (-n[0] + 1j * n[1]) / sqrt(2)}
    # Angular momentum
    for m in range(-l2, l2 + 1):
        for mp in range(-l1, l1 + 1):
            for s in range(2):
                if abs(m - mp) <= 1:
                    # See Robert Eder's lecture notes:
                    # "Multiplets in Transition Metal Ions"
                    # in Julich school.
                    # tij = d*n*c1(l=2,m;l=1,mp),
                    # d - radial integral
                    # n - polarization vector
                    # c - Gaunt coefficient
                    tij = gauntC(k=1, l=l2, m=m, lp=l1, mp=mp, prec=16)
                    tij *= nDict[m - mp]
                    if np.abs(tij) > 0:
                        tOp[(m, mp)] = tij
    return tOp


D = dipole_elements_gaunt(l1=3, l2=2, pol=(0,0,1), pol_basis="cart")
print(D)

D = getDipoleOperator(3, 2, [0,0,1])
print(D)