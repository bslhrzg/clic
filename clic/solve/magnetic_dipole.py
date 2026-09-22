"""Intra-atomic magnetic dipole T = s - 3 n (n.s), in units of hbar.

The shell has a common normalized radial function and n = r / |r|.  Only
the angular integral remains.  This is Quanty's T convention, without a
Bohr magneton or the XMCD sum-rule factor 7:
https://www.quanty.org/documentation/standard_operators/magnetic_dipole/tz

CLIC's one_rdm returns gamma[i,j] = <c_j^dagger c_i>.  In the current
backend, dot conjugates the smaller wavefunction (the argument on ties).
For each single c_i^dagger c_j application in one_rdm this is the applied
state, so wf.dot(phi) evaluates <phi|wf>.  Contract as Tr(gamma O), including
complex spin/orbital coherences, without an additional transpose.
"""

from functools import lru_cache

import numpy as np
from sympy.physics.wigner import gaunt


def _shell_l(n_orbitals):
    if n_orbitals < 1 or int(n_orbitals) != n_orbitals or n_orbitals % 2 != 1:
        raise ValueError("Magnetic dipole requires a complete shell with 2*l+1 orbitals")
    return (int(n_orbitals) - 1) // 2


@lru_cache(maxsize=None)
def gaunt_quadrupole_matrices(l):
    """Return Q[a,b,m,m'] = <lm|delta_ab - 3 n_a n_b|lm'>.

    Cartesian axes are x,y,z; m increases from -l to l.  The cached result
    is read-only.  Products n_a n_b are projected together, not obtained by
    multiplying separately projected direction operators.
    """
    if int(l) != l or l < 0:
        raise ValueError("l must be a nonnegative integer")
    l = int(l)
    n = 2 * l + 1
    c = {}
    for q in range(-2, 3):
        matrix = np.zeros((n, n), dtype=np.complex128)
        for j, mp in enumerate(range(-l, l + 1)):
            m = mp + q
            if -l <= m <= l:
                matrix[m + l, j] = (
                    (-1.0)**m * np.sqrt(4 * np.pi / 5)
                    * float(gaunt(l, 2, l, -m, q, mp))
                )
        c[q] = matrix

    qmat = np.zeros((3, 3, n, n), dtype=np.complex128)
    a = np.sqrt(3 / 2)
    qmat[0, 0] = c[0] - a * (c[-2] + c[2])
    qmat[1, 1] = c[0] + a * (c[-2] + c[2])
    qmat[2, 2] = -2 * c[0]
    qmat[0, 1] = qmat[1, 0] = 1j * a * (c[2] - c[-2])
    qmat[0, 2] = qmat[2, 0] = -a * (c[-1] - c[1])
    qmat[1, 2] = qmat[2, 1] = -1j * a * (c[-1] + c[1])
    qmat.setflags(write=False)
    return qmat


def _validate_shell_components(l_ops, s_ops, l, dim):
    """Reject projected/mixed shells, for which products of L are insufficient."""
    eye = np.eye(dim)

    def agrees(a, b):
        return np.allclose(a, b, atol=1e-8, rtol=1e-8)

    valid = all(op.shape == (dim, dim) for op in (*l_ops, *s_ops))
    if valid:
        valid = all(agrees(op, op.conj().T) for op in (*l_ops, *s_ops))
        valid &= agrees(sum(op @ op for op in l_ops), l * (l + 1) * eye)
        valid &= agrees(sum(op @ op for op in s_ops), 0.75 * eye)
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for components in (l_ops, s_ops):
                valid &= agrees(
                    components[a] @ components[b] - components[b] @ components[a],
                    1j * components[c],
                )
        valid &= all(agrees(lo @ so, so @ lo) for lo in l_ops for so in s_ops)
    if not valid:
        raise ValueError(
            "Magnetic dipole requires L and S for a complete single-l shell in "
            "the same basis and axes; projected or mixed shells need explicit "
            "projected magnetic-dipole matrices"
        )


def get_1p_magnetic_dipole_matrices(n_orbitals, *, l_components=None, s_components=None):
    """Return Tx, Ty, Tz as Hermitian single-particle matrices.

    By default use spherical AlphaFirst ordering, m=-l,...,l within each
    spin block.  Alternatively supply all three physical L and S components
    in the solver basis (as in RSPT).  For a complete shell the exact identity

        Q_ab = 3 ({L_a,L_b} - 2 delta_ab l(l+1)/3) / ((2l-1)(2l+3))

    transports the Gaunt tensor without reconstructing a basis rotation.
    Its normalization is taken from the Gaunt Qzz matrix below.  These are
    single-particle matrix products, not products of many-body observables.
    """
    l = _shell_l(n_orbitals)
    n_orbitals = 2 * l + 1
    qmat = gaunt_quadrupole_matrices(l)
    if l_components is None and s_components is None:
        spin = (
            np.array([[0, 1], [1, 0]]) / 2,
            np.array([[0, -1j], [1j, 0]]) / 2,
            np.diag([0.5, -0.5]),
        )
        return tuple(sum(np.kron(spin[b], qmat[a, b]) for b in range(3))
                     for a in range(3))
    if l_components is None or s_components is None:
        raise ValueError("Supply both l_components and s_components")
    if len(l_components) != 3 or len(s_components) != 3:
        raise ValueError("Supply x, y, z components for both L and S")
    l_ops = tuple(np.asarray(op, dtype=np.complex128) for op in l_components)
    s_ops = tuple(np.asarray(op, dtype=np.complex128) for op in s_components)
    dim = 2 * n_orbitals
    _validate_shell_components(l_ops, s_ops, l, dim)
    if l == 0:
        return tuple(np.zeros((dim, dim), dtype=np.complex128) for _ in range(3))

    # Match <l,l|Qzz|l,l> to the equivalent rank-2 tensor of L.
    factor = qmat[2, 2, -1, -1].real / (2 * l*l - 2 * l*(l + 1) / 3)
    result = []
    for a in range(3):
        t = np.zeros((dim, dim), dtype=np.complex128)
        for b in range(3):
            qab = l_ops[a] @ l_ops[b] + l_ops[b] @ l_ops[a]
            if a == b:
                qab = qab - (2 * l * (l + 1) / 3) * np.eye(dim)
            t += factor * qab @ s_ops[b]
        result.append(t)
    return tuple(result)


def expect_from_rdm(rdm, operator):
    """Return Tr(gamma O), with gamma[i,j]=<c_j^dagger c_i> from one_rdm."""
    rdm = np.asarray(rdm)
    operator = np.asarray(operator)
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1] or operator.shape != rdm.shape:
        raise ValueError("Density and operator must be square matrices of the same shape")
    if not np.all(np.isfinite(rdm)) or not np.all(np.isfinite(operator)):
        raise ValueError("Density and operator must be finite")
    if not np.allclose(rdm, rdm.conj().T, atol=1e-10, rtol=1e-8):
        raise ValueError("Density matrix must be Hermitian")
    if not np.allclose(operator, operator.conj().T, atol=1e-10, rtol=1e-8):
        raise ValueError("Observable matrix must be Hermitian")
    value = np.einsum("ij,ji->", rdm, operator)
    if abs(value.imag) > 1e-8 * max(1, abs(value.real)):
        raise ValueError("Hermitian observable has a complex expectation")
    return float(value.real)


def validate_rdm_observables(rdm, l_components, s_components, reference, *, atol=1e-8):
    """Check occupation and all L/S components against independent CLIC values.

    reference must contain occ, Lx/Ly/Lz, Sx/Sy/Sz.  The returned moments
    mS_z_muB=-2<Sz> and mL_z_muB=-<Lz> are numerical values in Bohr magnetons.
    The occupation reference is specific to the state being analyzed.
    """
    values = {"occ": expect_from_rdm(rdm, np.eye(len(rdm)))}
    for label, components in (("L", l_components), ("S", s_components)):
        for axis, op in zip("xyz", components):
            values[label + axis] = expect_from_rdm(rdm, op)
    for name, value in values.items():
        if not np.isclose(value, reference[name], atol=atol, rtol=1e-8):
            raise ValueError(
                f"RDM validation failed for {name}: density gives {value}, "
                f"CLIC reference gives {reference[name]}; check the density "
                "transpose, orbital/spin ordering, and basis transformation"
            )
    values["mS_z_muB"] = -2 * values["Sz"]
    values["mL_z_muB"] = -values["Lz"]
    return values
