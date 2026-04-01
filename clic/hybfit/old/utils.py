# hybfit/utils.py

import numpy as np
import numpy.linalg as npl
from clic.io_clic.io_utils import vprint

def load_delta_from_files(file_re, file_im, col_re=5, col_im=5):
    """
    Loads a scalar hybridization from two files (real and imaginary parts)
    and promotes it to the standard matrix shape (N, 1, 1).

    Args:
        file_re (str): Path to the file with the real part.
        file_im (str): Path to the file with the imaginary part.
        col_re (int): Column index for the real part data.
        col_im (int): Column index for the imaginary part data.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - omega_grid (shape (N,))
            - delta_complex (shape (N, 1, 1))
    """
    data_re = np.loadtxt(file_re)
    data_im = np.loadtxt(file_im)
    omega_re, re_vals = data_re[:, 0], data_re[:, col_re]
    omega_im, im_vals = data_im[:, 0], data_im[:, col_im]

    if not np.allclose(omega_re, omega_im, rtol=1e-12, atol=1e-14):
        raise ValueError("Omega grids in real and imaginary files do not match")
    
    delta_scalar = (re_vals + 1j * im_vals).astype(np.complex128)
    # Promote to standard (N, M, M) shape with M=1
    delta_matrix = delta_scalar[:, np.newaxis, np.newaxis]
    
    return omega_re.astype(float), delta_matrix

def create_dummy_delta(omega, n_poles, m_orb, e_range=(-2.0, 2.0), eta=0.01):
    """
    Creates a dummy matrix-valued hybridization function from random poles and residues.

    Args:
        omega (np.ndarray): Frequency grid.
        n_poles (int): Number of poles to generate.
        m_orb (int): Orbital dimension (M).
        e_range (tuple): Energy range (min, max) for the poles.
        eta (float): Broadening to apply to the generated function.

    Returns:
        tuple[np.ndarray, np.ndarray, list]:
            - delta_complex (shape (N, M, M))
            - true_eps (shape (n_poles,))
            - true_R (list of M x M ndarrays)
    """
    eps = np.sort(np.random.uniform(e_range[0], e_range[1], size=n_poles))
    R = []
    for _ in range(n_poles):
        V = np.random.rand(m_orb, m_orb) + 1j * np.random.rand(m_orb, m_orb)
        R_j = V @ V.conj().T
        R.append(R_j)

    z = omega + 1j * eta
    delta = delta_from_poles(z, eps, R)
    return delta, eps, R

def delta_from_poles(z, eps, residues):
    """
    Reconstructs the hybridization function Δ(z) from its pole representation.
    Δ(z) = Σ_j R_j / (z - ε_j)

    Args:
        z (np.ndarray): Complex frequency grid (shape (N,)).
        eps (np.ndarray): Pole energies (shape (P,)).
        residues (list[np.ndarray]): List of P residue matrices, each of shape (M, M).

    Returns:
        np.ndarray: The complex hybridization function (shape (N, M, M)).
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=np.complex128))
    if len(eps) == 0:
        M = residues[0].shape[0] if len(residues) > 0 else 1
        out = np.zeros(z_arr.shape + (M, M), dtype=np.complex128)
        return out[0] if np.isscalar(z) else out

    M = residues[0].shape[0]
    out = np.zeros(z_arr.shape + (M, M), dtype=np.complex128)
    for j, ej in enumerate(eps):
        denom = (z_arr - ej)[..., None, None]
        out += residues[j][None, :, :] / denom
    return out[0] if np.isscalar(z) else out

def residues_to_bath(eps, R_list, tol=1e-12):
    """From pole energies eps and M×M residues R_list, build diagonal H_b and V (M×Nb)."""
    eps_out = []
    Vcols = []
    for e, R in zip(np.asarray(eps, float), R_list):
        R = 0.5*(R + R.conj().T)              # Hermitize
        w, U = npl.eigh(R)
        w = np.clip(w, 0.0, None)             # clip tiny negatives
        for lam, u in zip(w, U.T):
            if lam > tol:
                Vcols.append(np.sqrt(lam) * u) # coupling vector for this bath state
                eps_out.append(e)
    if Vcols:
        V = np.column_stack(Vcols).astype(np.complex128)
        H_b = np.diag(np.asarray(eps_out, float))
    else:
        M = R_list[0].shape[0]
        V = np.zeros((M, 0), dtype=np.complex128)
        H_b = np.zeros((0, 0), dtype=float)
    return H_b, V

def delta_from_bath(omega, H_b, V, eta=0.0):
    eps = np.diag(H_b)
    M = V.shape[0]
    Delta = np.zeros((len(omega), M, M), dtype=np.complex128)
    for i, w in enumerate(omega):
        g = 1.0/(w + 1j*eta - eps)            # bath resolvent in diagonal basis
        Delta[i] = V @ np.diag(g) @ V.conj().T
    return Delta

def rel_l2_error(delta_true, delta_model):
    """
    Calculates the relative L2 error between two matrix-valued functions
    using the Frobenius norm. Error = ||A-B||_F / ||A||_F.

    Args:
        delta_true (np.ndarray): The true hybridization (shape (N, M, M)).
        delta_model (np.ndarray): The model hybridization (shape (N, M, M)).

    Returns:
        float: The relative L2 error.
    """
    diff = delta_true - delta_model
    num = np.sum(np.abs(diff)**2)
    den = np.sum(np.abs(delta_true)**2)
    return np.sqrt(num / (den + 1e-300))

def cost_l2_integral(delta_true, delta_model, omega, weight='const'):
    """
    Calculates the integrated weighted L2 difference (chi-squared).
    χ² = ∫ dω W(ω) ||Δ_true(ω) - Δ_model(ω)||_F^2

    Args:
        delta_true (np.ndarray): The true hybridization (shape (N, M, M)).
        delta_model (np.ndarray): The model hybridization (shape (N, M, M)).
        omega (np.ndarray): Frequency grid for integration.
        weight (str): Weighting function type ('const' or 'inv2').

    Returns:
        float: The chi-squared value.
    """
    diff = delta_true - delta_model
    
    # Squared Frobenius norm at each omega point
    integrand = np.sum(np.abs(diff)**2, axis=(-2, -1))

    if weight == 'const':
        W = 1.0
    elif weight == 'inv2':
        # Use a small regularizer to avoid division by zero
        W = 1.0 / (omega**2 + 1e-4)
    else:
        raise ValueError(f"Unknown weight: {weight}")

    return np.trapz(W * integrand, omega)

def check_moment_conservation(omega, delta_true, eps_fit, R_fit, n_max=5, logfile=None):
    """
    Compares the power moments of the spectral function trace for the true
    hybridization vs. the fitted pole model.

    Args:
        omega (np.ndarray): Frequency grid.
        delta_true (np.ndarray): The true hybridization (shape (N, M, M)).
        eps_fit (np.ndarray): Fitted pole energies.
        R_fit (list[np.ndarray]): Fitted residue matrices.
        n_max (int): Maximum moment order to check.

    Returns:
        dict: A dictionary containing moments, errors, and pass/fail flags.
    """
    # Spectral density is the trace of the spectral function matrix
    rho_true = -np.imag(np.trace(delta_true, axis1=1, axis2=2)) / np.pi
    
    # Trapezoid weights for integration
    dx = np.diff(omega)
    w_trapz = np.empty_like(omega)
    w_trapz[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w_trapz[0] = 0.5 * dx[0]
    w_trapz[-1] = 0.5 * dx[-1]
    
    mu_grid = rho_true * w_trapz
    
    M_grid = np.array([np.sum(mu_grid * (omega**n)) for n in range(n_max + 1)])
    
    # Moments from the fitted model
    w_fit = np.array([np.real(np.trace(R)) for R in R_fit])
    M_fit = np.array([np.sum(w_fit * (eps_fit**n)) for n in range(n_max + 1)])
    
    abs_err = np.abs(M_grid - M_fit)
    rel_err = abs_err / (np.abs(M_grid) + 1e-300)
    
    results = {
        "M_grid": M_grid, "M_fit": M_fit,
        "abs_err": abs_err, "rel_err": rel_err
    }
    
    vprint(1,"\n--- Moment Conservation Check (Tr[-ImΔ/π]) ---",filename=logfile)
    vprint(1,"n  M_grid        M_fit         Abs_Err   Rel_Err",filename=logfile)
    vprint(1,"-" * 50,filename=logfile)
    for n in range(n_max + 1):
        vprint(1,f"{n:<2d} {M_grid[n]:<+12.6e}  {M_fit[n]:<+12.6e}  {abs_err[n]:.2e}    {rel_err[n]:.2e}",filename=logfile)
        
    return results