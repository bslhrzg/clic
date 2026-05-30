import numpy as np
from scipy import optimize
from clic.symmetries import symmetries
import numpy.linalg as npl
from clic.io_clic.io_utils import dump


def cost_function(params, n_b, ws, target_delta, eta, weight_str, real_couplings=True):
    e = params[:n_b]
    if real_couplings:
        v = params[n_b:2*n_b]  # Purely real couplings
    else:
        v = params[n_b:2*n_b] + 1j * params[2*n_b:3*n_b]
    
    model_delta = hybridization_model(e, v, ws, eta)
    difference = target_delta - model_delta
    
    if weight_str == "const": weight = 1.0
    elif weight_str == "inv2": weight = 1.0 / (ws ** 2 + 1e-2)
    elif weight_str == "inv" : weight = 1.0 / (np.abs(ws) + 1e-2)
    else: raise ValueError(f"Unknown weight: {weight_str}")
        
    return np.sum(weight * (np.abs(difference))**2)

def hybridization_model(e, v, ws, eta):
    return np.sum(np.abs(v)**2 / (ws[:, None] + 1j * eta - e), axis=1)

def lorentzian_convolution(ws, y, width):
    result = np.zeros_like(y, dtype=np.complex128)
    for i, omega_i in enumerate(ws):
        kernel = (1 / np.pi) * width / ((omega_i - ws)**2 + width**2)
        result[i] = np.trapezoid(y * kernel, ws)
    return result


def scalar_fit(ws, hyb_scalar, 
            n_b,
            eta_0,
            weight_func = 'const',
            broadening_Gamma = 0.00,
            bounds_e=None,
            sym=False,
            real_couplings=True):  
        
    hyb_scalar = np.asarray(hyb_scalar).squeeze()
    if hyb_scalar.ndim != 1:
        raise ValueError("hyb_scalar must be 1D")

    if bounds_e is None:
        bounds_e = [np.min(ws), np.max(ws)]

    print("HybFitCost initialized:")
    print(f"  Method: Global optimization (scalar only)")
    print(f"  Number of target poles = {n_b}")
    print(f"  Couplings restricted to real numbers = {real_couplings}")
    if sym:
        print("  Enforcing particle-hole symmetry around omega=0")

    print("--- Fitting poles via Cost Minimization ---")
    print(f"Using energy bounds for bath sites: [{bounds_e[0]:.2f}, {bounds_e[1]:.2f}]")

    if broadening_Gamma > 0:
        print(f"Applying Lorentzian broadening of width Gamma = {broadening_Gamma}")
        target_delta = lorentzian_convolution(ws, hyb_scalar, broadening_Gamma)
    else:
        target_delta = hyb_scalar
    
    eta_fit = eta_0 + broadening_Gamma
    print(f"Fit will be performed with effective broadening eta_fit = {eta_fit:.4f}, and weight function {weight_func}")
    
    opt_defaults = {'strategy': 'best1bin', 'maxiter': 1000, 'popsize': 15, 'tol': 1e-3, 'disp': False}
    opt_settings = {**opt_defaults}
    
    if sym:
        # For symmetric mode, optimize N_half positive poles, and optionally 1 zero pole
        N_half = n_b // 2
        has_zero = (n_b % 2 != 0)
        e_max = max(abs(bounds_e[0]), abs(bounds_e[1]))
        
        if real_couplings:
            bounds_list = ([(0.0, e_max)] * N_half + [(-0.5, 0.5)] * N_half)
            if has_zero: bounds_list += [(-0.5, 0.5)]

            def sym_cost(params):
                e_half = params[:N_half]
                v_half = params[N_half:2*N_half]
                if has_zero:
                    v_0 = params[-1:]
                    e_full = np.concatenate([e_half, -e_half, [0.0]])
                    v_full = np.concatenate([v_half, v_half, v_0])
                else:
                    e_full = np.concatenate([e_half, -e_half])
                    v_full = np.concatenate([v_half, v_half])
                
                full_params = np.concatenate([e_full, v_full])
                return cost_function(full_params, n_b, ws, target_delta, eta_fit, weight_func, real_couplings=True)

        else:
            bounds_list = ([(0.0, e_max)] * N_half + [(-0.5, 0.5)] * N_half + [(-0.5, 0.5)] * N_half)
            if has_zero: bounds_list += [(-0.5, 0.5)] * 2
                
            def sym_cost(params):
                e_half = params[:N_half]
                idx = N_half
                v_re_half = params[idx:idx+N_half]; idx += N_half
                v_im_half = params[idx:idx+N_half]; idx += N_half
                if has_zero:
                    v_re_0 = params[idx:idx+1]
                    v_im_0 = params[idx+1:idx+2]
                    e_full = np.concatenate([e_half, -e_half, [0.0]])
                    v_re_full = np.concatenate([v_re_half, v_re_half, v_re_0])
                    v_im_full = np.concatenate([v_im_half, v_im_half, v_im_0])
                else:
                    e_full = np.concatenate([e_half, -e_half])
                    v_re_full = np.concatenate([v_re_half, v_re_half])
                    v_im_full = np.concatenate([v_im_half, v_im_half])
                    
                full_params = np.concatenate([e_full, v_re_full, v_im_full])
                return cost_function(full_params, n_b, ws, target_delta, eta_fit, weight_func, real_couplings=False)
            
        print(f"Starting symmetric global optimization ({opt_settings['strategy']})...")
        result = optimize.differential_evolution(sym_cost, bounds=bounds_list, **opt_settings)
        
        # Unpack optimal symmetric parameters
        opt_params = result.x
        e_half = opt_params[:N_half]
        if real_couplings:
            v_half = opt_params[N_half:2*N_half]
            if has_zero:
                v_0 = opt_params[-1:]
                e = np.concatenate([e_half, -e_half, [0.0]])
                v = np.concatenate([v_half, v_half, v_0])
            else:
                e = np.concatenate([e_half, -e_half])
                v = np.concatenate([v_half, v_half])
        else:
            idx = N_half
            v_re_half = opt_params[idx:idx+N_half]; idx += N_half
            v_im_half = opt_params[idx:idx+N_half]; idx += N_half
            if has_zero:
                v_re_0 = opt_params[idx:idx+1]
                v_im_0 = opt_params[idx+1:idx+2]
                e = np.concatenate([e_half, -e_half, [0.0]])
                v = np.concatenate([v_re_half, v_re_half, v_re_0]) + 1j * np.concatenate([v_im_half, v_im_half, v_im_0])
            else:
                e = np.concatenate([e_half, -e_half])
                v = np.concatenate([v_re_half, v_re_half]) + 1j * np.concatenate([v_im_half, v_im_half])

    else:
        if real_couplings:
            bounds_list = ([(bounds_e[0], bounds_e[1])] * n_b + [(-0.5, 0.5)] * n_b)
        else:
            bounds_list = ([(bounds_e[0], bounds_e[1])] * n_b + [(-0.5, 0.5)] * (2 * n_b))

        print(f"Starting global optimization ({opt_settings['strategy']})...")
        result = optimize.differential_evolution(
            cost_function, bounds=bounds_list,
            args=(n_b, ws, target_delta, eta_fit, weight_func, real_couplings),
            **opt_settings
        )
        
        opt_params = result.x
        e = opt_params[:n_b]
        if real_couplings:
            v = opt_params[n_b:2*n_b]
        else:
            v = opt_params[n_b:2*n_b] + 1j * opt_params[2*n_b:]

    if result.success:
        print(f"Optimization successful. Final cost (χ²): {result.fun:.6e}")
    else:
        print(f"WARNING: Optimization may not have converged. Final cost (χ²): {result.fun:.6e}")
    
    # Sort out energies ascendingly for standard output
    sort_idx = np.argsort(e)
    eps_final = e[sort_idx]
    v_final = v[sort_idx]
    R_final = [np.array([[np.abs(vi)**2]]) for vi in v_final] 
    
    print("--- Cost Fit Done ---")
    print("Final optimized poles:")
    for i in range(len(eps_final)):
        c = (np.abs(R_final[i][0,0]))
        print(f"  pole {i}: e = {eps_final[i]:+.6f}, coupling |v|^2 = {c:.6f}")

    return eps_final, R_final


#######

def residues_to_bath(eps, R_list, tol=1e-12):
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

def discretize_hyb(
    omega,
    hyb,           # (Nw, Nimp, Nimp)
    himp,          # (Nimp, Nimp)
    n_target_poles,
    eta_0,              # required by HybFitCost
    bounds_e = None,
    weight_func = 'const',
    broadening_Gamma = 0.0,
    tol = 1e-6,
    enforce_even_total = False,
    verbose = False,
    i_omegas = None,
    sym = False,
    real_couplings = False  #
):
    assert hyb.ndim == 3 and hyb.shape[1] == hyb.shape[2], "hyb must be (Nw, Nimp, Nimp)"
    assert himp.shape[0] == himp.shape[1] == hyb.shape[1], "himp must match hyb"
    
    Nw, Nimp, _ = hyb.shape
    
    M = Nimp // 2
    mid = Nw // 2

    Href = himp + hyb[mid]
    print(Href)
    
    sym_info = symmetries.analyze_symmetries(np.asarray(Href), tol=tol, verbose=verbose)
    blocks = sym_info["blocks"]
    identical_groups = sym_info["identical_groups"]

    print(f"DEBUG: blocks = {blocks}")
    print(f"DEBUG: identical_groups = {identical_groups}")

    # enforce scalar leaders
    leaders = [g[0] for g in identical_groups]
    for leader in leaders:
        if len(blocks[leader]) != 1:
            raise ValueError(
                f"process_hyb_cost requires 1×1 leader blocks, but block {leader} has size {len(blocks[leader])}."
            )

    # 2) fit each 1×1 leader block with cost minimization
    leader_results = {}
    for leader in leaders:
        idx = blocks[leader]  # single index [i]
        i = idx[0]
        hyb_blk = hyb[:, idx, :][:, :, idx]    # shape (Nw,1,1)

        if verbose:
            print(f"\n[CostFit] Fitting leader block {leader} (orbital {i}) with {n_target_poles} poles.")

        # Pass sym and real_couplings explicitly down to scalar_fit
        eps_poles, R_poles = scalar_fit(omega, hyb_blk, 
            n_target_poles,
            eta_0,
            weight_func = weight_func,
            broadening_Gamma = broadening_Gamma,
            bounds_e=bounds_e,
            sym=sym,
            real_couplings=real_couplings)

        # turn residues into bath; for scalar, this yields exactly one column per pole
        H_b, V_blk = residues_to_bath(eps_poles, R_poles)  # V_blk shape (1, Nb)

        leader_results[leader] = {
            "idx": idx,
            "eps_poles": np.asarray(eps_poles, float),
            "R_poles":   [np.asarray(R, np.complex128) for R in R_poles],
            "eps_cols":  np.diag(H_b).copy(),       
            "V":         V_blk.copy(),             
        }

    # 3) duplicate to all blocks and assemble V_full, H_b_full
    V_cols = []
    eps_all = []
    block_to_bath_cols = [[] for _ in range(len(blocks))]

    leader_of_block = {}
    for group in identical_groups:
        L = group[0]
        for b in group:
            leader_of_block[b] = L

    for bidx, idx in enumerate(blocks):
        L = leader_of_block[bidx]
        res = leader_results[L]

        Nb = res["V"].shape[1]
        eps_block = np.asarray(res["eps_cols"], float)
        if eps_block.shape[0] != Nb:
            if eps_block.shape[0] == 1:
                eps_block = np.full(Nb, float(eps_block[0]))
            else:
                raise ValueError(f"Block {bidx}: eps_cols length {eps_block.shape[0]} vs V columns {Nb}")

        V_block_full = np.zeros((Nimp, Nb), dtype=np.complex128)
        V_block_full[idx[0], :] = res["V"][0, :]

        col_start = len(eps_all)
        V_cols.append(V_block_full)
        eps_all.extend(eps_block.tolist())
        block_to_bath_cols[bidx] = list(range(col_start, col_start + Nb))

    V_full = np.hstack(V_cols) if V_cols else np.zeros((Nimp, 0), np.complex128)
    H_b_full = np.diag(np.asarray(eps_all, float)) if eps_all else np.zeros((0, 0), float)

    # 4) spin assignment for bath columns
    alpha_imp = np.arange(0, M)
    beta_imp  = np.arange(M, 2*M)
    alpha_cols= []
    beta_cols = []

    for j in range(V_full.shape[1]):
        coupl = V_full[:, j]
        norm_a = np.linalg.norm(coupl[:M])
        norm_b = np.linalg.norm(coupl[M:])
        if np.isclose(norm_a, norm_b, atol=1e-15):
            (alpha_cols if len(alpha_cols) <= len(beta_cols) else beta_cols).append(j)
        elif norm_a > norm_b:
            alpha_cols.append(j)
        else:
            beta_cols.append(j)

    Nbath = V_full.shape[1]
    if enforce_even_total and ((Nimp + Nbath) % 2 == 1):
        weights = np.sum(np.abs(V_full)**2, axis=0)
        drop_j = int(np.argmin(weights)) if Nbath > 0 else None
        if drop_j is not None:
            keep = np.ones(Nbath, dtype=bool); keep[drop_j] = False
            V_full = V_full[:, keep]
            H_b_full = np.diag(np.asarray(eps_all, float)[keep])
            alpha_cols = [j for j in alpha_cols if j != drop_j]
            beta_cols  = [j for j in beta_cols  if j != drop_j]
            old_to_new = {}
            c = 0
            for j in range(Nbath):
                if keep[j]:
                    old_to_new[j] = c; c += 1
            alpha_cols = [old_to_new[j] for j in alpha_cols]
            beta_cols  = [old_to_new[j] for j in beta_cols]
            for b in range(len(block_to_bath_cols)):
                block_to_bath_cols[b] = [old_to_new[j] for j in block_to_bath_cols[b] if j in old_to_new]

    # 5) build and permute to [imp α, bath α, imp β, bath β]
    imp_alpha = list(alpha_imp)
    imp_beta  = list(beta_imp)
    bath_offset = Nimp
    old_order = (
        imp_alpha
        + [bath_offset + j for j in alpha_cols]
        + imp_beta
        + [bath_offset + j for j in beta_cols]
    )
    perm = np.array(old_order, dtype=int)

    top    = np.hstack([himp, V_full])
    bottom = np.hstack([V_full.conj().T, H_b_full])
    H_full0 = np.vstack([top, bottom])
    H_full  = H_full0[np.ix_(perm, perm)]

    mapping = {
        "blocks": blocks,
        "identical_groups": identical_groups,
        "leader_results": leader_results,             
        "block_to_bath_cols": block_to_bath_cols,
        "perm_full_to_spin_sorted": perm,
        "alpha_imp_idx": alpha_imp,
        "beta_imp_idx": beta_imp,
        "alpha_bath_cols": alpha_cols,
        "beta_bath_cols": beta_cols,
    }

    P = np.eye(H_full.shape[0], dtype=H_full.dtype)[perm]
    H0 = P.T @ H_full @ P
    Vchk = H0[:Nimp, Nimp:]
    Hbck = H0[Nimp:, Nimp:]
    Delta_fit = delta_from_bath(omega, Hbck, Vchk, eta=eta_0 + broadening_Gamma)
    hyb_app = delta_from_bath(omega, Hbck, Vchk, eta=eta_0)

    if i_omegas is not None:
        if i_omegas[0].imag == 0 :
            Delta_mats_fit = delta_from_bath(i_omegas*1j, Hbck, Vchk, eta = 0)
        else:
            Delta_mats_fit = delta_from_bath(i_omegas, Hbck, Vchk, eta = 0)
    else: 
        Delta_mats_fit = None

    return H_full, Delta_fit, hyb_app, mapping, Delta_mats_fit



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


def create_dummy_delta(omega, n_poles, m_orb, e_range=(-1.0, 1.0), eta=0.01):
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
        V = 1e-1 * (np.random.rand(m_orb, m_orb) + 1j * np.random.rand(m_orb, m_orb))
        R_j = V @ V.conj().T
        R.append(R_j)

    z = omega + 1j * eta
    delta = delta_from_poles(z, eps, R)
    return delta, eps, R




########


# hybfit_poles

def fit_poles(omega_grid, delta_complex, n_lanczos_blocks, n_target_poles, warp_kind="atan", warp_w0=0.1, logfile=None):
    """
    Fit a pole representation to a matrix-valued hybridization Δ(ω) using
    a block Lanczos algorithm. Handles scalar hybridizations as a special case.
    
    Returns:
        eps_merged (np.ndarray): The energies of the fitted poles.
        R_merged (list of np.ndarray): The residue matrices of the fitted poles.
    """
    omega = np.asarray(omega_grid, dtype=float)
    delta_in = np.asarray(delta_complex, dtype=np.complex128)

    if delta_in.ndim == 1:
        delta_input = delta_in[:, np.newaxis, np.newaxis]
    elif delta_in.ndim == 3:
        delta_input = delta_in
    else:
        raise ValueError("delta_complex must be 1D or 3D")


    #delta_input = lorentzian_convolution(omega, delta_input[:,0,0], 0.05)
    #delta_input = delta_input[:,np.newaxis,np.newaxis]

    n_omega, m_orb, _ = delta_input.shape
    if omega.shape[0] != n_omega:
        raise ValueError("omega_grid and delta_complex length mismatch")

    n_lanczos_blocks = int(n_lanczos_blocks)
    n_target_poles = int(n_target_poles)

    print( "HybFitPoles initialized (Functional Mode):")
    print( f"  Orbital dimensions M = {m_orb}")
    print( f"  Lanczos blocks = {n_lanczos_blocks}")
    print( f"  Target poles = {n_target_poles}")

    # 1. Fit Lanczos
    T_lanczos, M0_sqrt_lanczos = fit_lanczos(omega, delta_input, n_lanczos_blocks, m_orb, logfile)

    # 2. Extract Poles
    eps_lanczos, R_lanczos = extract_poles(T_lanczos, M0_sqrt_lanczos, m_orb, logfile)

    #print(f"eps_lanczos = {eps_lanczos}")

    # 3. Merge Poles
    eps_merged, R_merged = merge_poles(
        eps_lanczos, R_lanczos, n_target_poles, warp_kind=warp_kind, warp_w0=warp_w0, logfile=logfile
    )

    print( "--- Pole Fit Done ---")
    print( "Final merged poles:")
    for i in range(len(eps_merged)):
        c = np.sqrt(max(np.real(np.trace(R_merged[i])), 0.0))
        print( f"  pole {i}: e = {eps_merged[i]:+.6f}, sqrt Tr(R) = {c:.6f}")

    return eps_merged, R_merged


def fit_lanczos(omega, delta_input, n_lanczos_blocks, m_orb, logfile=None):
    """Dispatcher for scalar vs. block Lanczos algorithm."""
    print( "--- Fitting poles via Lanczos ---")
    
    if m_orb == 1:
        print( "1) Using scalar Lanczos for M=1 case.")
        return run_scalar_lanczos(omega, delta_input, n_lanczos_blocks)
    else:
        print( f"1) Using block Lanczos for M={m_orb} case.")
        return run_block_lanczos(omega, delta_input, n_lanczos_blocks, m_orb, logfile)


def run_scalar_lanczos(omega, delta_input, n_lanczos_blocks):
    """Scalar Lanczos algorithm execution."""
    dx = np.diff(omega)
    w = np.empty_like(omega)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    
    rho_meas = -np.imag(delta_input[:, 0, 0]) / np.pi
    mu = rho_meas * w
    
    N = min(n_lanczos_blocks, len(omega))
    x = omega
    
    def ip(a, b): return float(np.dot(mu, a * b))
    def nrm(a): return np.sqrt(max(ip(a, a), 1e-300))
    
    v_im1 = np.zeros_like(x)
    v = np.ones_like(x)
    v /= nrm(v)
    
    al, be = [], []
    beta_im1 = 0.0
    for j in range(N):
        w_vec = x * v
        a = ip(v, w_vec)
        al.append(a)
        w_vec = w_vec - a * v - beta_im1 * v_im1
        beta = nrm(w_vec)
        if j < N - 1: 
            be.append(beta)
        v_im1, v = v, w_vec / (beta + 1e-300)
        beta_im1 = beta
    
    T = np.zeros((N, N))
    i = np.arange(N)
    T[i, i] = al
    j = np.arange(N - 1)
    T[j, j + 1] = be
    T[j + 1, j] = be
    
    M0_sqrt_lanczos = np.array([[np.sqrt(mu.sum())]])
    return T, M0_sqrt_lanczos


def run_block_lanczos(omega, delta_input, n_lanczos_blocks, block_size, logfile=None):
    """Block Lanczos algorithm execution."""
    dx = np.diff(omega)
    w = np.empty_like(omega)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    
    rho_meas = -np.imag(delta_input) / np.pi
    mu_grid = [rho_meas[i] * w[i] for i in range(len(omega))]
    
    A_blocks, B_blocks, _, M0 = block_lanczos_alg(
        x_vals=omega, weight_mats=mu_grid,
        K=n_lanczos_blocks, b=block_size, logfile=logfile
    )
    
    if B_blocks and len(B_blocks) != len(A_blocks) - 1:
        B_blocks = B_blocks[:max(0, len(A_blocks) - 1)]

    T_lanczos = build_block_tridiagonal(A_blocks, B_blocks)
    M0_sqrt_lanczos = sym_sqrt_psd(M0)
    
    print( f"   T shape = {T_lanczos.shape}")
    return T_lanczos, M0_sqrt_lanczos


def extract_poles(T_lanczos, M0_sqrt_lanczos, block_size, logfile=None):
    """Extract energies and residue matrices from the tridiagonal matrix."""
    print( "2) Extracting poles and residues from T")
    if T_lanczos is None or T_lanczos.size == 0:
        return np.array([]), []

    evals, evecs = npl.eigh(T_lanczos)
    eps_lanczos = evals

    b = block_size
    n_T = T_lanczos.shape[0]
    E1 = np.zeros((n_T, b))
    E1[:b, :b] = np.eye(b)

    R = []
    for j in range(n_T):
        uj = evecs[:, j:j+1]
        vj = M0_sqrt_lanczos.T @ E1.T @ uj
        Rj = vj @ vj.conj().T
        R.append(Rj)
        
    print( f"   poles extracted = {len(eps_lanczos)}")
    return eps_lanczos, R


def merge_poles(eps_lanczos, R_lanczos, n_target_poles, warp_kind="atan", warp_w0=0.1, logfile=None):
    """Main wrapper for pole merging to hit the target number of poles."""
    print( "--- Merging poles ---")
    print( f"merge poles, kind = {warp_kind}")
    
    if len(eps_lanczos) == 0:
        return eps_lanczos, R_lanczos

    eps = eps_lanczos
    W = eps.max() - eps.min()
    E_keep = 0.005 * W
    
    print( f"W={W}, E_keep = {E_keep}")
    print( f"3) Merging to {n_target_poles} poles with Appendix F + low-E bias")
    
    eps_merged, R_merged = merge_poles_block_appendixF(
        eps_lanczos, R_lanczos, n_target_poles,
        cleanup_negative=False, cull_outliers=True,
        bias="fd", w0=E_keep, p=2.0, Tstar=E_keep, gamma=0.0,
        use_warp_spacing=True, warp_kind=warp_kind, logfile=logfile
    )
    return eps_merged, R_merged


def sym_sqrt_psd(A, tol=1e-12):
    """Compute the symmetric square root of a positive semi-definite matrix."""
    A = 0.5 * (A + A.T.conj())
    w, Q = npl.eigh(A)
    w = np.clip(w, 0.0, None)
    return Q @ np.diag(np.sqrt(w)) @ Q.T.conj()


def make_warp(kind="atan", w0=0.1, logfile=None):
    """Return forward and inverse warp functions based on the specified kind."""
    print( f"make warp : kind = {kind}")
    if kind == "atan": return (lambda x: np.arctan(x / w0), lambda y: w0 * np.tan(y))
    if kind == "asinh": return (lambda x: np.arcsinh(x / w0), lambda y: w0 * np.sinh(y))
    if kind == "const": return (lambda x: x, lambda y: y)
    return (lambda x: x, lambda y: y)


def merge_poles_block_warped(eps, R_list, n_target, kind, w0, logfile=None):
    """Alternative warped pole merge (less commonly used vs Appendix F approach)."""
    phi, inv = make_warp(kind, w0, logfile=logfile)
    eps = np.asarray(eps, float).copy()
    R = list(R_list)
    order = np.argsort(eps)
    eps = eps[order]
    R = [R[i] for i in order]
    z = phi(eps)
    Tr_R = np.array([np.real(np.trace(Ri)) for Ri in R])
    
    while len(eps) > n_target:
        costs = (Tr_R[:-1] * Tr_R[1:] / (Tr_R[:-1] + Tr_R[1:] + 1e-300)) * (z[:-1] - z[1:])**2
        i = int(np.argmin(costs))
        R_new = R[i] + R[i+1]
        Tr_new = Tr_R[i] + Tr_R[i+1]
        z_new = (Tr_R[i]*z[i] + Tr_R[i+1]*z[i+1]) / (Tr_new + 1e-300) if Tr_new > 1e-300 else 0.5*(z[i]+z[i+1])
        e_new = inv(z_new)
        eps = np.concatenate([eps[:i], [e_new], eps[i+2:]])
        R = R[:i] + [R_new] + R[i+2:]
        Tr_R = np.concatenate([Tr_R[:i], [Tr_new], Tr_R[i+2:]])
        z = np.concatenate([z[:i], [z_new], z[i+2:]])
        
    return eps, R


def merge_poles_block_appendixF(eps, R_list, n_target,
                                cleanup_negative=False, cull_outliers=True, W_est=None,
                                bias="fd", w0=0.01, p=2.0, Tstar=None, gamma=0.5,
                                use_warp_spacing=True, warp_kind="const", logfile=None):
    """
    Appendix F with low-energy emphasis via biased selection.
    bias: "none" | "power" | "fd"
    w0, p: parameters for h_power
    Tstar: parameter for h_fd (if None uses w0)
    gamma: 0..1 controls spacing factor ℓ_k^gamma
    use_warp_spacing: if True, compute ℓ_k in z=phi(eps) space
    warp_kind: "asinh" | "atan" | "const" for spacing only
    """
    print( f"bias = {bias}, w0 = {w0}, p = {p}, gamma = {gamma}, warp_kind = {warp_kind}")
    tiny = 1e-300
    eps = np.asarray(eps, float).copy()
    R = [np.array(Ri, dtype=np.complex128, copy=True) for Ri in R_list]

    order = np.argsort(eps)
    eps = eps[order]
    R = [R[i] for i in order]

    def weights(R_arr): 
        return np.array([float(np.real(np.trace(Ri))) for Ri in R_arr], dtype=float)

    def make_phi(kind, param_w0):
        if kind == "atan":  return lambda x: np.arctan(x / param_w0)
        if kind == "asinh": return lambda x: np.arcsinh(x / param_w0)
        return lambda x: x

    phi = make_phi(warp_kind, max(w0, 1e-12))
    Tstar = Tstar if Tstar is not None else w0

    def h_abs_e(e):
        x = abs(e)
        if bias == "power":
            return 1.0 + (x/max(w0, 1e-12))**p
        if bias == "fd":
            arg = abs(e)/max(Tstar, 1e-12)
            return 1.0 + np.exp(np.clip(arg, 0.0, 50.0))
        return 1.0

    def eliminate_at(k, cur_eps, cur_R):
        a = cur_eps
        w = weights(cur_R)
        akm1, ak, akp1 = a[k-1], a[k], a[k+1]
        wkms, wk, wkps = w[k-1], w[k], w[k+1]
        denom = max(akp1 - akm1, tiny)
        fac_L = (akp1 - ak) / denom
        fac_R = (ak - akm1) / denom
        
        # new energies from first-moment conservation
        akm1_num = wkms * akm1 * (akp1 - akm1) + wk * ak * (akp1 - ak)
        akm1_den = wkms * (akp1 - akm1) + wk * (akp1 - ak) + tiny
        akm1_new = akm1_num / akm1_den
        
        akp1_num = wkps * akp1 * (akp1 - akm1) + wk * ak * (ak - akm1)
        akp1_den = wkps * (akp1 - akm1) + wk * (ak - akm1) + tiny
        akp1_new = akp1_num / akp1_den
        
        # matrix residues with same partition
        Rkm1_new = cur_R[k-1] + fac_L * cur_R[k]
        Rkp1_new = cur_R[k+1] + fac_R * cur_R[k]
        
        eps2 = np.concatenate([a[:k-1], [akm1_new, akp1_new], a[k+2:]])
        R2 = cur_R[:k-1] + [Rkm1_new, Rkp1_new] + cur_R[k+2:]
        
        # enforce strict monotonicity if needed
        if k-2 >= 0 and not (eps2[k-2] < eps2[k-1]):
            eps2[k-1] = np.nextafter(eps2[k-2], np.inf)
        if k < len(eps2)-1 and not (eps2[k-1] < eps2[k]):
            eps2[k] = np.nextafter(eps2[k-1], np.inf)
            
        return eps2, R2

    # optional: cull absurd outliers first
    if cull_outliers and len(eps) > n_target:
        w = weights(R)
        if W_est is None:
            lo, hi = (np.percentile(eps, [1, 99]) if len(eps) > 20 else (eps.min(), eps.max()))
            W_est = max(hi - lo, 1.0)
        keep = np.ones_like(eps, dtype=bool)
        keep &= ~(np.abs(eps) > 10 * W_est) | (w > 1e-7)
        if keep.sum() >= n_target:
            eps = eps[keep]
            R = [Ri for Ri, k in zip(R, keep) if k]
            order = np.argsort(eps)
            eps = eps[order]
            R = [R[i] for i in order]

    # negative-weight cleanup if requested
    if cleanup_negative:
        while True:
            w = weights(R)
            candidates = np.where(w[1:-1] < 0)[0] + 1
            if candidates.size == 0 or len(eps) <= max(n_target, 2): break
            # remove most negative first
            k = int(candidates[np.argmin(w[candidates])])
            eps, R = eliminate_at(k, eps, R)

    # main reduction with biased selection
    while len(eps) > max(n_target, 2):
        w = weights(R)
        interior = np.arange(1, len(eps)-1)
        # compute selection scores
        if use_warp_spacing:
            z = phi(eps)
            dL = z[interior] - z[interior-1]
            dR = z[interior+1] - z[interior]
        else:
            dL = eps[interior] - eps[interior-1]
            dR = eps[interior+1] - eps[interior]
        ell = np.minimum(dL, dR)
        h = np.array([h_abs_e(eps[i]) for i in interior], dtype=float)
        rho_loc = w[interior-1] + w[interior] + w[interior+1]

        # prioritize any negative weight if present
        neg_mask = w[interior] <= 0
        if np.any(neg_mask):
            idxs = interior[neg_mask]
            k = int(idxs[np.argmin(w[idxs])])
        else:
            # selection score
            S = (w[interior] / h) * np.maximum(ell, tiny)**gamma * np.maximum(rho_loc, tiny)**0.0
            k = int(interior[np.argmin(S)])

        eps, R = eliminate_at(k, eps, R)

    # handle the small-N tail exactly
    if len(eps) > n_target:
        assert len(eps) == 2 and n_target == 1, "unexpected small-N case"
        w = weights(R)
        if use_warp_spacing:
            z = phi(eps)
            z_new = (w[0]*z[0] + w[1]*z[1]) / (w[0] + w[1] + tiny)
            if warp_kind == "atan":
                inv = lambda y: w0 * np.tan(y)
            elif warp_kind == "asinh":
                inv = lambda y: w0 * np.sinh(y)
            else:
                inv = lambda y: y
            e_new = float(inv(z_new))
        else:
            e_new = float((w[0]*eps[0] + w[1]*eps[1]) / (w[0] + w[1] + tiny))
        R_new = R[0] + R[1]
        eps = np.array([e_new], dtype=float)
        R = [R_new]

    return eps, R


def block_lanczos_alg(x_vals, weight_mats, K, b, tol=1e-12, logfile=None):
    """Block Lanczos with full re-orthogonalization and robust breakdown handling."""
    S = len(x_vals)
    M = weight_mats[0].shape[0]

    mu_stack = np.stack(weight_mats, axis=0)
    
    if not np.isfinite(mu_stack).all():
        raise ValueError("Input weight matrices contain NaNs or Infs.")
    if not np.isfinite(x_vals).all():
        raise ValueError("Input omega grid contains NaNs or Infs.")

    def blk_ip_fast(Phi, Psi):
        acc = np.einsum('smi,smn,snj->ij', Phi.conj(), mu_stack, Psi, optimize=True)
        return 0.5 * (acc + acc.conj().T)

    def robust_normalization(W):
        G = blk_ip_fast(W, W)
        if not np.isfinite(G).all():
            return None, None, 0, np.inf

        eigvals, eigvecs = npl.eigh(G)
        max_eig = np.max(eigvals)
        
        if max_eig < 1e-14: 
            return np.zeros_like(G), np.zeros_like(G), 0, max_eig

        limit = max(tol * max_eig, 1e-14)
        keep_mask = eigvals > limit
        
        if not np.any(keep_mask):
            return np.zeros_like(G), np.zeros_like(G), 0, max_eig

        rank = np.sum(keep_mask)
        ev_kept = eigvals[keep_mask]
        U_kept = eigvecs[:, keep_mask]
        
        sqrt_ev = np.sqrt(ev_kept)
        inv_sqrt_ev = 1.0 / sqrt_ev
        
        Bk_pinv = U_kept @ np.diag(inv_sqrt_ev) @ U_kept.conj().T
        Bk = U_kept @ np.diag(sqrt_ev) @ U_kept.conj().T
        
        return Bk, Bk_pinv, rank, max_eig

    M0 = sum(weight_mats)
    M0 = 0.5 * (M0 + M0.conj().T)
    
    Phi0_raw = np.zeros((S, M, b), dtype=np.complex128)
    for s in range(S):
        Phi0_raw[s] = np.eye(M, b, dtype=np.complex128)

    Bk, Bk_pinv, rank, _ = robust_normalization(Phi0_raw)
    
    if rank < 1:
        print( "Warning: Initial weight matrix M0 is essentially zero. Returning empty fit.")
        return [], [], [], M0

    Phi_i = Phi0_raw @ Bk_pinv

    A_blocks = []
    B_blocks = []
    Phi_list = [Phi_i.copy()]
    
    Phi_im1 = np.zeros_like(Phi_i)
    B_im1 = np.zeros((b, b), dtype=np.complex128)

    for k in range(K):
        XPhi = x_vals[:, None, None] * Phi_i
        
        Ak = blk_ip_fast(Phi_i, XPhi)
        A_blocks.append(Ak)

        W = XPhi - (Phi_i @ Ak) - (Phi_im1 @ B_im1.conj().T)

        for _ in range(2):
            for P in Phi_list:
                C = blk_ip_fast(P, W)
                W -= P @ C

        if not np.isfinite(W).all():
            print(f"Warning: Block Lanczos breakdown at step {k}: Residual W contains NaNs/Infs.")
            break

        Bk, Bk_pinv, rank, w_norm = robust_normalization(W)
        
        if rank < b:
            print(f"Info: Block Lanczos exhausted subspace at step {k} (rank {rank}/{b}, norm {w_norm:.2e}).")
            break
        
        if k == K - 1:
            break

        B_blocks.append(Bk)
        Phi_ip1 = W @ Bk_pinv
        
        Phi_im1 = Phi_i
        Phi_i = Phi_ip1
        B_im1 = Bk
        
        Phi_list.append(Phi_i.copy())

    return A_blocks, B_blocks, Phi_list, M0


def build_block_tridiagonal(A_blocks, B_blocks):
    """Construct a large block-tridiagonal matrix from the A and B blocks."""
    if not A_blocks: return np.zeros((0, 0), dtype=float)
    
    b = A_blocks[0].shape[0]
    K_eff = len(B_blocks) + 1
    n = K_eff * b
    T = np.zeros((n, n), dtype=np.complex128)
    
    for k in range(K_eff):
        T[k*b:(k+1)*b, k*b:(k+1)*b] = A_blocks[k]
        if k < len(B_blocks):
            Bk = B_blocks[k]
            T[k*b:(k+1)*b, (k+1)*b:(k+2)*b] = Bk
            T[(k+1)*b:(k+2)*b, k*b:(k+1)*b] = Bk
            
    return np.real_if_close(0.5 * (T + T.conj().T))


def discretize_hyb_poles(
    omega,
    hyb,           # (Nw, Nimp, Nimp)
    himp,          # (Nimp, Nimp)
    n_target_poles,
    eta_0,              # required by HybFitCost
    bounds_e = None,
    weight_func = 'const',
    broadening_Gamma = 0.0,
    tol = 1e-6,
    enforce_even_total = False,
    verbose = False,
    i_omegas = None,
    sym = False,
    real_couplings = False  #
):
    assert hyb.ndim == 3 and hyb.shape[1] == hyb.shape[2], "hyb must be (Nw, Nimp, Nimp)"
    assert himp.shape[0] == himp.shape[1] == hyb.shape[1], "himp must match hyb"
    
    Nw, Nimp, _ = hyb.shape
    
    M = Nimp // 2
    mid = Nw // 2

    Href = himp + hyb[mid]
    print(Href)
    
    sym_info = symmetries.analyze_symmetries(np.asarray(Href), tol=tol, verbose=verbose)
    blocks = sym_info["blocks"]
    identical_groups = sym_info["identical_groups"]

    print(f"DEBUG: blocks = {blocks}")
    print(f"DEBUG: identical_groups = {identical_groups}")

    # enforce scalar leaders
    leaders = [g[0] for g in identical_groups]
    for leader in leaders:
        if len(blocks[leader]) != 1:
            raise ValueError(
                f"process_hyb_cost requires 1×1 leader blocks, but block {leader} has size {len(blocks[leader])}."
            )

    # 2) fit each 1×1 leader block with cost minimization
    leader_results = {}
    for leader in leaders:
        idx = blocks[leader]  # single index [i]
        i = idx[0]
        hyb_blk = hyb[:, idx, :][:, :, idx]    # shape (Nw,1,1)

        if verbose:
            print(f"\n[CostFit] Fitting leader block {leader} (orbital {i}) with {n_target_poles} poles.")

        n_lanczos_blocks = 301
        eps_poles, R_poles = fit_poles(omega, hyb_blk, 
                                       n_lanczos_blocks, 
                                       n_target_poles, warp_kind="const", warp_w0=0.1)


        # turn residues into bath; for scalar, this yields exactly one column per pole
        H_b, V_blk = residues_to_bath(eps_poles, R_poles)  # V_blk shape (1, Nb)

        leader_results[leader] = {
            "idx": idx,
            "eps_poles": np.asarray(eps_poles, float),
            "R_poles":   [np.asarray(R, np.complex128) for R in R_poles],
            "eps_cols":  np.diag(H_b).copy(),       
            "V":         V_blk.copy(),             
        }

    # 3) duplicate to all blocks and assemble V_full, H_b_full
    V_cols = []
    eps_all = []
    block_to_bath_cols = [[] for _ in range(len(blocks))]

    leader_of_block = {}
    for group in identical_groups:
        L = group[0]
        for b in group:
            leader_of_block[b] = L

    for bidx, idx in enumerate(blocks):
        L = leader_of_block[bidx]
        res = leader_results[L]

        Nb = res["V"].shape[1]
        eps_block = np.asarray(res["eps_cols"], float)
        if eps_block.shape[0] != Nb:
            if eps_block.shape[0] == 1:
                eps_block = np.full(Nb, float(eps_block[0]))
            else:
                raise ValueError(f"Block {bidx}: eps_cols length {eps_block.shape[0]} vs V columns {Nb}")

        V_block_full = np.zeros((Nimp, Nb), dtype=np.complex128)
        V_block_full[idx[0], :] = res["V"][0, :]

        col_start = len(eps_all)
        V_cols.append(V_block_full)
        eps_all.extend(eps_block.tolist())
        block_to_bath_cols[bidx] = list(range(col_start, col_start + Nb))

    V_full = np.hstack(V_cols) if V_cols else np.zeros((Nimp, 0), np.complex128)
    H_b_full = np.diag(np.asarray(eps_all, float)) if eps_all else np.zeros((0, 0), float)

    # 4) spin assignment for bath columns
    alpha_imp = np.arange(0, M)
    beta_imp  = np.arange(M, 2*M)
    alpha_cols= []
    beta_cols = []

    for j in range(V_full.shape[1]):
        coupl = V_full[:, j]
        norm_a = np.linalg.norm(coupl[:M])
        norm_b = np.linalg.norm(coupl[M:])
        if np.isclose(norm_a, norm_b, atol=1e-15):
            (alpha_cols if len(alpha_cols) <= len(beta_cols) else beta_cols).append(j)
        elif norm_a > norm_b:
            alpha_cols.append(j)
        else:
            beta_cols.append(j)

    Nbath = V_full.shape[1]
    if enforce_even_total and ((Nimp + Nbath) % 2 == 1):
        weights = np.sum(np.abs(V_full)**2, axis=0)
        drop_j = int(np.argmin(weights)) if Nbath > 0 else None
        if drop_j is not None:
            keep = np.ones(Nbath, dtype=bool); keep[drop_j] = False
            V_full = V_full[:, keep]
            H_b_full = np.diag(np.asarray(eps_all, float)[keep])
            alpha_cols = [j for j in alpha_cols if j != drop_j]
            beta_cols  = [j for j in beta_cols  if j != drop_j]
            old_to_new = {}
            c = 0
            for j in range(Nbath):
                if keep[j]:
                    old_to_new[j] = c; c += 1
            alpha_cols = [old_to_new[j] for j in alpha_cols]
            beta_cols  = [old_to_new[j] for j in beta_cols]
            for b in range(len(block_to_bath_cols)):
                block_to_bath_cols[b] = [old_to_new[j] for j in block_to_bath_cols[b] if j in old_to_new]

    # 5) build and permute to [imp α, bath α, imp β, bath β]
    imp_alpha = list(alpha_imp)
    imp_beta  = list(beta_imp)
    bath_offset = Nimp
    old_order = (
        imp_alpha
        + [bath_offset + j for j in alpha_cols]
        + imp_beta
        + [bath_offset + j for j in beta_cols]
    )
    perm = np.array(old_order, dtype=int)

    top    = np.hstack([himp, V_full])
    bottom = np.hstack([V_full.conj().T, H_b_full])
    H_full0 = np.vstack([top, bottom])
    H_full  = H_full0[np.ix_(perm, perm)]

    mapping = {
        "blocks": blocks,
        "identical_groups": identical_groups,
        "leader_results": leader_results,             
        "block_to_bath_cols": block_to_bath_cols,
        "perm_full_to_spin_sorted": perm,
        "alpha_imp_idx": alpha_imp,
        "beta_imp_idx": beta_imp,
        "alpha_bath_cols": alpha_cols,
        "beta_bath_cols": beta_cols,
    }

    P = np.eye(H_full.shape[0], dtype=H_full.dtype)[perm]
    H0 = P.T @ H_full @ P
    Vchk = H0[:Nimp, Nimp:]
    Hbck = H0[Nimp:, Nimp:]
    Delta_fit = delta_from_bath(omega, Hbck, Vchk, eta=eta_0 + broadening_Gamma)
    hyb_app = delta_from_bath(omega, Hbck, Vchk, eta=eta_0)

    if i_omegas is not None:
        if i_omegas[0].imag == 0 :
            Delta_mats_fit = delta_from_bath(i_omegas*1j, Hbck, Vchk, eta = 0)
        else:
            Delta_mats_fit = delta_from_bath(i_omegas, Hbck, Vchk, eta = 0)
    else: 
        Delta_mats_fit = None

    return H_full, Delta_fit, hyb_app, mapping, Delta_mats_fit

