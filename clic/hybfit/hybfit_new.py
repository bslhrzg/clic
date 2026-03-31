import numpy as np
from scipy import optimize
from clic.symmetries import symmetries
import numpy.linalg as npl
from clic.io_clic.io_utils import dump


def cost_function(params, n_b, ws, target_delta, eta, weight_str):
    e = params[:n_b]
    v = params[n_b:2*n_b] + 1j * params[2*n_b:3*n_b]
    
    model_delta = hybridization_model(e, v, ws, eta)
    difference = target_delta - model_delta
    
    if weight_str == "const": weight = 1.0
    elif weight_str == "inv2": weight = 1.0 / (ws**2 + 1e-2)
    else: raise ValueError(f"Unknown weight: {weight_str}")
        
    return np.sum((weight * np.abs(difference))**2)

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
            bounds_e=None):
        

        hyb_scalar = np.asarray(hyb_scalar).squeeze()
        if hyb_scalar.ndim != 1:
            raise ValueError("hyb_scalar must be 1D")

        # Final results
        eps_final = None
        R_final = None

        print("HybFitCost initialized:")
        print(f"  Method: Global optimization (scalar only)")
        print(f"  Number of target poles = {n_b}")

        """
        Execute the fitting by running the global optimization algorithm.
        """

        print("--- Fitting poles via Cost Minimization ---")
        
        if bounds_e is None:
            bounds_e = [np.min(ws), np.max(ws)]
        print(f"Using energy bounds for bath sites: [{bounds_e[0]:.2f}, {bounds_e[1]:.2f}]")

        if broadening_Gamma > 0:
            print(f"Applying Lorentzian broadening of width Gamma = {broadening_Gamma}")
            target_delta = lorentzian_convolution(ws,hyb_scalar, broadening_Gamma)
        else:
            target_delta = hyb_scalar
        
        eta_fit = eta_0 + broadening_Gamma
        print(f"Fit will be performed with effective broadening eta_fit = {eta_fit:.4f}, and weight function {weight_func}")
        
        # Set optimizer defaults if not provided
        opt_defaults = {'strategy': 'best1bin', 'maxiter': 1000, 'popsize': 15, 'tol': 1e-3, 'disp': False}
        opt_settings = {**opt_defaults}
        
        bounds_list = ([(bounds_e[0], bounds_e[1])] * n_b +
                       [(-0.1, 0.1)] * (2 * n_b))

        #guess = np.random.normal(0.1,0.01,3*self.n_target_poles)

        print(f"Starting global optimization ({opt_settings['strategy']})...")

        result = optimize.differential_evolution(
            cost_function,
            bounds=bounds_list,
            args=(n_b, ws, target_delta, eta_fit, weight_func),
            #x0 = guess,
            **opt_settings
        )

        if result.success:
            print(f"Optimization successful. Final cost (χ²): {result.fun:.6e}")
        else:
            print(f"WARNING: Optimization may not have converged. Final cost (χ²): {result.fun:.6e}")
        
        # Unpack, sort, and store results
        opt_params = result.x
        e = opt_params[:n_b]
        v = opt_params[n_b:2*n_b] + 1j * opt_params[2*n_b:]
        
        sort_idx = np.argsort(e)
        eps_final = e[sort_idx]
        v_final = v[sort_idx]
        R_final = [np.array([[np.abs(vi)**2]]) for vi in v_final] # Store as standard residue matrix
        
        print("--- Cost Fit Done ---")
        print("Final optimized poles:")
        for i in range(len(eps_final)):
            c = (np.abs(R_final[i][0,0]))
            print(f"  pole {i}: e = {eps_final[i]:+.6f}, coupling |v|^2 = {c:.6f}")

        return eps_final, R_final


#######

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
    enforce_even_total = True,
    verbose = False,
):
    """
    Symmetry-aware fitting pipeline using HybFitCost on scalar blocks only.

    Preconditions
    -------------
    Every non-equivalent leader block must be 1×1. If a leader block has size > 1,
    this function raises a ValueError. Use your poles pipeline for non-scalar blocks.

    Returns
    -------
    H_full : np.ndarray
        Assembled Hamiltonian (Nimp+Nbath)×(Nimp+Nbath) with α first.
    mapping : dict
    """


    assert hyb.ndim == 3 and hyb.shape[1] == hyb.shape[2], "hyb must be (Nw, Nimp, Nimp)"
    assert himp.shape[0] == himp.shape[1] == hyb.shape[1], "himp must match hyb"
    
    Nw, Nimp, _ = hyb.shape
    
    M = Nimp // 2
    mid = Nw // 2

    # 1) symmetry from reference one body
    Href = himp + hyb[mid]
    print(Href)
    sym = symmetries.analyze_symmetries(np.asarray(Href), tol=tol, verbose=verbose)
    blocks = sym["blocks"]
    identical_groups = sym["identical_groups"]

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

        eps_poles, R_poles = scalar_fit(omega, hyb_blk, 
            n_target_poles,
            eta_0,
            weight_func = weight_func,
            broadening_Gamma = broadening_Gamma,
            bounds_e=bounds_e)

        # turn residues into bath; for scalar, this yields exactly one column per pole
        H_b, V_blk = residues_to_bath(eps_poles, R_poles)  # V_blk shape (1, Nb)

        leader_results[leader] = {
            "idx": idx,
            # per-pole for analysis
            "eps_poles": np.asarray(eps_poles, float),
            "R_poles":   [np.asarray(R, np.complex128) for R in R_poles],
            # per-column for assembly
            "eps_cols":  np.diag(H_b).copy(),       # len == V_blk.shape[1]
            "V":         V_blk.copy(),              # (1, Nb)
        }

    # 3) duplicate to all blocks and assemble V_full, H_b_full
    V_cols = []
    eps_all = []
    block_to_bath_cols = [[] for _ in range(len(blocks))]

    # map any block to its leader
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
        # place V on the single impurity index of this block
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
        # deterministic routing by coupling norms on α vs β
        norm_a = np.linalg.norm(coupl[:M])
        norm_b = np.linalg.norm(coupl[M:])
        if np.isclose(norm_a, norm_b, atol=1e-15):
            # tie break to keep counts balanced
            (alpha_cols if len(alpha_cols) <= len(beta_cols) else beta_cols).append(j)
        elif norm_a > norm_b:
            alpha_cols.append(j)
        else:
            beta_cols.append(j)

    # optional: enforce even dimension
    Nbath = V_full.shape[1]
    if enforce_even_total and ((Nimp + Nbath) % 2 == 1):
        # drop globally smallest-weight column
        weights = np.sum(np.abs(V_full)**2, axis=0)
        drop_j = int(np.argmin(weights)) if Nbath > 0 else None
        if drop_j is not None:
            keep = np.ones(Nbath, dtype=bool); keep[drop_j] = False
            V_full = V_full[:, keep]
            H_b_full = np.diag(np.asarray(eps_all, float)[keep])
            alpha_cols = [j for j in alpha_cols if j != drop_j]
            beta_cols  = [j for j in beta_cols  if j != drop_j]
            # remap indices
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
        "leader_results": leader_results,             # contains eps_poles, R_poles, eps_cols, V
        "block_to_bath_cols": block_to_bath_cols,
        "perm_full_to_spin_sorted": perm,
        "alpha_imp_idx": alpha_imp,
        "beta_imp_idx": beta_imp,
        "alpha_bath_cols": alpha_cols,
        "beta_bath_cols": beta_cols,
    }

    # optional quick check at same eta as your input
    # un-permute back to block form and compare Delta_fit vs input
    P = np.eye(H_full.shape[0], dtype=H_full.dtype)[perm]
    H0 = P.T @ H_full @ P
    Vchk = H0[:Nimp, Nimp:]
    Hbck = H0[Nimp:, Nimp:]
    Delta_fit = delta_from_bath(omega, Hbck, Vchk, eta=eta_0 + broadening_Gamma)

    return H_full, Delta_fit, mapping


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



ws = np.linspace(-5,5,301)

nimp = 2
nb = 3
eta=0.1
himp = np.diag([0,1])
hyb = np.zeros((len(ws),nimp,nimp),dtype=complex)
for i in range(nimp):
    hyb_,eps,R = create_dummy_delta(ws,nb,1,eta=eta)
    print(eps)
    print(R)
    hyb[:,i,i] = hyb_[:,0,0]



Hfull,delta_fit, mapping = discretize_hyb(
    ws,
    hyb,           # (Nw, Nimp, Nimp)
    himp,          # (Nimp, Nimp)
    nb,
    eta
)

hybdos = -np.trace(hyb, axis1=1, axis2=2).imag
hybappdos = -np.trace(delta_fit,axis1=1, axis2=2).imag

dump(hybdos,ws,"hyb","./")
dump(hybappdos,ws,"hyb_app","./")