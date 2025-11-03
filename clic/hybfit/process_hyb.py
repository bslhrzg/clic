# hybfit/process_hyb.py

import numpy as np
from typing import Dict, List, Optional
import os
import matplotlib.pyplot as plt

from .hybfit import fit, analyze_fit
from . import utils
from .. import symmetries
from ..io_utils import vprint


def print_summary(title, H_full, mapping):
    vprint(1,f"=== {title} ===")
    vprint(1,np.round(np.real(H_full),3))
    vprint(1,"blocks:", mapping["blocks"])
    vprint(1,"identical_groups:", mapping["identical_groups"])
    Nb = H_full.shape[0] - len(mapping["alpha_imp_idx"]) - len(mapping["beta_imp_idx"])
    vprint(1,"H_full shape:", H_full.shape, " Nbath:", Nb)
    vprint(1,"alpha bath cols:", mapping["alpha_bath_cols"])
    vprint(1,"beta  bath cols:", mapping["beta_bath_cols"])


def _unpermute_to_block_form(H_full: np.ndarray, perm: np.ndarray, Nimp: int):
    P = np.eye(H_full.shape[0], dtype=H_full.dtype)[perm]
    H0 = P.T @ H_full @ P
    himp0 = H0[:Nimp, :Nimp]
    V     = H0[:Nimp, Nimp:]
    Hb    = H0[Nimp:, Nimp:]
    return H0, himp0, V, Hb

def analyze_block_fits(omega: np.ndarray,
                       hyb: np.ndarray,
                       mapping: dict,
                       eta: float,
                       logfile=None) -> dict:
    """
    Runs your analyze_fit on every non-equivalent leader block.
    Returns a dict keyed by leader block index with the error summaries.
    """
    results = {}
    leaders = [g[0] for g in mapping["identical_groups"]]
    for leader in leaders:
        info = mapping["leader_results"][leader]
        idx = info["idx"]
        eps = info["eps_poles"]
        R_list = info["R_poles"]
        delta_blk_true = hyb[:, idx, :][:, :, idx]
        vprint(1,f"\n[Block leader {leader}] size {len(idx)}",filename=logfile)
        res = analyze_fit(omega, delta_blk_true, eps, R_list, eta=eta, logfile=logfile)
        results[leader] = res
    return results

def _plot_all_components_grid(omega, Delta_in, Delta_fit, title_prefix, out_prefix):
    Nw, M, _ = Delta_in.shape
    # real parts
    fig_re, axes_re = plt.subplots(M, M, figsize=(3*M, 2.2*M), squeeze=False)
    for i in range(M):
        for j in range(M):
            ax = axes_re[i, j]
            ax.plot(omega, np.real(Delta_in[:, i, j]), label="Re Δ", lw=1.5)
            ax.plot(omega, np.real(Delta_fit[:, i, j]), label="Re Δ_fit", lw=1.0, ls="--")
            if i == M-1: ax.set_xlabel("ω")
            if j == 0:   ax.set_ylabel(f"{i},{j}")
            ax.tick_params(axis="both", labelsize=8)
    axes_re[0,0].legend(loc="upper right", fontsize=8)
    fig_re.suptitle(f"{title_prefix} real parts", fontsize=12)
    fig_re.tight_layout(rect=[0, 0, 1, 0.97])
    fig_re.savefig(f"{out_prefix}_real.png", dpi=150)
    plt.close(fig_re)
    # imaginary parts
    fig_im, axes_im = plt.subplots(M, M, figsize=(3*M, 2.2*M), squeeze=False)
    for i in range(M):
        for j in range(M):
            ax = axes_im[i, j]
            ax.plot(omega, np.imag(Delta_in[:, i, j]), label="Im Δ", lw=1.5)
            ax.plot(omega, np.imag(Delta_fit[:, i, j]), label="Im Δ_fit", lw=1.0, ls="--")
            if i == M-1: ax.set_xlabel("ω")
            if j == 0:   ax.set_ylabel(f"{i},{j}")
            ax.tick_params(axis="both", labelsize=8)
    axes_im[0,0].legend(loc="upper right", fontsize=8)
    fig_im.suptitle(f"{title_prefix} imaginary parts", fontsize=12)
    fig_im.tight_layout(rect=[0, 0, 1, 0.97])
    fig_im.savefig(f"{out_prefix}_imag.png", dpi=150)
    plt.close(fig_im)

def evaluate_full_fit_and_plots(omega: np.ndarray,
                                hyb: np.ndarray,
                                H_full: np.ndarray,
                                mapping: dict,
                                eta: float,
                                out_dir: str,
                                case_tag: str) -> dict:
    """
    Global assessment of the reconstructed AIM:
      - un-permute to get [himp, V; V†, Hb]
      - build Δ_fit with given eta
      - compute errors on Re and Im
      - dump per-component plots
      - return metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    Nimp = len(mapping["alpha_imp_idx"]) + len(mapping["beta_imp_idx"])
    perm = mapping["perm_full_to_spin_sorted"]

    _, _, V, Hb = _unpermute_to_block_form(H_full, perm, Nimp)
    Delta_fit = utils.delta_from_bath(omega, Hb, V, eta=eta)

    # errors, mirroring your analyze_fit style
    err_l2_re = utils.rel_l2_error(np.real(hyb), np.real(Delta_fit))
    err_l2_im = utils.rel_l2_error(np.imag(hyb), np.imag(Delta_fit))
    chi_const = utils.cost_l2_integral(hyb, Delta_fit, omega, weight='const')
    chi_inv2 = utils.cost_l2_integral(hyb, Delta_fit, omega, weight='inv2')

    vprint(1,"\n" + "="*50)
    vprint(1,f"GLOBAL FIT ANALYSIS  [{case_tag}]  with eta = {eta:.4f}")
    vprint(1,"="*50)
    vprint(1,f"  Relative L2 Error (Re): {err_l2_re:.4e}")
    vprint(1,f"  Relative L2 Error (Im): {err_l2_im:.4e}")
    vprint(1,f"  Cost L2 Integral (const): {chi_const:.4e}")
    vprint(1,f"  Cost L2 Integral (inv2):  {chi_inv2:.4e}")


    # plots
    out_prefix = os.path.join(out_dir, case_tag)
    _plot_all_components_grid(omega, hyb, Delta_fit, title_prefix=case_tag, out_prefix=out_prefix)

    return {
        "rel_l2_re": err_l2_re,
        "rel_l2_im": err_l2_im,
        "cost_const": chi_const,
        "cost_inv2": chi_inv2,
    }

def _restrict_hyb(hyb: np.ndarray, idx: List[int]) -> np.ndarray:
    # hyb shape (Nw, Nimp, Nimp) -> (Nw, k, k)
    return hyb[:, idx, :][:, :, idx]

def _pick_smallest_weight_global(eps_all: np.ndarray,
                                 V_full: np.ndarray) -> Optional[int]:
    """
    Identify the global bath column with the smallest spectral weight Tr(V v v† V†) proxy.
    For diagonal bath, weight of column j is ||V[:, j]||^2.
    Returns the column index to drop, or None if V_full has 0 columns.
    """
    if V_full.shape[1] == 0:
        return None
    weights = np.sum(np.abs(V_full)**2, axis=0)
    return int(np.argmin(weights))

def process_hyb_poles(
    omega: np.ndarray,
    hyb: np.ndarray,           # (Nw, Nimp, Nimp)
    himp: np.ndarray,          # (Nimp, Nimp)
    n_target_poles: int,
    n_lanczos_blocks: int = 101,
    warp_kind: str = "const",
    warp_w0: float = 0.01,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    enforce_even_total: bool = True,
    verbose: bool = False,
    logfile=None
):
    """
    Run symmetry-aware pole fitting and assemble the full impurity+bath Hamiltonian.

    Returns
    -------
    H_full : np.ndarray
        Assembled Hamiltonian of shape (Nimp + Nbath, Nimp + Nbath) with α indices first.
    mapping : dict
        {
          'blocks': List[List[int]],
          'identical_groups': List[List[int]],
          'leader_results': Dict[int, dict],   # per leader block
          'block_to_bath_cols': List[List[int]],
          'perm_full_to_spin_sorted': np.ndarray,  # permutation applied to H_full
          'alpha_imp_idx': np.ndarray,
          'beta_imp_idx': np.ndarray,
          'alpha_bath_cols': List[int],
          'beta_bath_cols': List[int],
        }
    """

    rng = np.random.default_rng(random_state)

    # basic checks
    omega = np.asarray(omega, float)
    hyb = np.asarray(hyb, np.complex128)
    himp = np.asarray(himp, np.complex128)

    assert hyb.ndim == 3 and hyb.shape[1] == hyb.shape[2], "hyb must be (Nw, Nimp, Nimp)"
    assert himp.shape[0] == himp.shape[1] == hyb.shape[1], "himp must match hyb"
    Nw, Nimp, _ = hyb.shape
    if Nimp % 2 != 0:
        raise ValueError("Nimp must be even for spin α|β split.")
    M = Nimp // 2
    mid = Nw // 2

    # symmetry on reference one body
    Href = himp + hyb[mid]
    sym = symmetries.analyze_symmetries(np.asarray(Href), tol=tol, verbose=verbose)
    blocks: List[List[int]] = sym["blocks"]
    identical_groups: List[List[int]] = sym["identical_groups"]

    # build group leader map
    leader_of_block = {}
    for group in identical_groups:
        leader = group[0]
        for bidx in group:
            leader_of_block[bidx] = leader

    # fit only leaders
    leader_results: Dict[int, Dict] = {}
    for leader in sorted({g[0] for g in identical_groups}):
        idx = blocks[leader]
        hyb_blk = _restrict_hyb(hyb, idx)  # (Nw, k, k)

        if verbose:
            vprint(1,f"\nFitting leader block {leader} with size {len(idx)}")

        eps, R_list = fit(
            omega=omega,
            delta=hyb_blk,
            n_poles=n_target_poles,
            method='poles_reconstruction',
            n_lanczos_blocks=(n_lanczos_blocks if n_lanczos_blocks is not None else 10*n_target_poles),
            warp_kind=warp_kind,   # or 'atan' if you prefer
            warp_w0=warp_w0,
            logfile=logfile
        )

        H_b, V_blk = utils.residues_to_bath(eps, R_list)

        # store leader outcome
        leader_results[leader] = {
            "idx": idx,
            # per-POLE data (same length): for analyze_fit()
            "eps_poles": np.asarray(eps, float),
            "R_poles": [np.asarray(R, np.complex128) for R in R_list],
            # per-COLUMN data (expanded by residue rank): for assembling V/H_b
            "eps_cols": np.diag(H_b).copy(),     # len == V_blk.shape[1]
            "V": V_blk.copy(),
        }

    V_cols = []
    eps_all = []
    block_to_bath_cols = [[] for _ in range(len(blocks))]

    for bidx, idx in enumerate(blocks):
        leader = leader_of_block[bidx]
        res = leader_results[leader]

        Nb = res["V"].shape[1]
        eps_block = np.asarray(res["eps_cols"], float)

        # robust guard if someone stored only pole energies by mistake
        if eps_block.shape[0] != Nb:
            if eps_block.shape[0] == 1:
                eps_block = np.full(Nb, float(eps_block[0]))
            else:
                raise ValueError(
                    f"Block {bidx}: eps length {eps_block.shape[0]} "
                    f"does not match V columns {Nb}. "
                    f"Store per-column eps from residues_to_bath."
                )

        V_block_full = np.zeros((Nimp, Nb), dtype=np.complex128)
        V_block_full[np.ix_(idx, list(range(Nb)))] = res["V"]

        col_start = len(eps_all)
        V_cols.append(V_block_full)
        eps_all.extend(eps_block.tolist())
        block_to_bath_cols[bidx] = list(range(col_start, col_start + Nb))

    V_full = np.hstack(V_cols) if V_cols else np.zeros((Nimp, 0), np.complex128)
    H_b_full = np.diag(np.asarray(eps_all, float)) if eps_all else np.zeros((0, 0), float)


    # spin assignment for bath columns
    alpha_imp = np.arange(0, M)
    beta_imp = np.arange(M, 2*M)
    alpha_mask_imp = np.zeros(Nimp, dtype=bool); alpha_mask_imp[alpha_imp] = True

    alpha_cols: List[int] = []
    beta_cols: List[int]  = []

    for j in range(V_full.shape[1]):
        # decide spin for bath column j based on where its couplings live
        coupl = V_full[:, j]
        # support sets
        supp = np.where(np.abs(coupl) > tol)[0]
        if supp.size == 0:
            # degenerate case: no visible coupling, just place by balancing counts
            target = alpha_cols if len(alpha_cols) <= len(beta_cols) else beta_cols
            target.append(j)
            continue

        frac_alpha = np.count_nonzero(alpha_mask_imp[supp]) / supp.size
        if np.isclose(frac_alpha, 1.0, atol=1e-8):
            alpha_cols.append(j)
        elif np.isclose(frac_alpha, 0.0, atol=1e-8):
            beta_cols.append(j)
        else:
            ## broken spin for this bath orbital, random assign
            #if rng.random() < 0.5:
            #    alpha_cols.append(j)
            #else:
            #    beta_cols.append(j)
            # replace the random assignment block
            norm_a = np.linalg.norm(coupl[:M])
            norm_b = np.linalg.norm(coupl[M:])
            if norm_a >= norm_b:
                alpha_cols.append(j)
            else:
                beta_cols.append(j)

    # enforce even total orbitals if requested
    Nbath = V_full.shape[1]
    if enforce_even_total and ((Nimp + Nbath) % 2 == 1):
        drop_j = _pick_smallest_weight_global(eps_all, V_full)
        if drop_j is not None:
            # drop column drop_j and its energy
            keep = np.ones(Nbath, dtype=bool)
            keep[drop_j] = False
            V_full = V_full[:, keep]
            H_b_full = np.diag(eps_all[keep])
            # update spin lists
            alpha_cols = [j for j in alpha_cols if j != drop_j]
            beta_cols  = [j for j in beta_cols if j != drop_j]
            # remap indices after drop
            old_to_new = {}
            new_idx = 0
            for j in range(Nbath):
                if keep[j]:
                    old_to_new[j] = new_idx
                    new_idx += 1
            alpha_cols = [old_to_new[j] for j in alpha_cols]
            beta_cols  = [old_to_new[j] for j in beta_cols]
            # also update block_to_bath_cols
            for b in range(len(block_to_bath_cols)):
                block_to_bath_cols[b] = [old_to_new[j] for j in block_to_bath_cols[b] if j in old_to_new]

    # final ordering: impurity α, bath α, impurity β, bath β
    # build permutation of full Hilbert space indices
    Nbath = V_full.shape[1]
    imp_alpha = list(alpha_imp)
    imp_beta  = list(beta_imp)

    # full index layout before permutation: [imp 0..Nimp-1, bath 0..Nbath-1]
    # we want [imp_alpha, bath_alpha, imp_beta, bath_beta]
    bath_offset = Nimp
    old_order = (
        imp_alpha
        + [bath_offset + j for j in alpha_cols]
        + imp_beta
        + [bath_offset + j for j in beta_cols]
    )
    perm = np.array(old_order, dtype=int)

    # assemble H_full before permutation
    top = np.hstack([himp, V_full])
    bottom = np.hstack([V_full.conj().T, H_b_full])
    H_full0 = np.vstack([top, bottom])

    # apply permutation
    H_full = H_full0[np.ix_(perm, perm)]

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

    Nimp = len(mapping["alpha_imp_idx"]) + len(mapping["beta_imp_idx"])
    perm = mapping["perm_full_to_spin_sorted"]

    # un-permute to block form
    P = np.eye(H_full.shape[0], dtype=H_full.dtype)[perm]
    H0 = P.T @ H_full @ P
    V_full = H0[:Nimp, Nimp:]
    H_b_full = H0[Nimp:, Nimp:]

    # use the same eta as your input when comparing (plumb it as an argument if needed)
    eta_compare = 0.02  # or pass-through from caller
    Delta_fit = utils.delta_from_bath(omega, H_b_full, V_full, eta=eta_compare)

    rel_err = utils.rel_l2_error(hyb, Delta_fit)
    chi2 = utils.cost_l2_integral(hyb, Delta_fit, omega, weight='const')
    vprint(1,f"rel L2 error = {rel_err:.3e},  chi2 = {chi2:.3e}")

    return H_full, mapping

def process_hyb_cost(
    omega: np.ndarray,
    hyb: np.ndarray,           # (Nw, Nimp, Nimp)
    himp: np.ndarray,          # (Nimp, Nimp)
    n_target_poles: int,
    *,
    eta_0: float,              # required by HybFitCost
    bounds_e: Optional[List[float]] = None,
    weight_func: str = 'const',
    broadening_Gamma: float = 0.0,
    tol: float = 1e-6,
    random_state: Optional[int] = None,
    enforce_even_total: bool = True,
    verbose: bool = False,
    logfile = None,
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
        Same keys as process_hyb(...), plus 'method_per_leader'.
    """
    rng = np.random.default_rng(random_state)

    omega = np.asarray(omega, float)
    hyb   = np.asarray(hyb, np.complex128)
    himp  = np.asarray(himp, np.complex128)

    assert hyb.ndim == 3 and hyb.shape[1] == hyb.shape[2], "hyb must be (Nw, Nimp, Nimp)"
    assert himp.shape[0] == himp.shape[1] == hyb.shape[1], "himp must match hyb"
    Nw, Nimp, _ = hyb.shape
    if Nimp % 2 != 0:
        raise ValueError("Nimp must be even for spin α|β split.")

    M = Nimp // 2
    mid = Nw // 2

    # 1) symmetry from reference one body
    Href = himp + hyb[mid]
    sym = symmetries.analyze_symmetries(np.asarray(Href), tol=tol, verbose=verbose)
    blocks: List[List[int]] = sym["blocks"]
    identical_groups: List[List[int]] = sym["identical_groups"]

    # enforce scalar leaders
    leaders = [g[0] for g in identical_groups]
    for leader in leaders:
        if len(blocks[leader]) != 1:
            raise ValueError(
                f"process_hyb_cost requires 1×1 leader blocks, but block {leader} has size {len(blocks[leader])}."
            )

    # 2) fit each 1×1 leader block with cost minimization
    leader_results: Dict[int, Dict] = {}
    for leader in leaders:
        idx = blocks[leader]  # single index [i]
        i = idx[0]
        hyb_blk = hyb[:, idx, :][:, :, idx]    # shape (Nw,1,1)

        if verbose:
            vprint(1,f"\n[CostFit] Fitting leader block {leader} (orbital {i}) with {n_target_poles} poles.")

        eps_poles, R_poles = fit(
            omega=omega,
            delta=hyb_blk,
            n_poles=n_target_poles,
            method='cost_minimization',
            eta_0=eta_0,
            broadening_Gamma=broadening_Gamma,
            weight_func=weight_func,
            bounds_e=bounds_e,
            logfile=logfile,
        )
        # turn residues into bath; for scalar, this yields exactly one column per pole
        H_b, V_blk = utils.residues_to_bath(eps_poles, R_poles)  # V_blk shape (1, Nb)

        leader_results[leader] = {
            "idx": idx,
            "method": "cost_minimization",
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
    alpha_cols: List[int] = []
    beta_cols:  List[int] = []

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
        "method_per_leader": {L: leader_results[L]["method"] for L in leaders},
    }

    # optional quick check at same eta as your input
    # un-permute back to block form and compare Δ_fit vs input
    P = np.eye(H_full.shape[0], dtype=H_full.dtype)[perm]
    H0 = P.T @ H_full @ P
    Vchk = H0[:Nimp, Nimp:]
    Hbck = H0[Nimp:, Nimp:]
    Delta_fit = utils.delta_from_bath(omega, Hbck, Vchk, eta=eta_0 + broadening_Gamma)
    rel_err = utils.rel_l2_error(hyb, Delta_fit)
    chi2 = utils.cost_l2_integral(hyb, Delta_fit, omega, weight='const')
    if verbose:
        vprint(1,f"[process_hyb_cost] rel L2 = {rel_err:.3e}, chi2 = {chi2:.3e}")

    return H_full, mapping

