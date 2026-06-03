# clic/solve/solver_api.py
import numpy as np
from scipy.linalg import eigh, block_diag
import copy

# Internal imports
import clic_clib as cc 
from clic.model import bath_transform, double_chains, hamiltonians
from clic.basis import basis_1p, basis_Np
from clic.ops import ops
from clic.solve import sci,fci
from clic.mf import mf
from clic.symmetries import symmetries


def solve_sector(h0_0,U_0,Nelec,clicvars):

    """
    solve in a given Nelec sector for the lowest states
    """

    M_imp = clicvars.M_imp

    #if clicvars.N_target is not None: 
    #    print("DEBUG: ENFORCING N_TARGET HERE")
    #    mu, w, Z, avgN = fit_mu_for_nelec_target(clicvars.N_target)
    #    M = self.base_model.h0.shape[0]
    #    mu_op = np.eye(M) * mu
    #    self.base_model.h0 -= mu_op

    def prepare_basis():
        method = clicvars.basis_prep_method
        print(f"Preparing one-particle basis using method: '{method}' for Nelec={Nelec}")
        
        if method == "none":
            return h0_0.copy(),U_0.copy(),None
        
        # --- Restricted Hartree Fock ---
        elif method == 'rhf':
            hmf, es, Vs, rho = mf.mfscf(h0_0, U_0, Nelec, spinsym_only=True)
            h0, U_mat = basis_1p.basis_change_h0_U(h0_0, U_0, Vs)
            return h0,U_mat,Vs

        # --- Impurity Bath Transforms ---
        elif method == "dbl_chain":
            
            print(f"Applying multi-orbital transformation for {M_imp} impurity orbitals...")
            
            imp_indices_spatial = [i for i in range(M_imp)]
            

            hmf, _, _, rho_mf = mf.mfscf(h0_0, U_0, Nelec)

            hmf_ab = basis_1p.transform_h0_alphafirst_to_interleaved(hmf)
            rhomf_ab = basis_1p.transform_h0_alphafirst_to_interleaved(rho_mf)

            Nimp = len(imp_indices_spatial) * 2
            hdc_ab, C_ab, meta = double_chains.double_chain_by_blocks(
                hmf_ab, rhomf_ab, Nimp, Nelec,
                symmetries.analyze_symmetries, double_chains.get_double_chain_transform_multi
            )

            hdc = basis_1p.transform_integrals_interleaved_to_alphafirst(hdc_ab)
            C = basis_1p.transform_integrals_interleaved_to_alphafirst(C_ab)
            
            h0 = C.conj().T @ h0_0 @ C 

            print(f"DEBUG: h0 transformed = ")
            print(h0)
            
            return h0,U_0.copy(),C
            
        else:
            raise NotImplementedError(f"Basis prep method '{method}' not implemented.")
  
    def run_sci(h0,U,C,seed):
        max_iter = clicvars.max_iter
        num_roots = clicvars.num_roots
        M = clicvars.M_spatial
        conv_tol = clicvars.conv_tol 
        prune_thr = clicvars.prune_thr
        Nmul = clicvars.Nmul 
        U_nz_ind = None 
        if clicvars.is_impurity_model: 

            U_nz_ind = [i for i in range(clicvars.M_imp)]
            U_nz_ind += [i + M for i in U_nz_ind]
            clicvars.imp_indices_spinfull = U_nz_ind

        result_obj = sci.selective_ci(
            h0=h0, 
            U=U,
            C=C,
            M=M, 
            Nelec=Nelec,
            seed=seed,
            generator=sci.hamiltonian_generator, 
            selector=sci.cipsi_select,
            num_roots=num_roots,
            max_iter=max_iter, 
            conv_tol=conv_tol,
            prune_thr=prune_thr,
            Nmul=Nmul, 
            verbose=True, 
            U_nz_ind=U_nz_ind,
        )
        return result_obj

    if Nelec == 0:
        vacuum_det = cc.SlaterDeterminant(clicvars.M_spatial, [], [])
        psis = [cc.Wavefunction(clicvars.M_spatial, [vacuum_det], [0+0j])]
        
        

        result = {
            "Nelec": Nelec,
            "energies": [0],
            "wavefunctions": psis,
            "C": None,                 # or None
            "basis": [vacuum_det],         # only if needed by your wf object
            }
        return result

    h0,U,C = prepare_basis()

    if clicvars.ci_type == "fci":
        result = fci.do_fci(
            h0=h0, U=U, C=C, M=clicvars.M_spatial, 
            Nelec=Nelec, num_roots=clicvars.num_roots, Sz=None, verbose=True
        )

    elif clicvars.ci_type == "sci":
        # --- Seed Generation ---
        if clicvars.basis_prep_method == 'rhf':
            initial_seed = basis_Np.get_rhf_determinant(Nelec, clicvars.M_spatial)

        else:
            fill_thr = 1e-1
            if clicvars.Nelec_imp is not None \
                and clicvars.M_spatial > clicvars.M_imp: # Not HIA right ? Else regular starting basis

                
                imp_indices_spatial = [i for i in range(M_imp)]            
                initial_seed = basis_Np.get_imp_starting_basis(
                    np.real(h0), 
                    Nelec, 
                    clicvars.Nelec_imp, 
                    imp_indices_spatial,
                    tol = fill_thr,
                )
            else: 
                initial_seed = basis_Np.get_starting_basis(np.real(h0), Nelec,tol=fill_thr)

        # --- Run SCI ---
        result = run_sci(h0,U,C,seed=initial_seed)


    return result


def solve_fockspace(h0_0, U_0, clicvars):
    """
    Scan particle-number sectors by repeatedly calling solve_sector().

    Returns
    -------
    dict
        A dictionary {Nelec: sector_result}.
    """

    print(f"Entering solve_fockspace")

    def solve_single_nelec(nelec, cache):
        if nelec in cache:
            return cache[nelec]

        print(f"\n--- Solving for Nelec = {nelec} ---")
        nelec_result = solve_sector(h0_0, U_0, nelec, clicvars)
        cache[nelec] = nelec_result
        return nelec_result

    def get_sector_ground_energy(sector_result):
        energies = np.asarray(sector_result["energies"])
        if energies.size == 0:
            raise ValueError("Sector result contains no energies.")
        return float(np.min(energies))

    def get_nelec_start_guess(h0):
        if clicvars.is_impurity_model and clicvars.Nelec_imp is not None:
            M_imp = clicvars.M_imp

            if clicvars.M_spatial > M_imp:
                fill_thr = 1e-3
                nelec_bath = hamiltonians.calculate_bath_filling(h0, M_imp, fill_thr)
                start = int(clicvars.Nelec_imp + nelec_bath)
                print(
                    f"INFO: 'auto' range. Estimated filling: "
                    f"{clicvars.Nelec_imp} (imp) + {nelec_bath} (bath) = {start}"
                )
                if clicvars.nelec_parity == 0 : 
                    if start % 2 == 1 : 
                        start += 1
                return start
            else:
                return int(clicvars.Nelec_imp)

        print("INFO: 'auto' range. No impurity info found, defaulting to half-filling.")
        return clicvars.M_spatial

    def find_optimal_nelec():
        nelec_start = get_nelec_start_guess(h0_0)

        print(f"\n--- Starting Automatic Search for Optimal Nelec (start = {nelec_start}) ---")

        energies = {}
        results_cache = {}

        # Initial point
        result0 = solve_single_nelec(nelec_start, results_cache)
        energies[nelec_start] = get_sector_ground_energy(result0)

        # Upward search
        nelec_curr = nelec_start
        while True:
            if clicvars.nelec_parity == 0 :
                nelec_next = nelec_curr + 2
            else:
                nelec_next = nelec_curr + 1
            if nelec_next > 2 * clicvars.M_spatial:
                break

            e_curr = energies[nelec_curr]
            result_next = solve_single_nelec(nelec_next, results_cache)
            e_next = get_sector_ground_energy(result_next)
            energies[nelec_next] = e_next

            if e_next >= e_curr:
                print("Energy increasing. Stopping upward search.")
                break

            nelec_curr = nelec_next

        # Downward search
        nelec_curr = nelec_start
        while True:
            if clicvars.nelec_parity == 0 :
                nelec_next = nelec_curr - 2
            else : 
                nelec_next = nelec_curr - 1
            if nelec_next < 0:
                break

            e_curr = energies[nelec_curr]
            result_next = solve_single_nelec(nelec_next, results_cache)
            e_next = get_sector_ground_energy(result_next)
            energies[nelec_next] = e_next

            if e_next >= e_curr:
                print("Energy increasing. Stopping downward search.")
                break

            nelec_curr = nelec_next

        nelec_min = min(energies, key=energies.get)
        print(f"Minimum found at Nelec={nelec_min} with E={energies[nelec_min]}")

        # Keep minimum and neighbors for later finite-T use
        final_results = {}
        for n in [nelec_min - 1, nelec_min, nelec_min + 1]:
            if n in results_cache:
                final_results[n] = results_cache[n]

        return final_results

    nelec_setting = clicvars.nelec_range
    print(f"clicvars.ci_type  = {clicvars.ci_type}")
    all_sector_results = {}

    # Strategy A: automatic search
    if nelec_setting == "auto":
        all_sector_results = find_optimal_nelec()

    # Strategy B: single fixed Nelec
    elif isinstance(nelec_setting, int):
        results_cache = {}
        result = solve_single_nelec(nelec_setting, results_cache)
        all_sector_results[nelec_setting] = result

    # Strategy C: fixed range/list
    else:
        if isinstance(nelec_setting, tuple):
            nelec_list = range(nelec_setting[0], nelec_setting[1] + 1)
        else:
            nelec_list = nelec_setting

        results_cache = {}
        for nelec in nelec_list:
            result = solve_single_nelec(nelec, results_cache)
            all_sector_results[nelec] = result

    return all_sector_results


def build_state_list_and_ne_dict(all_sector_results):
    """
    From
        all_sector_results[nelec] = {
            "Nelec": ...,
            "energies": ...,
            "wavefunctions": ...,
            "C": ...
        }

    build
        states  = [{"ne": ..., "psi": ..., "e": ..., "bw": 0.0}, ...]
        Ne_dict = {nelec: C}
    """
    states = []
    Ne_dict = {}

    for nelec, sector in all_sector_results.items():
        energies = sector["energies"]
        wavefunctions = sector["wavefunctions"]
        C = sector.get("C", None)

        Ne_dict[nelec] = C

        for e, psi in zip(energies, wavefunctions):
            states.append({
                "ne": nelec,
                "psi": psi,
                "e": e,
                "bw": 0.0,
            })

    states.sort(key=lambda s: s["e"])
    return states, Ne_dict


def set_boltzmann_weights(states, temperature, k_B):
    """
    Update in place the Boltzmann weights of a list of states.

    Each state must have at least:
        {"e": energy, "bw": ...}

    Returns
    -------
    Z : float
        Partition function built from shifted energies.
    """
    if len(states) == 0:
        return 0.0
    
    if len(states) == 1: 
        return 1.0

    beta = 1.0 / (k_B * temperature)

    energies = np.array([state["e"] for state in states], dtype=float)
    e0 = np.min(energies)

    w_unnorm = np.exp(-beta * (energies - e0))
    Z = np.sum(w_unnorm)
    w = w_unnorm / Z

    for state, wi in zip(states, w):
        state["bw"] = float(wi)


def prune_states(states, thr):
    """
    Return a new list containing only states with bw >= thr.
    """
    return [state for state in states if state["bw"] >= thr]



def fit_chemical_potential_for_target_N(
    states: list[tuple[float, int, "cc.Wavefunction"]],
    temperature_K: float,
    N_target: float,
    k_B_in_energy_per_K: float,
    mu_bounds: tuple[float, float] = (-10.0, 10.0),
    tol_N: float = 1e-8,
    max_iter: int = 200,
):
    """
    Fit mu such that <N>_mu = N_target for the grand-canonical ensemble of
    precomputed eigenstates.

    Weights:
        w_s(mu) ∝ exp(-beta * (E_s - mu * N_s))

    Parameters
    ----------
    states : list of (energy, nelec, wavefunction)
        Energies must be in the same energy unit as mu (e.g. Ry or eV).
    temperature_K : float
    N_target : float
        Desired average particle number.
    k_B_in_energy_per_K : float
        Boltzmann constant in the same energy unit as energies and mu.
        (You already have k_B_IN_RY_PER_K.)
    mu_bounds : (mu_lo, mu_hi)
        Bracketing interval for bisection.
    tol_N : float
        Converge when |<N>-N_target| < tol_N.
    max_iter : int

    Returns
    -------
    mu_star : float
    weights : np.ndarray, shape (n_states,)
        Normalized Boltzmann weights at mu_star.
    Z_unnormalized : float
        Sum of unnormalized weights (with the usual energy shift for stability).
    avgN : float
        Achieved <N> at mu_star.
    """

    beta = 1.0 / (k_B_in_energy_per_K * temperature_K)

    E = np.asarray([s[0] for s in states], dtype=float)
    E = E - np.mean(E)
    N = np.asarray([s[1] for s in states], dtype=float)

    def _weights_and_avgN(mu: float):
        # x_s = E_s - sign * mu * N_s
        x = E - mu * N
        x0 = np.min(x)  # stability shift (depends on mu!)
        w_unn = np.exp(-beta * (x - x0))
        Z = np.sum(w_unn)
        w = w_unn / Z
        avgN = float(np.dot(w, N))
        return w, Z, avgN

    def _f(mu: float):
        return _weights_and_avgN(mu)[2] - N_target

    mu_lo, mu_hi = float(mu_bounds[0]), float(mu_bounds[1])

    f_lo = _f(mu_lo)
    f_hi = _f(mu_hi)

    # Expand bracket if needed
    if f_lo * f_hi > 0:
        # Try geometric expansion a few times
        span = mu_hi - mu_lo
        for _ in range(30):
            span *= 2.0
            mu_lo_try = mu_lo - span
            mu_hi_try = mu_hi + span
            f_lo = _f(mu_lo_try)
            f_hi = _f(mu_hi_try)
            if f_lo * f_hi <= 0:
                mu_lo, mu_hi = mu_lo_try, mu_hi_try
                break
        else:
            # No bracket: target unreachable with provided states
            # Return best endpoint with diagnostic.
            w_lo, Z_lo, avgN_lo = _weights_and_avgN(mu_lo)
            w_hi, Z_hi, avgN_hi = _weights_and_avgN(mu_hi)
            if abs(avgN_lo - N_target) <= abs(avgN_hi - N_target):
                raise RuntimeError(
                    f"Could not bracket N_target. "
                    f"With mu={mu_lo:.6g}, <N>={avgN_lo:.6g}; "
                    f"with mu={mu_hi:.6g}, <N>={avgN_hi:.6g}. "
                    f"Likely missing relevant N sectors / excited states."
                )
            else:
                raise RuntimeError(
                    f"Could not bracket N_target. "
                    f"With mu={mu_hi:.6g}, <N>={avgN_hi:.6g}; "
                    f"with mu={mu_lo:.6g}, <N>={avgN_lo:.6g}. "
                    f"Likely missing relevant N sectors / excited states."
                )

    # Bisection (monotone at T>0 if spectrum coverage is adequate)
    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        w_mid, Z_mid, avgN_mid = _weights_and_avgN(mu_mid)
        f_mid = avgN_mid - N_target

        if abs(f_mid) < tol_N:
            return mu_mid, w_mid, Z_mid, avgN_mid

        # keep the sign change interval
        if f_lo * f_mid <= 0:
            mu_hi, f_hi = mu_mid, f_mid
        else:
            mu_lo, f_lo = mu_mid, f_mid

    # If not converged, return the midpoint result
    mu_mid = 0.5 * (mu_lo + mu_hi)
    w_mid, Z_mid, avgN_mid = _weights_and_avgN(mu_mid)

    return mu_mid, w_mid, Z_mid, avgN_mid


#def fit_mu_for_nelec_target(
#    N_target: float,
#    mu_bounds=(-50.0, 50.0),
#    tol_N=1e-8,
#    max_iter=200
#    ):
#    mu, w, Z, avgN = fit_chemical_potential_for_target_N(
#        states=._all_states,
#        temperature_K=self._temperature,
#        N_target=N_target,
#        k_B_in_energy_per_K=k_B_IN_RY_PER_K,
#        mu_bounds=mu_bounds,
#        tol_N=tol_N,
#        max_iter=max_iter,
#    )        
#    print(f"fit_mu_for_nelec_target: mu = {mu}")
#    for k, (E, n, wf) in enumerate(self._all_states):
#        self._all_states[k] = (E - mu*n, n, wf)
#    self._all_states.sort(key=lambda s: s[0])#

#    return mu, w, Z, avgN
    