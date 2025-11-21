# postprocessing.py
 
import numpy as np
from scipy.sparse import lil_matrix
from . import results
from clic.ops import ops
from ..api import Model

def _get_1p_Sz_matrix(M):
    sz_diag = np.concatenate([np.full(M, 0.5), np.full(M, -0.5)])
    return np.diag(sz_diag)

class StateAnalyzer:
    def __init__(self, result_obj, model: Model):
        if isinstance(result_obj, results.NelecLowEnergySubspace):
            res_dict = {result_obj.Nelec: result_obj}
            self.thermal_state = results.ThermalGroundState(res_dict, temperature=1e-6)
        elif isinstance(result_obj, results.ThermalGroundState):
            self.thermal_state = result_obj
        else:
            raise TypeError("result_obj must be NelecLowEnergySubspace or ThermalGroundState")
        self.model = model
        self._1p_op_cache = {}

    def _get_1p_operators(self):
        if "Sz" in self._1p_op_cache:
            return self._1p_op_cache
        num_imp_spatial = len(self.model.imp_indices_spatial)
        Sz_op = _get_1p_Sz_matrix(num_imp_spatial)
        self._1p_op_cache = {"Sz": Sz_op}
        return self._1p_op_cache

    def _calculate_single_state_stats(self, wf, nelec):
        stats = {}

        # many body expectations without building operators
        _, Sz_full = ops.expect_S2(wf, self.model.M_spatial)

        # S is only meaningful if eigenstate; still report the derived value
        #S = 0.5 * (np.sqrt(1.0 + 4.0 * np.real(S2)) - 1.0)
        #stats["S2"] = np.real(S2)
        #stats["S"]  = np.real(S)

        if self.model.is_impurity_model:
            imp_spinfull = self.model.imp_indices_spatial + [i + self.model.M_spatial for i in self.model.imp_indices_spatial]
            rdm_imp = ops.one_rdm(wf, self.model.M_spatial, block=imp_spinfull)
            stats["occ"] = float(np.sum(np.real(np.diag(rdm_imp))))
            stats["rdm"] = rdm_imp

            op_1p = self._get_1p_operators()
            exp_Sz_imp = np.trace(op_1p["Sz"] @ rdm_imp)
            stats["Sz"] = float(np.real(exp_Sz_imp))
        else:
            stats["occ"] = self.model.Nelec
            stats["rdm"] = None
            # use the full-system Sz from ops.expect_S2 return
            stats["Sz"] = float(np.real(Sz_full))

        return stats

    def print_analysis(self):
        all_states = self.thermal_state._all_states
        weights = self.thermal_state.boltzmann_weights
        if not all_states:
            print("No states to analyze.")
            return

        _, gs_energy, _ = self.thermal_state.find_absolute_ground_state()

        print("-"*50)
        print("RETAINED STATES:")
        print("-"*50)
        print(f"GS: e0 = {gs_energy:.12f}")

        all_stats = []
        avg_rdm_imp = 0
        for i, (energy, nelec, wf) in enumerate(all_states):
            stats = self._calculate_single_state_stats(wf, nelec)
            if self.model.is_impurity_model:
                avg_rdm_imp += weights[i] * stats["rdm"]
            all_stats.append(stats)

            print(f"e-e0: {energy - gs_energy:10.8f}, "
                  f"ne: {nelec}, "
                  f"weight: {weights[i]:10.4f}, "
                  f"occ: {stats['occ']:10.4f}, "
                  #f"S2: {stats['S2']:10.4f}, "
                  #f"S: {stats['S']:10.4f}, "
                  f"Sz: {stats['Sz']:10.4f}")

        if self.model.is_impurity_model:
            print("Saving thermally-averaged impurity density matrix...")
            np.savetxt("real-imp-dens.dat", np.real(avg_rdm_imp), fmt="% 8.5f")
            print("-> Saved 'real-imp-dens.dat'")
            np.savetxt("imag-imp-dens.dat", np.imag(avg_rdm_imp), fmt="% 8.5f")
            print("-> Saved 'imag-imp-dens.dat'")

        if len(all_states) > 1:
            avg_occ = float(np.sum(weights * [s["occ"] for s in all_stats]))
            #avg_S2  = float(np.sum(weights * [s["S2"] for s in all_stats]))
            #avg_S   = float(np.sum(weights * [s["S"]  for s in all_stats]))
            avg_Sz  = float(np.sum(weights * [s["Sz"] for s in all_stats]))
            print("thermal averages:")
            print(f"<occ> = {avg_occ:.8f}")
            #print(f"<S2>  = {avg_S2:.8f}")
            #print(f"<S>   = {avg_S:.8f}")
            print(f"<Sz>  = {avg_Sz:.8f}")
        print("-"*50)