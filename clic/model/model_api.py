# clic/model/model_api.py

import numpy as np
from clic.model import hamiltonians, create_model_from_hyb
from clic.model.config_models import HybFitConfig

class Model:
    def __init__(self, h0, U):
        # 1. Basic Physics Data
        self.h0 = np.array(h0, dtype=np.complex128)
        self.U = np.array(U, dtype=np.complex128)
        
        self.M_spatial = self.h0.shape[0] // 2

        # 2. Impurity Data (Defaults to "Not an impurity")
        self.is_impurity_model = False
        self.imp_indices_spatial = []
        self.imp_indices_spinfull= []
        
        # Dictionary to hold hybridization and frequency data (w, iw, delta_original, delta_fitted)
        self.hyb_data = None 

    @classmethod
    def from_hybridization(cls, h_imp, U_imp, ws, hyb, n_bath_poles, eta_hyb,
                           fit_method="cost_minimization",
                           warp_kind = "emph0",
                           warp_w0 = 0.01,
                           eta_broad = 0.0,
                           iws = None):
        """
        Creates a Model by fitting the hybridization.
        """
        print(f"Fitting hybridization with {n_bath_poles} poles...")

        fit_config = HybFitConfig(n_target_poles=n_bath_poles, eta_in = eta_hyb, method=fit_method,
                                  warp_kind=warp_kind,warp_w0=warp_w0,eta_broad=eta_broad)
        
        # Calculate the full hamiltonian (Impurity + Bath)
        if n_bath_poles > 0:
            full_h0, hyb_approx = create_model_from_hyb.build_model_from_hyb(h_imp, ws, hyb, fit_config)
        else :
            print(f"n_bath_poles was set to 0, going with Hubbard I Approximation")
            full_h0 = h_imp 
            hyb_approx = None 
            
        # --- 2. Build the full U ---
        # Assume impurity is at the start
        M_total_spatial = full_h0.shape[0] // 2
        M_imp_spatial = h_imp.shape[0] // 2
        M_total_spinfull = 2 * M_total_spatial
        
        full_U = np.zeros((M_total_spinfull,)*4, dtype=U_imp.dtype)
        M_imp_spinfull = U_imp.shape[0]
        M_imp_spatial = M_imp_spinfull // 2

        impindex_alpha = [i for i in range(M_imp_spatial)]
        impindex_beta = [i for i in range(M_total_spatial, M_total_spatial+M_imp_spatial)]
        impindex = impindex_alpha + impindex_beta
        full_U[np.ix_(impindex, impindex, impindex, impindex)] = U_imp


        # --- 4. Create the Class ---
        # 'cls' just calls Model(...)
        model = cls(h0=full_h0, U=full_U)
        
        # --- 5. Fill in the Impurity Details ---
        model.is_impurity_model = True
        model.imp_indices_spatial = impindex_alpha
        model.imp_indices_spinfull = impindex


        
        model.hyb_data = {
            "ws": ws,
            "iws": iws,
            "original": hyb,
            "fitted": hyb_approx
        }
        
        return model
    

