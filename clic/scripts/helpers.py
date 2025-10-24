# clic/scripts/helpers.py
import sys
import numpy as np
from clic.config_models import ModelConfig
from clic.api import Model
from clic import hamiltonians

def create_model_from_config(model_config: ModelConfig) -> Model:
    """
    Builds the API Model object from a validated configuration, handling
    automatic Nelec calculation for impurity models.
    """
    p = model_config.parameters
    h0, U = None, None
    M = 0

    if p.type in ['anderson_impurity_model', 'impurity_from_file']:
        if p.type == 'anderson_impurity_model':
            # First, build the integrals so we have h0
            u = p.interaction_u
            mu = u / 2.0 if p.mu == "u/2" else p.mu
            e_bath = np.linspace(p.bath.min_e, p.bath.max_e, p.bath.nb)
            V_bath = np.full(p.bath.nb, p.bath.hybridization_V)
            
            M = p.M_spatial
            h0, U = hamiltonians.get_impurity_integrals(M, u, e_bath, V_bath, mu)

            # Now, determine the total number of electrons
            if p.Nelec is None:
                # Automatic calculation is needed
                nelec_bath = hamiltonians.calculate_bath_filling(h0, p.M_imp)
                nelec_total = p.Nelec_imp + nelec_bath
                print(f"INFO: 'Nelec' not specified. Automatically determined bath filling: {nelec_bath}.")
                print(f"      Total Nelec set to {p.Nelec_imp} (imp) + {nelec_bath} (bath) = {nelec_total}.")
            else:
                # User provided a value, so we use it
                nelec_total = p.Nelec
                print(f"INFO: Using user-provided total Nelec = {nelec_total}.")

        
        elif p.type == 'impurity_from_file':
            print(f"Loading impurity model integrals from file: {p.filepath}")
            # First, load the integrals so we have h0 and M
            h0, U, M = hamiltonians.get_integrals_from_file(p.filepath, p.spin_structure)
            print(f"M = {M}")
            #print("diag h0 = ")
            #print(np.diag(h0))

            # Now, use the same logic as the AIM to determine total Nelec
            if p.Nelec is None:
                # Automatic calculation is needed
                nelec_bath = hamiltonians.calculate_bath_filling(h0, p.M_imp)
                nelec_total = p.Nelec_imp + nelec_bath
                print(f"INFO: 'Nelec' not specified. Automatically determined bath filling: {nelec_bath}.")
                print(f"      Total Nelec set to {p.Nelec_imp} (imp) + {nelec_bath} (bath) = {nelec_total}.")
            else:
                # User provided a value, so we use it
                nelec_total = p.Nelec
                print(f"INFO: Using user-provided total Nelec = {nelec_total}.")
            
        model = Model(h0=h0, U=U, M_spatial=M, Nelec=nelec_total)
        model.is_impurity_model = True
        model.imp_indices = list(range(p.M_imp)) 
        model.Nelec_imp = p.Nelec_imp
        return model

    elif p.type == 'from_file':
        print(f"Loading model integrals from file: {p.filepath}")
        h0, U, M = hamiltonians.get_integrals_from_file(p.filepath, p.spin_structure)
        # For this type, Nelec is required by Pydantic, so no calculation needed
        return Model(h0=h0, U=U, M_spatial=M, Nelec=p.Nelec)
    


    else:
        # This case should be unreachable if Pydantic validation is working
        print(f"ERROR: Unknown model parameter type '{p.type}'", file=sys.stderr)
        sys.exit(1)