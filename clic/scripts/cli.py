# clic/scripts/cli.py
import sys
import tomli
import argparse
import numpy as np
from pydantic import ValidationError

# Import API and Pydantic models using absolute package paths
from clic.config_models import ModelConfig, SolverConfig
from clic.api import Model, GroundStateSolver
from clic import hamiltonians # Need this for building integrals from file

def main():
    parser = argparse.ArgumentParser(description="Run a ground state calculation using CLIC.")
    parser.add_argument("solver_config_file", type=str, help="Path to the solver TOML config.")
    args = parser.parse_args()

    try:
        # --- 1. Load and Validate Config Files ---
        with open(args.solver_config_file, "rb") as f:
            solver_config = SolverConfig(**tomli.load(f))
        
        with open(solver_config.model_file, "rb") as f:
            model_config = ModelConfig(**tomli.load(f))

    except (FileNotFoundError, ValidationError) as e:
        print(f"ERROR: Failed to load or validate configuration.\n{e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Create the Model Object (from config) ---
    p = model_config.parameters
    if model_config.source_type == 'parametric' and p.type == 'anderson_impurity_model':
        u = p.interaction_u
        mu = u / 2.0 if p.mu == "u/2" else p.mu
        e_bath = np.linspace(p.bath.min_e, p.bath.max_e, p.bath.nb)
        V_bath = np.full(p.bath.nb, p.bath.hybridization_V)
        h0, U = hamiltonians.get_impurity_integrals(p.M_spatial, u, e_bath, V_bath, mu)
        model = Model(h0=h0, U=U, M_spatial=p.M_spatial, Nelec=p.Nelec)
    else:
        print(f"ERROR: Unsupported model source '{model_config.source_type}'", file=sys.stderr)
        sys.exit(1)
    
    # --- 3. Run the Solver using the API ---
    solver = GroundStateSolver(model, solver_config.solver)
    result = solver.solve()

    # --- 4. Save the Output ---
    # (Simplified save logic for now. We can improve this later.)
    np.savez_compressed(
        solver_config.output.ground_state_file,
        energy=result["energy"],
        # Add other relevant data to save...
    )
    print(f"Output saved to '{solver_config.output.ground_state_file}'")

if __name__ == "__main__":
    main()