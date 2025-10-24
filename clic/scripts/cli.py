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
from clic.scripts.helpers import create_model_from_config 


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
    model = create_model_from_config(model_config)
    
    # --- 3. Run the Solver using the API ---
    solver = GroundStateSolver(model, solver_config.solver)
    result = solver.solve()

    # --- 4. Save the Output ---
    solver.save_result(solver_config.output.ground_state_file)
    print(f"Output saved to '{solver_config.output.ground_state_file}'")

if __name__ == "__main__":
    main()