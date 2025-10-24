# clic/scripts/thermal_cli.py
import sys
import tomli
import argparse
import numpy as np
from pydantic import ValidationError

# Import API and Pydantic models
from clic.config_models import ModelConfig, SolverConfig
from clic.api import Model, FockSpaceSolver # <-- Use FockSpaceSolver
from clic import hamiltonians
from clic.scripts.helpers import create_model_from_config 


def main():
    parser = argparse.ArgumentParser(
        description="Run a Fock space calculation to find the thermal ground state using CLIC."
    )
    parser.add_argument("solver_config_file", type=str, help="Path to the solver TOML config.")
    args = parser.parse_args()

    try:
        # --- 1. Load and Validate Config Files (same as before) ---
        with open(args.solver_config_file, "rb") as f:
            solver_config = SolverConfig(**tomli.load(f))
        
        with open(solver_config.model_file, "rb") as f:
            model_config = ModelConfig(**tomli.load(f))

    except (FileNotFoundError, ValidationError) as e:
        print(f"ERROR: Failed to load or validate configuration.\n{e}", file=sys.stderr)
        sys.exit(1)

    # --- VALIDATE that the required config is present for this solver ---
    if solver_config.solver.nelec_range is None:
        print("ERROR: The 'nelec_range' parameter must be specified in the [solver] "
              "section of the solver config for a thermal calculation.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Create the Model Object (using the helper) ---
    model = create_model_from_config(model_config)
    
    # --- 3. Run the FockSpaceSolver using the API ---
    print("\nInstantiating FockSpaceSolver...")
    solver = FockSpaceSolver(
        model=model,
        settings=solver_config.solver,
        nelec_range=solver_config.solver.nelec_range
    )
    
    # Run the solve method, passing the initial temperature from the config
    thermal_result = solver.solve(
        initial_temperature=solver_config.solver.initial_temperature
    )

    # --- 4. Save the Output ---
    # The output filename is still controlled by the same config field
    output_file = solver_config.output.ground_state_file
    solver.save_result(output_file)
    print(f"\nThermal ground state result saved to '{output_file}'")

if __name__ == "__main__":
    main()