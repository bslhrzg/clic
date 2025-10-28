# clic/scripts/thermal_cli.py
import sys
import tomli
import argparse
import numpy as np
from pydantic import ValidationError

# Import API and Pydantic models
from clic.config_models import ModelConfig, SolverConfig
from clic.api import Model, FockSpaceSolver 
from clic import hamiltonians
from clic.scripts.helpers import create_model_from_config 
from clic.postprocessing import StateAnalyzer 


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

    # --- MODIFICATION: Validate that nelec_range is set for this CLI ---
    nelec_setting = solver_config.solver.nelec_range
    if nelec_setting is None:
        print("ERROR: For 'clic-thermal-run', the 'nelec_range' parameter must be specified "
              "in the [solver] section. Use a range like [10, 14] or the string 'auto'.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Create the Model Object (using the helper) ---
    model = create_model_from_config(model_config)
    
    # --- 3. Run the FockSpaceSolver using the API ---
    print("\nInstantiating FockSpaceSolver...")
    solver = FockSpaceSolver(
        model=model,
        settings=solver_config.solver,
        nelec_range=nelec_setting # Pass the validated setting
    )
    
    # Run the solve method, passing the initial temperature from the config
    thermal_result = solver.solve(
        initial_temperature=solver_config.solver.initial_temperature
    )

    # Prune the result to get rid of small weight states
    prune_threshold = 1e-2  
    print(f"\nPruning thermal state with a hardcoded weight threshold of {prune_threshold:.1e}...")
    thermal_result.prune(threshold=prune_threshold)

    analyzer = StateAnalyzer(thermal_result, model)
    analyzer.print_analysis()

    # --- 4. Save the Output ---
    # The output filename is still controlled by the same config field
    output_file = solver_config.output.ground_state_file
    solver.save_result(output_file)
    print(f"\nThermal ground state result saved to '{output_file}'")

if __name__ == "__main__":
    main()