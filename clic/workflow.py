# clic/workflow.py
from typing import Optional

# High-level logic imports from the API and helpers
from .api import create_model_from_config,GroundStateSolver, FockSpaceSolver, GreenFunctionCalculator
from .config_models import CalculationConfig
from . import results

def run_workflow(config: CalculationConfig):
    """
    Orchestrates the entire calculation based on the parsed config.
    """
    print("--- Starting CLIC Workflow ---")

    # --- 1. Model Creation (always happens) ---
    print("\n[Step 1/3] Creating model...")
    model = create_model_from_config(config)
    ground_state_result: Optional[results.ThermalGroundState] = None

    # The expected filename for the ground state is always derived from the output basename.
    # Define it here so it's always available.
    gs_filename = f"{config.output.basename}_gs.h5"

    # --- 2. Solver Execution (if configured) ---
    if config.solver:
        print("\n[Step 2/3] Solver section found. Running ground state calculation...")
        
        # Automatically select the correct solver
        if config.solver.nelec_range is not None:
            print("-> 'nelec_range' detected. Using FockSpaceSolver.")
            solver = FockSpaceSolver(
                model=model,
                settings=config.solver,
                nelec_range=config.solver.nelec_range
            )
        else:
            print("-> No 'nelec_range'. Using GroundStateSolver for fixed Nelec.")
            solver = GroundStateSolver(model, config.solver)

        ground_state_result = solver.solve()

        gs_filename = f"{config.output.basename}_gs.h5"
        print(f"Saving ground state result to '{gs_filename}'...")
        solver.save_result(gs_filename)
        
        # Optional: Run Analysis
        from .postprocessing import StateAnalyzer
        print("\n--- Post-Solver Analysis ---")
        analyzer = StateAnalyzer(ground_state_result, model)
        analyzer.print_analysis()

    else:
        print("\n[Step 2/3] No solver section found. Skipping ground state calculation.")

    # --- 3. Green's Function Calculation (if configured) ---
    if config.green_function:
        print("\n[Step 3/3] Green's function section found. Running GF calculation...")
        
        calculator = GreenFunctionCalculator(
            gf_config=config.green_function,
            output_config=config.output,
            ground_state_filepath=gs_filename
        )
                
        ws, G, A = calculator.run(ground_state_result=ground_state_result)
        
    else:
        print("\n[Step 3/3] No Green's function section found. Skipping GF calculation.")

    print("\n--- CLIC Workflow Finished ---")