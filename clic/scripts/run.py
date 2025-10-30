# clic/scripts/run.py
import sys
import tomli
import argparse
from pydantic import ValidationError

from clic.config_models import CalculationConfig
from clic.workflow import run_workflow

def main():
    parser = argparse.ArgumentParser(description="Run a CLIC calculation workflow.")
    parser.add_argument("config_file", type=str, help="Path to the main TOML config file.")
    args = parser.parse_args()

    try:
        with open(args.config_file, "rb") as f:
            config_data = tomli.load(f)
            # Use the new CalculationConfig here!
            config = CalculationConfig(**config_data)
    except (FileNotFoundError, ValidationError) as e:
        # The error you are seeing is caught right here
        print(f"ERROR: Failed to load or validate configuration.\n{e}", file=sys.stderr)
        sys.exit(1)

    # Hand off to the main orchestrator
    run_workflow(config)

if __name__ == "__main__":
    main()