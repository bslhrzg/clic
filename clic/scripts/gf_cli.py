import sys
import tomli
import argparse
from pydantic import ValidationError

# Import API and Pydantic models
from clic.config_models import GfConfig
from clic.api import GreenFunctionCalculator

def main():
    parser = argparse.ArgumentParser(description="Run a Green's function calculation using CLIC.")
    parser.add_argument("gf_config_file", type=str, help="Path to the Green's function TOML config.")
    args = parser.parse_args()

    try:
        # Load and validate the configuration file
        with open(args.gf_config_file, "rb") as f:
            gf_config = GfConfig(**tomli.load(f))

    except (FileNotFoundError, ValidationError) as e:
        print(f"ERROR: Failed to load or validate configuration.\n{e}", file=sys.stderr)
        sys.exit(1)
    
    # Run the calculator using the API
    try:
        calculator = GreenFunctionCalculator(gf_config)
        calculator.run()
    except Exception as e:
        print(f"An error occurred during the Green's function calculation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()