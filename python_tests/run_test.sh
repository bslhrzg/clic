#!/bin/bash

# --- Test Runner Script for CLIC ---

# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================="
echo "  RUNNING CLIC PYTHON TESTS              "
echo "========================================="

# Activate the virtual environment if it's not already active
# This makes the script more robust
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# The directory where the tests are located
TEST_DIR="./"

# Run each test script individually
# The `python -m` command is often more robust for finding modules
python "${TEST_DIR}/test_ed_tools.py"
python "${TEST_DIR}/test_h2o_sto3g.py"
python "${TEST_DIR}/test_hubbard_dimer.py"
python "${TEST_DIR}/test_impurity.py"
python "${TEST_DIR}/test_impurity_fixed.py"

python "${TEST_DIR}/test_green_hubbard_dimer.py"
python "${TEST_DIR}/test_kernel.py"
python "${TEST_DIR}/test_large_system.py"
python "${TEST_DIR}/test_matrix_elements.py"
python "${TEST_DIR}/test_operators.py"
python "${TEST_DIR}/test_operators_2.py"
python "${TEST_DIR}/test_wavefunction_io.py"

#python "${TEST_DIR}/test_h2o_631g.py"

echo "-----------------------------------------"
echo "  ALL TESTS PASSED SUCCESSFULLY!         "
echo "-----------------------------------------"
