.. _usage_guide:

Usage Guide & Examples
======================

This guide shows a typical workflow for running a ground state calculation
using CLIC's command-line interface.

Step 1: Define the Physical Model
---------------------------------

First, create a model configuration file (e.g., ``model.toml``). This file
describes the Hamiltonian of your system. You can define a parametric model
or load integrals from a file.

**Example: Loading integrals from an HDF5 file**

.. code-block:: toml
   :caption: model.toml

   model_name = "H2O-STO3G-from-file"

   [parameters]
   type = "from_file"
   filepath = "./h2o_sto3g.h5"
   Nelec = 10
   # Optional: specify the spin structure if not alpha_first
   # spin_structure = "interleaved"

Step 2: Define the Solver Settings
----------------------------------

Next, create a solver configuration file (e.g., ``solver.toml``). This file
points to the model file and specifies the numerical methods and parameters
to use.

.. code-block:: toml
   :caption: solver.toml

   model_file = "./model.toml"

   [solver]
   basis_prep_method = "none" # or "rhf", "dbl_chain", etc.

   [solver.ci_method]
   type = "sci"
   generator = "hamiltonian_generator"
   selector = "cipsi"
   num_roots = 1
   max_iter = 10
   conv_tol = 1.0e-6
   prune_thr = 1.0e-6
   Nmul = 1.0

   [output]
   ground_state_file = "h2o_results.h5"

Step 3: Run the Calculation
---------------------------

Execute the solver from your terminal using the ``clic-run`` command:

.. code-block:: bash

   clic-run solver.toml

This will perform the calculation and save the results to the file specified
in ``output.ground_state_file`` (``h2o_results.h5`` in this case).

Step 4: Load and Analyze Results
--------------------------------

The results are saved in a structured HDF5 file and can be loaded using the
``NelecLowEnergySubspace`` class for further analysis.

.. code-block:: python
   :caption: analyze.py

   from clic.results import NelecLowEnergySubspace

   # Load the entire result object from the file
   results = NelecLowEnergySubspace.load("h2o_results.h5")

   # Access the computed properties
   print(f"Calculation for Nelec = {results.Nelec}")
   print(f"Ground state energy: {results.ground_state_energy:.8f} Ha")
   print("All computed energies:")
   print(results.energies)

   # The ground state wavefunction is also available
   gs_wavefunction = results.ground_state_wavefunction