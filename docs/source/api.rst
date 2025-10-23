.. _api_reference:

API Reference
=============

This page provides the auto-generated documentation for the CLIC package,
organized from high-level APIs to low-level modules.

High-Level Solver API (`clic.api`)
------------------------------------
These are the main user-facing classes for running calculations.

.. automodule:: clic.api
   :members: Model, GroundStateSolver, GreenFunctionCalculator
   :undoc-members:
   :show-inheritance:

Results Data Structure (`clic.results`)
-----------------------------------------
This class is used to store and manage the results of a calculation.

.. automodule:: clic.results
   :members: NelecLowEnergySubspace
   :undoc-members:
   :show-inheritance:

Configuration Models (`clic.config_models`)
-------------------------------------------
These Pydantic models define the structure of the ``.toml`` configuration files.

.. automodule:: clic.config_models
   :members: ModelConfig, SolverConfig, GfConfig, CiMethodConfig, AimParameters, FileDataSource
   :undoc-members:
   :show-inheritance:

Low-Level Modules
-----------------

These modules contain the core implementation details. They are useful for
advanced users or developers who wish to extend CLIC's functionality.

**CI Solvers (`clic.sci`)**

.. automodule:: clic.sci
   :members: selective_ci, do_fci
   :undoc-members:

**Green's Function (`clic.gfs`)**

.. automodule:: clic.gfs
   :members: green_function_block_lanczos_fixed_basis
   :undoc-members:

**Hamiltonians (`clic.hamiltonians`)**

.. automodule:: clic.hamiltonians
   :members:
   :undoc-members:

**Basis Manipulation (`clic.basis_1p`, `clic.basis_Np`)**

.. automodule:: clic.basis_1p
   :members:
   :undoc-members:

.. automodule:: clic.basis_Np
   :members:
   :undoc-members:

**C++ Backend (`clic.clic_clib`)**

.. automodule:: clic.clic_clib
   :members: Wavefunction, SlaterDeterminant
   :undoc-members: