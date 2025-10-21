.. _api_reference:

API Reference
=============

This page provides the auto-generated documentation for the CLIC package.

Top-Level API (`clic`)
----------------------
The main ``clic`` package exposes the most important C++ classes and high-level
solver functions for convenience.

.. automodule:: clic
   :members: Wavefunction, SlaterDeterminant, Spin, SpinOrbitalOrder, get_impurity_integrals, green_function_block_lanczos_fixed_basis
   :undoc-members:

Hamiltonians (`clic.hamiltonians`)
----------------------------
Functions for building Hamiltonian integrals.

.. automodule:: clic.hamiltonians
   :members:
   :undoc-members:

One-Body Basis (`clic.basis_1p`)
--------------------------------
Functions for manipulating the one-particle basis 

.. automodule:: clic.basis_1p
   :members:
   :undoc-members:

Many-Body Basis (`clic.basis_Np`)
---------------------------------
Functions for generating and working with determinant bases.

.. automodule:: clic.basis_Np
   :members:
   :undoc-members:

Green's Function Solvers (`clic.gfs`)
-------------------------------------
Functions implementing the Lanczos algorithm.

.. automodule:: clic.gfs
   :members:
   :undoc-members:
