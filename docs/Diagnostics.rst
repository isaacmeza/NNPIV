Estimation Diagnostics
======================

.. admonition:: Start Here

   - Install and environment setup: :doc:`Installation`
   - Core estimator objectives: :doc:`Longitudinal`
   - Canonical diagnostics API and math details: :doc:`diagnostics/Universal`
   - Post-estimation inference context: :doc:`Semiparametrics`

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

NNPIV includes estimator-agnostic diagnostics under ``nnpiv.diagnostics``.
These diagnostics are optional and non-invasive: they do not modify estimator
objectives, but help characterize whether a DGP appears favorable for nested
sequential vs simultaneous estimation before fitting models, and (optionally)
how severe relative conditioning is on the realized first-stage error
direction after estimation.

Assumptions
-----------

- Data blocks are row-aligned for :math:`(A, C, C')` diagnostics and optional
  :math:`e_g` post-estimation checks.
- Diagnostics are finite-dimensional approximations over chosen feature maps
  (for example RFF or polynomial sieves).
- Conditioning conclusions are empirical and should be read jointly with
  sieve-size and stabilization paths.

Notation
--------

- :math:`\kappa`: pre-estimation relative well-posedness diagnostic.
- :math:`\kappa_{\mathrm{eff}}`: post-estimation diagnostic along error
  direction :math:`e_g = \hat{g} - g_0`.
- :math:`\eta`: stabilization level in generalized-eigenvalue computations.

Progressive Recipe
------------------

.. code-block:: python

   # Step 1: prepare nested NPIV blocks
   from nnpiv.diagnostics import relative_wellposedness_from_nested_npiv

   # Step 2: run diagnostic A
   out = relative_wellposedness_from_nested_npiv(
       A=A,
       D=C_prime,
       B=B,
       C=C,
       feature_map="rff",
       n_features=300,
       eta=1e-6,
       random_state=123,
   )

   # Step 3: inspect key outputs
   print(out["kappa"], out["nullspace_violation_flag"], out["unstable_flag"])

Canonical Diagnostics Reference
-------------------------------

.. toctree::
   :maxdepth: 2

   diagnostics/Universal

Related Pages
-------------

- :doc:`Longitudinal`
- :doc:`Semiparametrics`
- :doc:`API`
