API Documentation
=================

.. admonition:: Start Here

   - Estimation theory and estimator families: :doc:`Longitudinal`
   - Diagnostics workflow: :doc:`Diagnostics`
   - Semiparametric estimator narratives: :doc:`Semiparametrics`

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

This page is the reference hub for public APIs. Canonical mathematical and
workflow discussions remain in the domain pages; this page links to curated
reference surfaces for direct function/class lookup.

Longitudinal Estimator APIs
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 20 20

   * - API Group
     - When To Use
     - Required Inputs
     - Key Outputs
   * - RKHS (`RKHSIV*`, `RKHS2IV*`)
     - Kernel-based sequential/joint NPIV with regularization and CV.
     - `A, B, C, C', Y`
     - Fitted bridge/structural predictors
   * - Neural Network (`AGMM*`, `AGMM2L2`)
     - Adversarial minimax estimation with flexible representation learning.
     - `A, B, C, C', Y` tensors/arrays
     - Learned predictors and training diagnostics
   * - Ensemble / Random Forest (`Ensemble*`)
     - Oracle-style ensemble approximations for minimax objectives.
     - `A, B, C, C', Y`
     - Ensemble estimators for `g`/`h`
   * - Sparse/Regularized Linear + TSLS
     - Interpretable baselines and constrained optimization settings.
     - Matrix covariates + outcomes
     - Coefficients and predictions

.. toctree::
   :maxdepth: 2

   mliv_overview

Diagnostics APIs
----------------

Canonical diagnostics reference:
:doc:`diagnostics/Universal`

Semiparametric APIs
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 20 15

   * - API
     - Target
     - Core Inputs
     - Main Return
   * - `DML_npiv`
     - NPIV functional inference
     - `Y, D, Z, W` (+ options)
     - `theta, var, ci`
   * - `DML_mediated`
     - Mediation estimands
     - mediated DGP blocks
     - effect estimates + CI
   * - `DML_longterm`
     - Long-term treatment effects
     - long-term/surrogacy blocks
     - effect estimates + CI

.. toctree::
   :maxdepth: 2

   module_overview

Related Pages
-------------

- :doc:`Longitudinal`
- :doc:`Diagnostics`
- :doc:`Semiparametrics`
