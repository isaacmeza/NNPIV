Semiparametric Estimation
=========================

.. admonition:: Start Here

   - Estimator families and nuisance-learning context: :doc:`Longitudinal`
   - Conditioning checks before/after nuisance estimation: :doc:`Diagnostics`
   - Semiparametric API reference: :doc:`module_overview`
   - Setup and runnable notebook pointers: :doc:`Installation`

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

The goal is general-purpose learning and inference for a nonparametric causal
parameter :math:`\theta_0 \in \mathbb{R}`.
Many targets admit multiply robust moment functions with nuisance parameters
:math:`(\nu_0, \delta_0, \alpha_0, \eta_0)`. This section summarizes the
debiased machine learning (DML) meta-algorithm used in the package to convert
nuisance estimators into valid point estimates and confidence intervals.

Assumptions
-----------

- Nuisance estimators are trained on auxiliary folds and evaluated out-of-fold.
- Each nuisance learner converges in mean-square error at rates compatible with
  DML remainder control.
- Moment conditions are evaluated on held-out folds for orthogonalization.

Notation
--------

- :math:`\theta_0`: scalar target parameter.
- :math:`(\nu_0, \delta_0, \alpha_0, \eta_0)`: nuisance components entering
  the orthogonal score.
- :math:`I_\ell`: fold index set; :math:`I_\ell^c` its complement.

Debiased Machine Learning Meta-Algorithm
----------------------------------------

Given a sample :math:`(Y_i, W_i)` (:math:`i = 1, \ldots, n`), partition the
sample into folds :math:`I_\ell` (:math:`\ell = 1, \ldots, L`).
Denote by :math:`I^c_\ell` the complement of :math:`I_\ell`.

1. For each fold :math:`\ell`, estimate
   :math:`(\hat{\nu}_\ell, \hat{\delta}_\ell, \hat{\alpha}_\ell, \hat{\eta}_\ell)`
   from observations in :math:`I^c_\ell`.

2. Estimate :math:`\theta_0` as

   .. math::

      \hat{\theta} = \frac{1}{n} \sum_{\ell=1}^L \sum_{i \in I_\ell}
      \left[ \hat{\nu}_\ell(W_i) + \hat{\alpha}_\ell(W_i)\{Y_i - \hat{\delta}_\ell(W_i)\} + \hat{\eta}_\ell(W_i)\{\hat{\delta}_\ell(W_i) - \hat{\nu}_\ell(W_i)\} \right].

3. Estimate the :math:`(1 - \alpha)100\%` confidence interval as
   :math:`\hat{\theta} \pm c_\alpha \hat{\sigma} n^{-1/2}`, where
   :math:`c_\alpha` is the :math:`1 - \alpha/2` quantile of the standard
   Gaussian and

   .. math::

      \hat{\sigma}^2 = \frac{1}{n} \sum_{\ell=1}^L \sum_{i \in I_\ell}
      \left[ \hat{\nu}_\ell(W_i) + \hat{\alpha}_\ell(W_i)\{Y_i - \hat{\delta}_\ell(W_i)\} + \hat{\eta}_\ell(W_i)\{\hat{\delta}_\ell(W_i) - \hat{\nu}_\ell(W_i)\} - \hat{\theta} \right]^2.

Interpretation: fold-wise orthogonalization reduces sensitivity to nuisance
estimation errors, enabling practical inference with flexible first-stage
learners.

.. _localized-ratio-targets:

Localized Ratio Targets
-----------------------

Let :math:`H_i(v)` denote an uncentered score contribution and let :math:`a_i(v)` denote the loading that defines the target.
The estimator solves

.. math::

   \mathbb{E}_n[H_i(v)-a_i(v)\theta(v)]=0,
   \qquad
   \widehat\theta(v)
   =\frac{\mathbb{E}_n[H_i(v)]}{\mathbb{E}_n[a_i(v)]}.

The corresponding centered score contribution is

.. math::

   \widehat\phi_i(v)
   =\frac{H_i(v)-a_i(v)\widehat\theta(v)}
          {\mathbb{E}_n[a_i(v)]}.

For an ordinary average, :math:`a_i=1`.
For kernel localization, :math:`H_i(v)=\ell_i(v)H_i` and :math:`a_i(v)=\ell_i(v)`, so when the kernel loading has empirical mean one, :math:`\widehat\phi_i(v)=\ell_i(v)\{H_i-\widehat\theta(v)\}`.
A subgroup target uses a normalized group loading :math:`q_i`; a localized subgroup target uses :math:`q_i\ell_i(v)`.
Thus localization changes not only the numerator of the point estimate but also the centering used for variance and covariance.

This correction applies to every score route.
If the base score is a valid influence-function score (notably MR), the centered contribution is its localized influence value; OR, IPW, and hybrid retain their nuisance correctness and rate requirements.
For a conditional causal interpretation, include ``V`` in nuisance conditioning with ``include_V=True``.
The target is finite-bandwidth, conditional on the selected grid and bandwidth.

Progressive Recipe
------------------

.. code-block:: python

   # Step 1: configure nuisance learners and data blocks
   import numpy as np
   from nnpiv.rkhs import ApproxRKHSIVCV
   from nnpiv.semiparametrics import DML_npiv

   g_model = ApproxRKHSIVCV(kernel_approx="nystrom", n_components=200, cv=3)

   # Step 2: fit DML NPIV estimator
   dml = DML_npiv(Y=Y, D=D, Z=Z, W=W, model1=g_model, modelq1=g_model, n_folds=5)
   theta, var, ci = dml.dml()

   # Step 3: inspect estimate and uncertainty
   se = np.sqrt(var / len(Y))
   print(theta, se, ci)

Model-Specific Semiparametric APIs
----------------------------------

.. toctree::
   :maxdepth: 2

   semiparametrics/NPIV
   semiparametrics/Mediation Analysis
   semiparametrics/Long Term Effect
   semiparametrics/Dynamic Treatment Effect

Related Pages
-------------

- :doc:`Longitudinal`
- :doc:`Diagnostics`
- :doc:`API`
