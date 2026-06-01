Estimators for Sequential and Simultaneous Nested NPIV
======================================================

.. admonition:: Start Here

   - Prerequisites: :doc:`Installation`
   - Diagnostic context before estimation: :doc:`Diagnostics`
   - API reference for these estimators: :doc:`mliv_overview`
   - Next after model fitting: :doc:`Semiparametrics`

.. contents:: On This Page
   :local:
   :depth: 2

Overview
--------

This section summarizes the optimization targets for nested NPIV estimators
under different function classes and links each target to practical
implementations (RKHS, random forest/ensemble, neural network, sparse/regularized
linear, and linear baselines).

Assumptions
-----------

- Observations are i.i.d. draws of :math:`(A, B, C, C', Y)`.
- Function classes :math:`\mathcal{G}, \mathcal{H}, \mathcal{F}, \mathcal{F}'`
  are chosen by the estimator family.
- Penalization and/or norm constraints are used to regularize finite-sample
  minimax estimation.

Notation
--------

- :math:`A`: first-stage endogenous treatment/features.
- :math:`B`: second-stage endogenous treatment/features.
- :math:`C'`: first-stage instruments for recovering :math:`g`.
- :math:`C`: second-stage instruments for recovering :math:`h`.
- :math:`g`: first-stage bridge function, :math:`h`: structural function of
  primary interest.

Estimator Objectives
--------------------

**Sequential Nested NPIV:**

Given observations :math:`(A_i, B_i, C_i)`, an initial estimator
:math:`\hat{g}`, and hyperparameter values :math:`(\lambda, \mu)`, estimate

.. math::
   \hat{h} = \arg\min_{h \in \mathcal{H}} \left[ \sup_{f \in \mathcal{F}} \left( 2 \cdot \text{loss}(f, \hat{g}, h) - \text{penalty}(f, \lambda) \right) + \text{penalty}(h, \mu) \right]

where
:math:`\text{penalty}(f, \lambda) = \mathbb{E}_m\{f(C)^2\} + \lambda \cdot \|f\|^2_{\mathcal{F}}`
and :math:`\text{penalty}(h, \mu) = \mu \cdot \|h\|^2_{\mathcal{H}}`.

Interpretation: the adversary :math:`f` probes IV moment violations for fixed
:math:`h`, while the learner regularizes complexity to stabilize inversion.

**Sequential Nested NPIV: Ridge:**

Given observations :math:`(A_i, B_i, C_i)`, an initial estimator
:math:`\hat{g}`, and hyperparameter :math:`\mu`, estimate

.. math::
   \hat{h} = \arg\min_{h \in \mathcal{H}} \left[ \sup_{f \in \mathcal{F}} \left( 2 \cdot \text{loss}(f, \hat{g}, h) - \text{penalty}(f) \right) + \text{penalty}(h, \mu) \right]

where :math:`\text{penalty}(f) = \mathbb{E}_m\{f(C)^2\}` and
:math:`\text{penalty}(h, \mu) = \mu \cdot \mathbb{E}_m\{h(B)^2\}`.

Interpretation: this variant emphasizes prediction-space regularization for
:math:`h` via :math:`\mathbb{E}[h(B)^2]`.

**Simultaneous Nested NPIV:**

Given observations :math:`(A_i, B_i, C_i, C_i')` and hyperparameters
:math:`(\mu', \mu)`, estimate

.. math::
   (\hat{g}, \hat{h}) = \arg\min_{g \in \mathcal{G}, h \in \mathcal{H}} \left[ \sup_{f' \in \mathcal{F}} \left( 2 \cdot \text{loss}(f', Y, g) - \text{penalty}(f') \right) + \text{penalty}(g, \mu') \right. \\
    \left. + \sup_{f \in \mathcal{F}} \left( 2 \cdot \text{loss}(f, g, h) - \text{penalty}(f) \right) + \text{penalty}(h, \mu) \right]

Interpretation: joint estimation can propagate first-stage uncertainty into the
second stage; diagnostics in :doc:`Diagnostics` help assess conditioning before
fitting.

Progressive Recipe
------------------

.. code-block:: python

   # Step 1: prepare arrays (A, B, C_prime, C, Y)
   from nnpiv.rkhs import RKHS2IVL2

   est = RKHS2IVL2(mu=0.1, mu_prime=0.1)

   # Step 2: fit simultaneous nested NPIV
   est.fit(A=A, B=B, C=C, D=C_prime, Y=Y)

   # Step 3: inspect structural predictions
   h_hat = est.predict(B_test)

Estimator Families
------------------

.. toctree::
   :maxdepth: 2

   longitudinal/RKHS
   longitudinal/Random Forest
   longitudinal/Neural Network
   longitudinal/Sparse Linear
   longitudinal/Regularized Linear
   longitudinal/Linear

Related Pages
-------------

- :doc:`Diagnostics`
- :doc:`Semiparametrics`
- :doc:`API`
