Nested Nonparametric Instrumental Variable Regression
=====================================================

.. image:: https://readthedocs.org/projects/testingnn/badge/?version=latest
    :target: https://testingnn.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Overview
--------

This package aims to solve or estimate nonparametrically nested moment conditions. We analyze the closed form or approximate solutions under different function classes for the following estimators:

Estimators
----------

NPIV
~~~~
Given set of observations :math:`(Y, A, C')_i`; we want to estimate nonparametrically :math:`g` in :math:`\mathbb{E}\left[Y | C'\right]= \mathbb{E}\left[g(A) | C'\right]`, where A is the set of endogenous variables, and C' the set of instruments.
We solve the inverse problem adversarially:

.. math::

   \hat{g} = \arg \min_{g \in \mathcal{G}} \max_{f' \in \mathcal{F'}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] + \mu' \mathbb{E}_n \{ g(A)^2 \}

and we also consider norm regularization instead of ridge regularization:

.. math::

   \hat{g} = \arg \min_{g \in \mathcal{G}} \max_{f' \in \mathcal{F'}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] - \lambda \|f\|_{\mathcal{F}}^2 + \mu' \|g\|_{\mathcal{G}}^2

Nested NPIV
~~~~~~~~~~~
Whenever we have the set of observations :math:`(Y, A, B, C, C')_i`; and want to solve the system:

.. math::
    \mathbb{E}\left[Y | C'\right]= \mathbb{E}\left[g(A) | C'\right]

.. math::
    \mathbb{E}\left[g(A) | C\right]= \mathbb{E}\left[h(B) | C\right]

we estimate :math:`g` and :math:`h` by solving:

.. math::

   (\hat{g},\hat{h}) = \arg \min_{g \in \mathcal{G}, h \in \mathcal{H}} \max_{f' \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ g(A) - Y \right\} f'(C') - f'(C')^2 \right] + \mu' \mathbb{E}_n \{ g(A)^2 \}

.. math::

   + \max_{f \in \mathcal{F}} \mathbb{E}_n \left[ 2 \left\{ h(B) - g(A) \right\} f(C) - f(C)^2 \right] + \mu \mathbb{E}_n \{ h(B)^2 \}

and similarly when using norm-regularization.

Implementation
--------------

Longitudinal Estimation
~~~~~~~~~~~~~~~~~~~~~~~

This package implements longitudinal estimation of functions :math:`g` and :math:`h` for several function classes:

- RKHS
- Random Forest
- Neural Networks
- Sparse Linear
- Linear

Semiparametric Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

The package also implements debiased machine learning for estimation of a functional of the nuisance longitudinal parameter :math:`g` or :math:`h`:

.. math::
    \theta = \mathbb{E}\left[h(B)\right]

based on constructing orthogonal moments for:

- Mediation analysis
- Long term effect

