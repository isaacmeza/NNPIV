NPIV
====

The package supports Debiased Machine Learning (DML) for the semiparametric model where the parametric part is a functional of a standard nonparametric instrumental variables (NPIV) inverse problem. 

Localization
------------

When ``V`` is supplied, ``DML_npiv`` estimates the finite-bandwidth target

.. math::

   \theta_\lambda(v)
   =\frac{\mathbb{E}[K\{(V-v)/\lambda\}H]}
          {\mathbb{E}[K\{(V-v)/\lambda\}]},

where :math:`H` is the uncentered MR, OR, or IPW score selected by the user.
Writing :math:`\ell_{\lambda,v}=K/\mathbb{E}[K]`, the centered influence value
is :math:`\ell_{\lambda,v}\{H-\theta_\lambda(v)\}`, not
:math:`\ell_{\lambda,v}H-\theta_\lambda(v)`. Pointwise and uniform inference
use this ratio centering. Without ``V``, the loading is one and the estimator
reduces to the ordinary average-score calculation.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.semiparametrics.DML_npiv

**References**

- Chernozhukov, V., Newey, W. K., Singh, R., 2023. `A simple and general debiased machine learning theorem with finite-sample guarantees <https://doi.org/10.1093/biomet/asac033>`_.
