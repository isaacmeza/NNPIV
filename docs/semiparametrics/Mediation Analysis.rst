Mediation Analysis
==================

This module performs Debiased Machine Learning for mediation analysis, using joint estimation for longitudinal nonparametric parameters (in the Nested NPIV framework). It provides tools for estimating causal effects with mediation using a combination of machine learning models and instrumental variables techniques.


Different Estimands
-------------------

Our DML framework allows for various types of estimands. Here, we describe each estimand in detail:

- **ATE (Average Treatment Effect):** This estimand measures the average effect of treatment :math:`A` on outcome :math:`Y` across the entire population. It is defined as the difference in expected outcomes when the treatment is applied versus when it is not.

  .. math::
     E[Y(1)] - E[Y(0)]

- **Indirect Effect:** This estimand captures the effect of the treatment :math:`A` on the outcome :math:`Y` that is mediated through the intermediate variable :math:`M`. It is defined as the expected difference in :math:`Y` if all individuals received the treatment but the mediator was set to the level it would take without the treatment.

  .. math::
     E[Y(1, M(1)) - Y(1, M(0))]

- **Direct Effect:** This estimand measures the direct effect of treatment :math:`A` on outcome :math:`Y` independent of the mediator :math:`M`. It is defined as the expected difference in :math:`Y` if the treatment is applied versus not, while keeping the mediator at the level it would take without the treatment.

  .. math::
     E[Y(1, M(0)) - Y(0, M(0))]

- **E[Y1]:** This estimand represents the expected outcome when the treatment is applied to the entire population.

  .. math::
     E[Y(1)]

- **E[Y0]:** This estimand represents the expected outcome when the treatment is not applied to the entire population.

  .. math::
     E[Y(0)]

- **E[Y(1, M(0))]:** This estimand captures the expected outcome if the treatment is applied, but the mediator is set to the level it would take without the treatment.

  .. math::
     E[Y(1, M(0))]

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   dml_mediated.DML_mediated

**References**

- Cui, Y., VanderWeele, T. J., Tchetgen Tchetgen, E. J., 2020. `Proximal causal learning for mediation analysis <https://doi.org/10.48550/arXiv.2011.08411>`_.
- Oliver Dukes, Ilya Shpitser, Eric J Tchetgen Tchetgen, 2023. `Proximal mediation analysis <https://doi.org/10.1093/biomet/asad015>`_, Biometrika, Volume 110, Issue 4.
- Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., 2020. `Nonparametric identification of the mediation functional <https://doi.org/10.48550/arXiv.2009.10982>`_.
