Dynamic Treatment Effect
========================

The package supports Debiased Machine Learning (DML) for two-period dynamic treatment effects. The observed data are :math:`W=(X_1,D_1,X_2,D_2,Y)`, where :math:`X_1` are baseline covariates, :math:`D_1 \in \{0,1\}` is the first-period treatment, :math:`X_2` are intermediate covariates, :math:`D_2 \in \{0,1\}` is the second-period treatment, and :math:`Y` is the final outcome.

.. admonition:: Dynamic treatment mean

   For a treatment path :math:`d=(d_1,d_2)`, the target estimand is :math:`\mathbb{E}[Y(d_1,d_2)]`.

Let :math:`S_1=(1,X_1^\top)^\top` and :math:`S_2=(1,X_1^\top,X_2^\top)^\top`. First define the path-specific sequential propensities

.. math::
   \begin{aligned}
   \pi_{1d}(S_1) &= \mathbb{P}(D_1=d_1 \mid S_1), \\
   \pi_{2d}(S_2) &= \mathbb{P}(D_2=d_2 \mid S_2,D_1=d_1).
   \end{aligned}

Following the package's nuisance notation, define

.. math::
   \begin{aligned}
   \nu_{d}(W) &= \mathbb{E}[\delta_{d}(W) \mid S_1,D_1=d_1], \\
   \delta_{d}(W) &= \mathbb{E}[Y \mid S_2,D_1=d_1,D_2=d_2], \\
   \alpha_{d}(W) &=
      \frac{\mathbb{1}(D_1=d_1,D_2=d_2)}
      {\pi_{1d}(S_1)\pi_{2d}(S_2)}, \\
   \eta_{d}(W) &=
      \frac{\mathbb{1}(D_1=d_1)}
      {\pi_{1d}(S_1)}.
   \end{aligned}

The implemented multiply robust score is

.. math::
   \begin{aligned}
   \psi_d(W)
   = \nu_d(W)
   &+ \alpha_d(W)\{Y-\delta_d(W)\} \\
   &+ \eta_d(W)\{\delta_d(W)-\nu_d(W)\}.
   \end{aligned}

This is the same bilinear influence-function structure used in the nested NPIV paper. In the paper's generic notation,

.. math::
   h_1(B_1)+h_3(B_3)\{Y-h_2(B_2)\}+h_4(B_4)\{h_2(B_2)-h_1(B_1)\},

the dynamic-treatment specialization is

.. math::
   h_1=\nu_d,\qquad
   h_2=\delta_d,\qquad
   h_3=\alpha_d,\qquad
   h_4=\eta_d.

Thus the dynamic score follows the same convention as the long-term and mediated scores: :math:`\nu_d` is the outer outcome/state regression, :math:`\delta_d` is the inner outcome regression, and :math:`\alpha_d,\eta_d` are the balancing weights multiplying the two residuals.

The first-period nuisance :math:`\nu_d(W)` can be fitted by regressing :math:`\hat{\delta}_d(W)` on :math:`S_1` among observations with :math:`D_1=d_1`, or with the sequential doubly robust learner (S-DRL) pseudo-outcome

.. math::
   \hat{\delta}_d(W)
   + \frac{\mathbb{1}(D_2=d_2)}{\hat{\pi}_{2d}(S_2)}
     \{Y-\hat{\delta}_d(W)\}.

Localization works as in the other semiparametric DML classes. Since dynamic effects localize on period-1 variables, ``V`` is appended to ``X1`` when ``include_V=True`` and is also used to form local kernel weights.

The outcome stage is always nested/sequential. By default, ``DML_dynamic`` uses an RKHS IV learner for both :math:`\delta_d` and :math:`\nu_d`, matching the default style of the other DML classes. Outcome learners should follow the package NPIV-style interface ``fit(Z, T, Y)`` and ``predict(T)``; linear notebook examples use ``nnpiv.tsls.tsls`` for this reason. To use distinct learners for the two nested regressions, pass them as ``model1=[delta_model, nu_model]``. Neural network learners can be used by setting ``nn_1=True`` for both stages or ``nn_1=[delta_is_nn, nu_is_nn]`` for stage-specific control. The propensity score model defaults to ``LogisticRegression()``; overlap trimming is handled through ``CHIM=True``.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   dml_dynamic.DML_dynamic
 
**References**

- Bradic, J., Ji, W., Zhang, Y., 2024. `High-dimensional inference for dynamic treatment effects <https://doi.org/10.1214/24-AOS2352>`_. The Annals of Statistics, 52(2), 415-440.
- Bodory, H., Huber, M., Laffers, L., 2022. `Evaluating (weighted) dynamic treatment effects by double machine learning <https://doi.org/10.1093/ectj/utac018>`_. The Econometrics Journal, Volume 25, Issue 3.
