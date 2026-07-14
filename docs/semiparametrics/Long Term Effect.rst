Long-term Effect Analysis
==========================

Let :math:`X \in \mathbb{R}^{p}` be baseline covariates. Let :math:`D \in \{0, 1\}` indicate treatment assignment. Let :math:`M \in \mathbb{R}` be an intermediate/short-term outcome and :math:`Y \in \mathbb{R}` be a long-term outcome. An analyst may wish to measure the effect of :math:`D` on :math:`Y` (a long-term outcome), yet the experimental sample only includes :math:`M` (a short-term outcome).

If the analyst has access to an additional observational sample that includes the long-term outcome, then long-term causal inference is still possible. Specifically, assume the analyst has access to (i) an experimental sample, indicated by :math:`G=0`, where :math:`(D, M, X)` are observed; and (ii) an observational sample, indicated by :math:`G=1`, where :math:`(M, X, Y)` are observed, and :math:`D` is either observed or not. Depending on whether :math:`D` is also revealed in the observational sample will give rise to different assumptions that identify the long-term treatment effect. Specifically, the key identifying assumption when we do not observe :math:`D` is that the short-term outcome is a statistical surrogate for the long-term outcome, while the identifying assumption for the case when we observe :math:`D` is that unobserved confounding is mediated through the short-term outcome in the observational sample. Following `Athey et al., 2016 <https://arxiv.org/abs/1603.09326>`_, we refer to these models as *Surrogacy model* or *Latent unconfounded model*, respectively (`Athey et al., 2020 <https://arxiv.org/abs/2006.09676>`_).

.. admonition:: Long-term effect

   Formally, define the long-term counterfactual :math:`\mathbb{E}\left[Y^{(d)}\right]` as the counterfactual mean outcome for the full population in the thought experiment in which everyone is assigned treatment value :math:`D=d`.

Effects for the experimental or observational subpopulation use target-specific influence-function weights. Select them with ``sample_G="G=0"`` or ``sample_G="G=1"``, respectively. Subgroup targeting is not obtained merely by multiplying the pooled-population score by a group indicator: the leading regression term and the residual-correction weights change with the target population.

.. note::

   In addition to the population ATE, the implementation supports **conditional long‑term ATEs** within the experimental or observational subpopulations:
   :math:`\mathbb{E}[Y^{(1)}-Y^{(0)} \mid G=0]` and :math:`\mathbb{E}[Y^{(1)}-Y^{(0)} \mid G=1]`.
   Set ``sample_G="G=0"`` or ``sample_G="G=1"`` to target these effects; use ``sample_G="all"`` for the pooled-population target. Identification and influence‑function estimators (with the associated nuisance components) for the Surrogacy and Latent‑Unconfounded models are given in **Theorem 3.1** and **Theorem B.2** of Chen & Ritzwoller (2023).

.. admonition:: Project STAR application

   The Project STAR application sets ``sample_G="G=0"`` so that the proposed estimators target the experimental Project STAR population, while retaining both the STAR and NYC observations for nuisance estimation. For the heterogeneous analysis by prior ability, prior ability is included in the nuisance models, and the evaluation percentiles, bandwidth, and kernel normalization are computed using Project STAR observations with nonmissing prior ability. Thus, localization is specific to the target subgroup even though the NYC observations remain in the estimation sample to learn the long-term outcome mechanism.

Surrogacy Model
----------------

Define the regression and the conditional distribution

.. math::
   \begin{aligned}
   \gamma_{0}(m, x, g) & = \mathbb{E}[Y \mid M=m, X=x, G=g] \\
   \mathbb{P}(m \mid d, x, g) & = \mathbb{P}(M=m \mid D=d, X=x, G=g)
   \end{aligned}

For the pooled target, ``sample_G="all"``, the four nuisances associated with the model are

.. math::
   \begin{aligned}
   \nu_{0}(W) & = \int \gamma_{0}(m, X, 1) \mathrm{d} \mathbb{P}(m \mid d, X, 0) \\
   \delta_{0}(W) & = \gamma_{0}(M, X, 1) \\
   \alpha_{0}(W) & = \frac{\mathbb{1}_{G=1}}{\mathbb{P}(G=1 \mid M, X)} \frac{\mathbb{P}(d \mid M, X, G=0) \mathbb{P}(G=0 \mid M, X)}{\mathbb{P}(d \mid X, G=0) \mathbb{P}(G=0 \mid X)} \\
   \eta_{0}(W) & = \frac{\mathbb{1}_{G=0} \mathbb{1}_{D=d}}{\mathbb{P}(d \mid X, G=0) \mathbb{P}(G=0 \mid X)}
   \end{aligned}

and the long-term counterfactual is

.. math::
   \begin{aligned}
   \operatorname{LONG}(d) & = \mathbb{E}\left\{\int \gamma_{0}(m, X, 1) \mathrm{d} \mathbb{P}(m \mid d, X, 0)\right\} \\
   & =\mathbb{E}\left[\nu_0\left(W\right)+\alpha_0(W)\left\{Y-\delta_0(W)\right\}+\eta_0(W)\left\{\delta_0(W)-\nu_0(W)\right\}\right] 
   \end{aligned}


Latent Unconfounded Model
-------------------------

When we observe :math:`D` in the observational sample, the regression becomes

.. math::
   \begin{aligned}
   \gamma_{0}(m, x, g, d) & = \mathbb{E}[Y \mid M=m, X=x, G=g, D=d] \\
   \mathbb{P}(m \mid d, x, g) & = \mathbb{P}(M=m \mid D=d, X=x, G=g)
   \end{aligned}

For the pooled target, ``sample_G="all"``, the nuisances under this model are given by

.. math::
   \begin{aligned}
   \nu_{0}(W) & = \int \gamma_{0}(m, X, 1, d) \mathrm{d} \mathbb{P}(m \mid d, X, 0) \\
   \delta_{0}(W) & = \gamma_{0}(M, X, 1, d) \\
   \alpha_{0}(W) & = \frac{\mathbb{1}_{G=1}\mathbb{1}_{D=d}}{\mathbb{P}(G=1 \mid M, X, D=d)} \frac{\mathbb{P}(G=0 \mid M, X, D=d)}{\mathbb{P}(D=d \mid X, G=0) \mathbb{P}(G=0 \mid X)} \\
   \eta_{0}(W) & = \frac{\mathbb{1}_{G=0} \mathbb{1}_{D=d}}{\mathbb{P}(D=d \mid X, G=0) \mathbb{P}(G=0 \mid X)}
   \end{aligned}

The long-term counterfactual is

.. math::
   \begin{aligned}
   \operatorname{LONG}(d) & = \mathbb{E}\left\{\int \gamma_{0}(m, X, 1, d) \mathrm{d} \mathbb{P}(m \mid d, X, 0)\right\} \\
   & =\mathbb{E}\left[\nu_0\left(W\right)+\alpha_0(W)\left\{Y-\delta_0(W)\right\}+\eta_0(W)\left\{\delta_0(W)-\nu_0(W)\right\}\right] 
   \end{aligned}

Experimental-population target
------------------------------

Let :math:`\pi_0=\mathbb{P}(G=0)` and

.. math::
   q_0(W)=\frac{\mathbb{1}_{G=0}}{\pi_0}.

For the Surrogacy model, write
:math:`e_d(x)=\mathbb{P}(D=d\mid X=x,G=0)`,
:math:`e_d(m,x)=\mathbb{P}(D=d\mid M=m,X=x,G=0)`, and
:math:`s(m,x)=\mathbb{P}(G=1\mid M=m,X=x)`.  The experimental-target
residual weights are

.. math::
   \begin{aligned}
   \alpha_d^{(0)}(W)
   &=\frac{\mathbb{1}_{G=1}e_d(M,X)\{1-s(M,X)\}}
           {s(M,X)e_d(X)\pi_0},\\
   \eta_d^{(0)}(W)
   &=\frac{\mathbb{1}_{G=0}\mathbb{1}_{D=d}}
           {e_d(X)\pi_0}.
   \end{aligned}

For the Latent-Unconfounded model, let
:math:`\gamma(x)=\mathbb{P}(G=1\mid X=x)` and define the counterfactual
treatment propensity

.. math::
   \rho_d(m,x)
   =\mathbb{P}\{D=d\mid M(d)=m,X=x,G=1\}.

Although :math:`M(d)` is not jointly observed with both treatment states, this
propensity is identified by the observable Bayes-rule representation

.. math::
   \rho_d(m,x)
   =\frac{\mathbb{P}(G=1\mid M=m,D=d,X=x)}
          {\mathbb{P}(G=0\mid M=m,D=d,X=x)}
    \frac{\mathbb{P}(G=0\mid D=d,X=x)}
         {\mathbb{P}(G=1\mid D=d,X=x)}
    \mathbb{P}(D=d\mid X=x,G=1).

This is the propensity estimated by the corresponding ``sample_G="G=0"``
branch; it is generally not the ordinary observed-data propensity
:math:`\mathbb{P}(D=d\mid M=m,X=x,G=1)`.  Then

.. math::
   \begin{aligned}
   \alpha_d^{(0)}(W)
   &=\frac{\mathbb{1}_{G=1}\mathbb{1}_{D=d}}{\pi_0}
     \frac{1-\gamma(X)}{\gamma(X)}
     \frac{1}{\rho_d(M,X)},\\
   \eta_d^{(0)}(W)
   &=\frac{\mathbb{1}_{G=0}\mathbb{1}_{D=d}}
           {e_d(X)\pi_0}.
   \end{aligned}

For either model, the uncentered score for treatment arm :math:`d` is

.. math::
   H_d^{(0)}(W)
   =q_0(W)\nu_d(X)
    +\alpha_d^{(0)}(W)\{Y-\delta_d(W)\}
    +\eta_d^{(0)}(W)\{\delta_d(W)-\nu_d(X)\}.

Target-specific localization
----------------------------

For a localization variable :math:`V`, evaluation value :math:`v`, and
bandwidth :math:`\lambda`, the experimental-target weight is

.. math::
   \ell_{\lambda,v}^{(0)}(V)
   =\frac{K\{(V-v)/\lambda\}}
          {\mathbb{E}[K\{(V-v)/\lambda\}\mid G=0]}.

The finite-bandwidth target is a ratio.  Consequently, if
:math:`H_{d,\lambda}^{(0)}=\ell_{\lambda,v}^{(0)}H_d^{(0)}`, its centered
influence value is

.. math::
   H_{d,\lambda}^{(0)}(W)
   -q_0(W)\ell_{\lambda,v}^{(0)}(V)\theta_{d,\lambda}^{(0)},

not :math:`H_{d,\lambda}^{(0)}-\theta_{d,\lambda}^{(0)}`.  The implementation
uses this ratio centering for pointwise and uniform covariance estimation.
For a treatment contrast, subtract the corresponding expressions for
:math:`d=0` from those for :math:`d=1`.

The observational-population construction is analogous, with
:math:`q_1(W)=\mathbb{1}_{G=1}/\mathbb{P}(G=1)` and kernel normalization
conditional on :math:`G=1`. For the pooled target, :math:`q=1` and the kernel
is normalized over the pooled population.

When ``sample_G`` selects a subgroup, automatic bandwidth selection and kernel
normalization use that subgroup. If ``v_values`` is omitted, the implementation
localizes at the target-subgroup mean; explicitly supplied values should be
chosen from the intended target distribution.
Setting ``include_V=True`` appends :math:`V` to the covariates used by the
nuisance models in addition to using it for localization.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.semiparametrics.DML_longterm

**References**

- Athey, S., Chetty, R., Imbens, G., 2020. `Using experiments to correct for selection in observational studies <https://arxiv.org/abs/2006.09676>`_.
- Athey, S., Chetty, R., Imbens, G., Kang, H., 2016. `Estimating treatment effects using multiple surrogates: the role of the surrogate score and the surrogate index <https://arxiv.org/abs/1603.09326>`_.
- Chen, J., Ritzwoller, D. M., 2023. `Semiparametric estimation of long-term treatment effects <https://doi.org/10.1016/j.jeconom.2023.105545>`_. Journal of Econometrics, Volume 237, Issue 2, Part A.
