Universal Diagnostics API
=========================

.. currentmodule:: nnpiv.diagnostics

Diagnostic A: Relative Well-posedness
-------------------------------------

Even if full :math:`L_2` verification is hard, one can check relative
well-posedness on the finite function class used in simulation or estimation.
Choose a feature map

.. math::

   b(A)=\{b_1(A),\ldots,b_J(A)\}^{\top}

and restrict to functions

.. math::

   u(A)=\theta^{\top}b(A), \qquad \theta \in \mathbb{R}^J.

Define

.. math::

   m_S(C') := E\{b(A)\mid C'\}, \qquad
   m_T(C) := E\{b(A)\mid C\}.

Then the induced quadratic forms are

.. math::

   \|Su\|_2^2=\theta^{\top}\Sigma_S\theta,
   \qquad
   \Sigma_S := E\!\left[m_S(C')m_S(C')^{\top}\right],

and

.. math::

   \|T_gu\|_2^2=\theta^{\top}\Sigma_T\theta,
   \qquad
   \Sigma_T := E\!\left[m_T(C)m_T(C)^{\top}\right].

The finite-dimensional bounded relative well-posedness condition is

.. math::

   \Sigma_T \preceq \kappa_J^2 \Sigma_S.

Equivalently, the sharp constant is the generalized eigenvalue bound

.. math::

   \kappa_J^2
   =
   \lambda_{\max}\!\left(
   \Sigma_S^{\dagger/2}\Sigma_T\Sigma_S^{\dagger/2}
   \right)
   =
   \sup_{\theta:\,\theta^{\top}\Sigma_S\theta>0}
   \frac{\theta^{\top}\Sigma_T\theta}{\theta^{\top}\Sigma_S\theta}.

If there exists :math:`\theta` with

.. math::

   \theta^{\top}\Sigma_S\theta=0
   \quad\text{but}\quad
   \theta^{\top}\Sigma_T\theta>0,

then relative well-posedness fails on that feature span.

In practice, use a small ridge :math:`\eta>0` for numerical stability. The
default implementation uses the feature Gram stabilization
:math:`\widehat{\Sigma}_I`:

.. math::

   \kappa_{J,\eta}^2
   = \lambda_{\max}\!\left[(\widehat{\Sigma}_S+\eta \widehat{\Sigma}_I)^{-1/2}
   \widehat{\Sigma}_T(\widehat{\Sigma}_S+\eta \widehat{\Sigma}_I)^{-1/2}\right].

where

.. math::

   \widehat{\Sigma}_I
   =
   \frac{1}{N}\sum_{i=1}^{N} b(A_i)b(A_i)^\top.

Set ``eta_mode='identity'`` to recover the older :math:`\eta I` stabilization.

Why :math:`\Sigma_I` vs :math:`I` and how to choose :math:`\eta`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The matrix :math:`\Sigma_I` is the feature-space second moment:

.. math::

   \Sigma_I = E[b(A)b(A)^\top],
   \qquad
   \widehat{\Sigma}_I = \frac{1}{N}\sum_{i=1}^N b(A_i)b(A_i)^\top.

It differs from identity-ridge in an important way:

* ``eta_mode='identity'`` regularizes by :math:`\eta\|\theta\|_2^2`;
* ``eta_mode='sigma_i'`` regularizes by
  :math:`\eta\,\theta^\top\Sigma_I\theta = \eta\,E[(\theta^\top b(A))^2]`,
  i.e., by function magnitude under :math:`P_A`.

So ``sigma_i`` is often more geometry-aware and less sensitive to arbitrary
feature scaling/rotation. In a basis close to orthonormal under :math:`P_A`,
the two choices are similar.

Finite-sample caveat:

* if :math:`\widehat{\Sigma}_I` has very small eigenvalues, then
  :math:`\eta\widehat{\Sigma}_I` may provide weak regularization in some
  directions relative to :math:`\eta I`;
* this can inflate :math:`\widehat{\kappa}_{J,\eta}` at very small ``eta``.

Practical ``eta`` guidance:

* to reduce extreme finite-sample :math:`\kappa`, increase ``eta``;
* to approximate the unregularized generalized-eigenvalue target, decrease
  ``eta``;
* always inspect an ``eta``-path (and preferably a ``J``-by-``eta`` grid)
  instead of a single value.

Unregularized limit:

* yes, conceptually ``eta=0`` corresponds to no ridge regularization
  (pseudo-inverse formulation);
* in finite samples this can be numerically unstable or effectively singular,
  so the implementation requires ``eta > 0`` for stability.

This is the primary pre-estimation simulation diagnostic.

Key output fields for failure-mode detection:

* ``nullspace_violation_flag``: ``True`` means the diagnostic found directions
  where :math:`\widehat{\Sigma}_S` is (near) null but
  :math:`\widehat{\Sigma}_T` still has signal. This is the empirical failure
  pattern for relative well-posedness on the tested feature span.
* ``nullspace_leakage_sigma_t_on_null_sigma_s`` (``nullspace_leakage`` in
  notebook tables): how much :math:`\widehat{\Sigma}_T` ``leaks`` into the
  near-null space of :math:`\widehat{\Sigma}_S` (larger is worse). Formally,
  it is the largest eigenvalue of
  :math:`U_0^\top \widehat{\Sigma}_T U_0`, where :math:`U_0` spans the
  near-null eigenspace of :math:`\widehat{\Sigma}_S`.
* ``stabilization_dominance_ratio``: rough size of ridge relative to
  :math:`\widehat{\Sigma}_S`,
  :math:`\|\eta R\|_{op}/\|\widehat{\Sigma}_S\|_{op}` with
  :math:`R=\widehat{\Sigma}_I` or :math:`I`. Large values mean regularization
  may be driving behavior more than the raw operator pair.
* ``max_diag_ratio_sigma_t_over_sigma_s`` (``max_diag_ratio`` in notebook
  tables): maximum coordinatewise ratio
  :math:`\mathrm{diag}(\widehat{\Sigma}_T) / \mathrm{diag}(\widehat{\Sigma}_S)`.
  This is a quick worst-direction proxy; large values suggest weak
  conditioning.

How to read :math:`\kappa`
^^^^^^^^^^^^^^^^^^^^^^^^^^

The number :math:`\kappa_{J,\eta}` is a restricted relative condition number.
Ignoring the ridge for a moment,

.. math::

   \kappa_J
   =
   \sup_{u \in \operatorname{span}(b):\,\|Su\|_2>0}
   \frac{\|T_g u\|_2}{\|Su\|_2}.

Thus :math:`\kappa_J^2` is the worst-case ratio of squared signal seen by
:math:`T_g` relative to squared signal seen by :math:`S` on the chosen feature
span.

Interpretation:

* :math:`\kappa_J \approx 1`: the two conditional-mean operators have similar
  strength on the feature span;
* :math:`\kappa_J < 1`: :math:`S` sees these directions at least as strongly as
  :math:`T_g`;
* large :math:`\kappa_J`: there are directions in the feature span that are much
  more visible to :math:`T_g` than to :math:`S`, so simultaneous estimation can
  have weak finite-sample curvature;
* divergent :math:`\kappa_J` as the feature span grows: evidence against
  bounded relative well-posedness for the full function class.

Gaussian linear-basis example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose :math:`(A,C,C')` are mean-zero jointly Gaussian and use the linear basis
:math:`b(A)=A`. Then

.. math::

   E[A\mid C]=\Sigma_{AC}\Sigma_{CC}^{-1}C,
   \qquad
   E[A\mid C']=\Sigma_{AC'}\Sigma_{C'C'}^{-1}C'.

Therefore

.. math::

   \Sigma_T
   =
   \Sigma_{AC}\Sigma_{CC}^{-1}\Sigma_{CA},
   \qquad
   \Sigma_S
   =
   \Sigma_{AC'}\Sigma_{C'C'}^{-1}\Sigma_{C'A}.

The diagnostic is the largest generalized eigenvalue of this pair. In the
scalar case with correlations
:math:`\rho_T=\operatorname{Corr}(A,C)` and
:math:`\rho_S=\operatorname{Corr}(A,C')`,

.. math::

   \Sigma_T=\operatorname{Var}(A)\rho_T^2,
   \qquad
   \Sigma_S=\operatorname{Var}(A)\rho_S^2,
   \qquad
   \kappa_J=\left|\frac{\rho_T}{\rho_S}\right|.

So for Gaussian data with a linear basis, :math:`\kappa` is literally the ratio
of how predictive :math:`C` is for :math:`A` relative to how predictive
:math:`C'` is for :math:`A`. If :math:`\rho_S=0` but :math:`\rho_T\ne0`, the
relative condition fails on this span.

For Gaussian scalar variables and an orthogonal Hermite basis, the degree
:math:`k` component scales like :math:`\rho^k`. Heuristically, if the sieve
contains degrees up to :math:`J`, then high-order components can behave like

.. math::

   \max_{1\leq k\leq J}
   \left|\frac{\rho_T}{\rho_S}\right|^k.

This explains why a DGP can look benign for linear features but deteriorate
quickly once nonlinear directions are added.

How :math:`\Sigma_S,\Sigma_T` are estimated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the DGP is known, use a large auxiliary Monte Carlo sample.

1. Choose features :math:`b(A)`.

   Use estimator-aligned features whenever possible: polynomial/Hermite bases
   (transparent), random Fourier features for RKHS, or learned
   representations/last-layer features for neural nets. A simple first pass is
   polynomial features in :math:`A`, including directions relevant for
   :math:`g_0(A)` (for example, cubic directions when :math:`g_0(A)=A_1^3`).

2. Estimate conditional means.

   Estimate :math:`m_S(C')=E[b(A)\mid C']` and
   :math:`m_T(C)=E[b(A)\mid C]` with flexible regressions on the auxiliary
   sample.

   If desired, nested Monte Carlo can be used:

   .. math::

      \widehat{m}_S(c')
      =
      \frac{1}{M}\sum_{m=1}^{M} b\!\left(A_{c'}^{(m)}\right),

   with draws :math:`A_{c'}^{(m)}\sim \mathcal{L}(A\mid C'=c')`, and similarly
   for :math:`\widehat{m}_T(c)`.

3. Build Gram matrices.

   .. math::

      \widehat{\Sigma}_S
      =
      \frac{1}{N}\sum_{i=1}^{N}
      \widehat{m}_S(C_i')\widehat{m}_S(C_i')^{\top},
      \qquad
      \widehat{\Sigma}_T
      =
      \frac{1}{N}\sum_{i=1}^{N}
      \widehat{m}_T(C_i)\widehat{m}_T(C_i)^{\top}.

4. Compute :math:`\kappa_{J,\eta}`.

   .. math::

      \kappa_{J,\eta}^2
      = \lambda_{\max}\!\left[(\widehat{\Sigma}_S+\eta\widehat{\Sigma}_I)^{-1/2}
      \widehat{\Sigma}_T(\widehat{\Sigma}_S+\eta\widehat{\Sigma}_I)^{-1/2}\right].

Interpretation of scale and stability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{diagnostic target: } \kappa_{J,\eta} \text{ modest and stable as } J \uparrow,\, \eta \downarrow.

Implementation defaults:

* Feature map: ``'rff'`` (alternatives: ``'polynomial'``, callable, or precomputed matrix).
* Conditional-mean learner: ridge regression.
* Stabilization: ``eta=1e-6``.

Practical interpretation:

* modest and stable :math:`\kappa_{J,\eta}` across feature sizes supports a DGP favorable to joint estimation;
* large values suggest weak finite-sample curvature for simultaneous estimation;
* instability as :math:`J` grows or :math:`\eta \downarrow 0` suggests possible span-level failure of relative well-posedness.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   relative_wellposedness_diagnostic
   relative_wellposedness_from_data
   relative_wellposedness_from_nested_npiv

Growing-sieve diagnostic (J and eta paths)
------------------------------------------

Because :math:`\kappa_{J,\eta}` depends on the chosen feature span, a single
fixed basis is only a local check. A more informative diagnostic computes the
same quantity over a growing sieve:

.. math::

   J_1 < J_2 < \cdots < J_K,
   \qquad
   \kappa_{J_1,\eta},\kappa_{J_2,\eta},\ldots,\kappa_{J_K,\eta}.

Stable, modest values as :math:`J` grows support relative well-posedness on the
explored sieve. Rapid growth means the increasingly rich feature class contains
directions that :math:`S` struggles to see relative to :math:`T_g`.

For random Fourier features, the sieve grid controls ``n_features``. For
polynomial features, it controls ``poly_degree``.

.. code-block:: python

   from nnpiv.diagnostics import relative_wellposedness_sieve_from_nested_npiv

   sieve = relative_wellposedness_sieve_from_nested_npiv(
       A=A,
       D=D,
       B=B,
       C=C,
       feature_map="polynomial",
       sieve_grid=[1, 2, 3, 4],
       eta_grid=[1e-4, 1e-6, 1e-8],
   )

   rows = sieve["rows"]
   summary = sieve["summary"]

Each row contains ``sieve_value`` (J), ``eta``, and ``kappa``. This supports:

* :math:`J \mapsto \widehat{\kappa}_{J,\eta}` for fixed :math:`\eta`;
* :math:`\eta \mapsto \widehat{\kappa}_{J,\eta}` for fixed :math:`J`.

The returned rows also include ``kappa_cummax`` so monotonic-envelope paths are
easy to plot.
Because finite-sample conditional-mean regressions are re-estimated at each
``J``, the raw empirical path may show small local non-monotonicity; the
cummax path is the finite-sample monotone envelope.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   relative_wellposedness_sieve_diagnostic
   relative_wellposedness_sieve_from_data
   relative_wellposedness_sieve_from_nested_npiv

Post-estimation error-direction diagnostic (kappa_eff)
------------------------------------------------------

After estimating the first stage, let :math:`e_g=\widehat g-g_0`. The
restricted empirical diagnostic is

.. math::

   \kappa_{\mathrm{eff}}
   =
   \frac{\|T_g e_g\|_2}{\|S e_g\|_2}.

In finite dimensions, ``relative_wellposedness_effective_diagnostic`` projects
:math:`e_g` onto the same feature span and computes this ratio with the same
estimated :math:`\widehat\Sigma_S,\widehat\Sigma_T`.

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   relative_wellposedness_effective_diagnostic
   relative_wellposedness_effective_from_data
   relative_wellposedness_effective_from_nested_npiv

Post-estimation sieve path (kappa_eff by J and eta)
---------------------------------------------------

.. code-block:: python

   from nnpiv.diagnostics import (
       relative_wellposedness_sieve_from_nested_npiv,
       relative_wellposedness_effective_sieve_from_nested_npiv,
   )

   pre = relative_wellposedness_sieve_from_nested_npiv(
       A=A, D=D, B=B, C=C,
       feature_map="rff",
       sieve_grid=[50, 100, 200, 400],
       eta_grid=[1e-4, 1e-6],
       random_state=123,
   )

   # e_g must be provided post-estimation (same row count as A).
   post = relative_wellposedness_effective_sieve_from_nested_npiv(
       A=A, D=D, B=B, C=C,
       e_g=e_g,
       feature_map="rff",
       sieve_grid=[50, 100, 200, 400],
       eta_grid=[1e-4, 1e-6],
       random_state=123,
   )

   pre_rows = pre["rows"]   # includes kappa, kappa_cummax
   post_rows = post["rows"] # includes kappa_eff, kappa_eff_cummax

Plotting the pre/post paths is then straightforward:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt

   pre_df = pd.DataFrame(pre_rows)
   post_df = pd.DataFrame(post_rows)

   fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
   for eta, g in pre_df.groupby("eta"):
       g = g.sort_values("sieve_value")
       ax[0].plot(g["sieve_value"], g["kappa"], marker="o", label=f"eta={eta:g}")
   ax[0].set_title("Pre: kappa_J,eta")
   ax[0].set_xlabel("J")
   ax[0].set_ylabel("kappa")
   ax[0].legend()

   for eta, g in post_df.groupby("eta"):
       g = g.sort_values("sieve_value")
       ax[1].plot(g["sieve_value"], g["kappa_eff"], marker="o", label=f"eta={eta:g}")
   ax[1].set_title("Post: kappa_eff,J,eta")
   ax[1].set_xlabel("J")
   ax[1].set_ylabel("kappa_eff")
   ax[1].legend()
   plt.tight_layout()

.. autosummary::
   :toctree: _autosummary
   :template: function.rst

   relative_wellposedness_effective_sieve_diagnostic
   relative_wellposedness_effective_sieve_from_data
   relative_wellposedness_effective_sieve_from_nested_npiv

Availability and workflow
-------------------------

* Diagnostic A is the default pre-estimation check whenever nested inputs
  :math:`(A, C, C')` are available.
* Simulation runners can enable A via ``diagnostics_opts.enabled=True`` and
  write CSV/JSON artifacts alongside Monte Carlo outputs.

Plug-and-play usage with dataset blocks
---------------------------------------

For minimal notebook usage, call the wrapper with dataset-level block selectors:

.. code-block:: python

   from nnpiv.diagnostics import relative_wellposedness_from_data

   # Example for nested NPIV data tuple -> dict
   data = {"A": A, "B": B, "C": C, "C_prime": D}
   out = relative_wellposedness_from_data(
       data,
       A="A",
       B="B",  # optional, accepted for (A,B,C,C') interface consistency
       C="C",
       C_prime="C_prime",
       feature_map="rff",
       n_features=300,
       eta=1e-6,
       random_state=123,
   )

   print(out["kappa"], out["kappa2"])

The same API supports:

* ``data`` as pandas ``DataFrame`` with ``A/C/C_prime`` as column names or lists;
* ``data`` as a mapping of arrays;
* ``data`` as a 2D array with integer/slice column selectors.
* Optional ``B`` selector can be passed but is not used by Diagnostic A.

Canonical nested NPIV shortcut
------------------------------

For the common simulation/notebook layout
``A, D, B, C, Y, tau_fn = dgps.get_data(...)``, use:

.. code-block:: python

   from nnpiv.diagnostics import relative_wellposedness_from_nested_npiv

   out = relative_wellposedness_from_nested_npiv(
       A=A,
       D=D,   # treated as C'
       B=B,   # accepted for interface consistency
       C=C,
       feature_map="rff",
       n_features=300,
       eta=1e-6,
       random_state=123,
   )

   print(out["kappa"], out["kappa2"])
