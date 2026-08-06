Random Forest
=============

The ensemble estimators use regression forests as conditional-moment critics
and regression or classification forests as learner best responses.  Write

.. math::

    r_1(g)=g(A)-Y, \qquad r_2(g,h)=h(B)-Wg(A),

where :math:`W=1` by default.  The regression oracle fits a supplied target by
least squares.  The weighted-classification oracle fits the sign of a learner
direction, using its absolute value as the observation weight, and returns a
function bounded by ``max_abs_value``.

The optimization statements below describe the exact-oracle algorithms.
Their regret guarantees require exact best responses and the stated
convexity and sign-symmetry conditions on the empirical evaluation sets.
Random forests provide approximate best responses, so these results motivate
the implementation but are not finite-sample guarantees for a fitted forest.

Estimator 1
-----------

When the learner and critic classes are already norm constrained, the
one-stage estimator solves

.. math::

    \widehat g = \arg\min_{g\in\mathcal G}\max_{f'\in\mathcal F'}
    \mathbb E_n\!\left[2\{g(A)-Y\}f'(C')-f'(C')^2\right].

``EnsembleIV`` uses the equivalent sign convention in which the critic fits
:math:`Y-\bar g_{t-1}`.  Its weighted-classification learner follows the
direction :math:`f'_t` and the returned estimate averages the resulting
bounded learners.  Under exact oracles, convexity of
:math:`\mathcal F'_{C'}`, and sign symmetry, this is the standard
:math:`O\{(\log T+1)/T\}` ensemble approximation.

``EnsembleIVStar`` instead accumulates the critic through an adaptive linear
combination chosen by residual fit.  It is a useful heuristic, but it is not
the exact-oracle update covered by the preceding guarantee.

.. autosummary::
   :toctree: _autosummary
   :template: ensemble_class

   nnpiv.ensemble.EnsembleIV
   nnpiv.ensemble.EnsembleIVStar

Estimator 2
-----------

The ridge-regularized one-stage objective is

.. math::

    \widehat g = \arg\min_{g\in\mathcal G}\max_{f'\in\mathcal F'}
    \mathbb E_n\!\left[2\{g(A)-Y\}f'(C')-f'(C')^2\right]
    +\mu\mathbb E_n[g(A)^2].

``EnsembleIVL2`` uses the effective penalty

.. math::

    \delta_n=\frac{\texttt{delta\_scale}}{n^{\texttt{delta\_exp}}},
    \qquad \mu=\texttt{alpha}\,\delta_n^2,

with ``delta_scale=5`` and ``delta_exp=0.4`` when these arguments are
``'auto'``.  Given critics :math:`f'_1,\ldots,f'_t` fitted to
:math:`Y-\bar g`, the regression learner targets
:math:`(\mu t)^{-1}\sum_{s\leq t}f'_s(C')`.

When ``CV=True``, each candidate ``alpha`` is fitted on the training portion
of a fold.  A new critic is trained on the training residual and the candidate
is scored on the held-out conditional-moment payoff

.. math::

    \mathbb E_{\mathrm{test}}[2r f-f^2].

The scale with the smallest average score is stored as ``best_alpha_`` and is
then refitted on the full sample.

.. autosummary::
   :toctree: _autosummary
   :template: ensemble_class

   nnpiv.ensemble.EnsembleIVL2

Estimator 3
-----------

The ridge-regularized simultaneous estimator uses a common effective penalty
:math:`\mu` for the two learner functions:

.. math::

    (\widehat g,\widehat h)
    =\arg\min_{g\in\mathcal G,\,h\in\mathcal H}
    \max_{f'\in\mathcal F'}
    \mathbb E_n[2r_1(g)f'(D)-f'(D)^2]
    +\max_{f\in\mathcal F}
    \mathbb E_n[2r_2(g,h)f(C)-f(C)^2]
    +\mu\mathbb E_n[g(A)^2+h(B)^2].

Here :math:`D` instruments the first bridge equation and :math:`C`
instruments the second.  If the current critics fit
:math:`f'_t\simeq r_1` and :math:`f_t\simeq r_2`, the learner directions are

.. math::

    d_{g,t}=-f'_t(D)+Wf_t(C), \qquad d_{h,t}=-f_t(C).

``Ensemble2IVL2`` regresses :math:`g` and :math:`h` on the running averages of
:math:`d_g/\mu` and :math:`d_h/\mu`, respectively.  It uses the same
:math:`\mu=\texttt{alpha}\,\delta_n^2` definition as ``EnsembleIVL2``.
Cross-validation sums the two held-out conditional-moment payoffs and stores
the selected scale as ``best_alpha_``.

.. autosummary::
   :toctree: _autosummary
   :template: ensemble_class

   nnpiv.ensemble.Ensemble2IVL2

Subsetted estimator
^^^^^^^^^^^^^^^^^^^

With ``subsetted=True``, the first critic is fitted on the rows selected by
``subset_ind1`` and the second on those selected by ``subset_ind2``.  If the
second indicator is omitted, it is the complement of the first.  Explicitly
supplied nonempty binary indicators may overlap and need not exhaust the
sample.

For subset sizes :math:`p` and :math:`q`, the critic actions entering the
full-sample learner update are

.. math::

    \widetilde f'_i=\frac{n}{p}1\{i\in I_1\}f'(D_i), \qquad
    \widetilde f_i=\frac{n}{q}1\{i\in I_2\}f(C_i).

Thus :math:`d_g=-\widetilde f'+W\widetilde f` and
:math:`d_h=-\widetilde f`, which implements subset empirical moments together
with full-sample learner regularization.  Invalid or empty masks raise an
error.  ``n_burn_in`` performs preliminary critic/learner updates; the running
learner targets and prediction averages are reset afterward, and predictions
average only the following ``n_iter`` learners.

Estimator 4 (function class bounded)
------------------------------------

Without the ridge terms, the simultaneous objective is

.. math::

    (\widehat g,\widehat h)
    =\arg\min_{g\in\mathcal G,\,h\in\mathcal H}
    \max_{f'\in\mathcal F'}\mathbb E_n[2r_1(g)f'(D)-f'(D)^2]
    +\max_{f\in\mathcal F}\mathbb E_n[2r_2(g,h)f(C)-f(C)^2].

``Ensemble2IV`` fits weighted classifiers to the signs of
:math:`d_g=-f'+Wf` and :math:`d_h=-f`, with absolute directions as weights.
Each learner therefore takes values ``-max_abs_value`` or
``max_abs_value``.  The subset scaling and burn-in convention are exactly as
described above.

.. autosummary::
   :toctree: _autosummary
   :template: ensemble_class

   nnpiv.ensemble.Ensemble2IV
