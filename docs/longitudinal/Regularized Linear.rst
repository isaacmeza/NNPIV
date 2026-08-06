.. _regularized-linear-function-spaces:

Regularized Linear Function Spaces (:math:`\ell_2-\ell_2`)
==========================================================

These estimators use linear learners and critics constrained to Euclidean
balls. Their optimistic projected-gradient updates are the
:math:`\ell_2` analogue of the entropy updates in
:ref:`sparse-linear-function-spaces`. The critic objectives are linear: the
current implementations do not subtract a quadratic critic covariance term.

For each class, returned coefficients and critic weights average feasible
iterates. The fitted ``duality_gap_`` evaluates the exact constrained learner
best response. Consequently, it handles both active ball boundaries and
singular empirical covariance matrices. Moment violations are reported in the
same :math:`\ell_2` geometry as the critic ball.

One-stage estimators
--------------------

Let

.. math::

   m(\alpha)=\mathbb E_n[Z(X^\top\alpha-Y)],
   \qquad Q_X=\mathbb E_n[XX^\top].

Estimator 1 - coefficient-L2 penalty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The coefficient-regularized class solves

.. math::
   :label: minimax-regularized-est1

   \min_{\|\alpha\|_2\leq B}
   \max_{\|\theta\|_2\leq1}
   \theta^\top m(\alpha)+\frac{\lambda}{2}\|\alpha\|_2^2.

The critic best response is the unit direction of :math:`m(\alpha)`, with the
zero vector used when the moment is zero. The learner update is projected onto
the radius-``B`` Euclidean ball.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse_l2vsl2

Estimator 2 - empirical-L2 penalty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fitted-value ridge class replaces the coefficient norm by

.. math::
   :label: minimax-regularized-est2

   \min_{\|\alpha\|_2\leq B}
   \max_{\|\theta\|_2\leq1}
   \theta^\top m(\alpha)
   +\frac{\lambda}{2}\alpha^\top Q_X\alpha.

Its learner best response is a convex trust-region problem. The gap
calculation uses the unconstrained solution when feasible and otherwise solves
for the boundary multiplier; it does not approximate the response using a
single eigenvalue bound.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse_ridge_l2vsl2

Nested estimators
-----------------

Let :math:`\mathbb E_p` and :math:`\mathbb E_q` be the normalized empirical
means used for the two bridge moments, and define

.. math::

   \begin{aligned}
   r_1(\alpha)&=\mathbb E_p[D(Y-A^\top\alpha)],\\
   r_2(\alpha,\beta)&=
      \mathbb E_q[C((WA)^\top\alpha-B^\top\beta)],\\
   Q_A&=\mathbb E_n[AA^\top],\qquad
   Q_B=\mathbb E_n[BB^\top].
   \end{aligned}

``W`` is the observation-level multiplier on the first learner in the second
bridge moment; it is not a sample weight. ``W=None`` uses one. If
``subsetted=False``, both stage means use all observations. If
``subsetted=True``, ``subset_ind1`` is a required nonempty binary mask and an
omitted ``subset_ind2`` is its complement. Two explicit nonempty masks may
overlap or leave observations unused. The ridge matrices :math:`Q_A,Q_B`
always use the full sample.

Estimator 3 - coefficient-L2 penalties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simultaneous coefficient-regularized game is

.. math::
   :label: minimax-regularized-est3

   \min_{\substack{\|\alpha\|_2\leq V_1\\
                   \|\beta\|_2\leq V_2}}
   \max_{\substack{\|\theta_1\|_2\leq1\\
                   \|\theta_2\|_2\leq1}}
   \theta_1^\top r_1+\theta_2^\top r_2
   +\frac{\mu}{2}(\|\alpha\|_2^2+\|\beta\|_2^2).

Both learner and critic updates are optimistic gradient steps followed by
Euclidean projection onto their respective balls. The reported gap solves the
two isotropic learner best responses exactly.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse2_l2vsl2

Estimator 4 - empirical-L2 penalties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nested fitted-value ridge class solves

.. math::
   :label: minimax-regularized-est4

   \min_{\substack{\|\alpha\|_2\leq V_1\\
                   \|\beta\|_2\leq V_2}}
   \max_{\substack{\|\theta_1\|_2\leq1\\
                   \|\theta_2\|_2\leq1}}
   \theta_1^\top r_1+\theta_2^\top r_2
   +\frac{\mu}{2}
      (\alpha^\top Q_A\alpha+\beta^\top Q_B\beta).

The two gap components are exact Euclidean trust-region minima. An
eigendecomposition of each symmetric learner covariance supports both
full-rank and singular cases without changing the estimator's moment game.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse2_ridge_l2vsl2
