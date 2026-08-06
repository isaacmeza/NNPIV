.. _sparse-linear-function-spaces:

Sparse Linear Function Spaces (:math:`\ell_1-\ell_1`)
======================================================

Consider linear learners :math:`g(a)=a^\top\alpha` and, for the nested
problem, :math:`h(b)=b^\top\beta`. Sparsity is measured by

.. math::

   \|\alpha\|_0=|\{j:\alpha_j\ne0\}|.

The estimators use convex :math:`\ell_1` relaxations. A signed coefficient is
represented by
:math:`\alpha=\rho^+-\rho^-`, where
:math:`\rho^+=(\alpha)^+` and :math:`\rho^-=(-\alpha)^+`. Thus
:math:`\|\alpha\|_1\leq V` is represented by the positive sub-simplex
:math:`\rho=(\rho^+;\rho^-)\geq0`, :math:`\|\rho\|_1\leq V`. Critics use the
same positive-negative lift of a unit :math:`\ell_1` ball.

The implementations average feasible iterates from
Optimistic-Follow-the-Regularized-Leader (OFTRL). For a convex-concave game,
OFTRL applies an optimistic gradient step to each player followed by its
entropy mirror map. Under the usual Lipschitz condition, the average iterate
has an :math:`O(T^{-1})` saddle residual. The fitted ``duality_gap_`` values
evaluate the constrained best-response problems for the objective implemented
by each class.

One-stage moment notation
-------------------------

Let

.. math::

   m(\alpha)=\mathbb E_n[Z(X^\top\alpha-Y)],
   \qquad Q_X=\mathbb E_n[XX^\top].

Estimator 1 - coefficient :math:`\ell_1` penalty
-------------------------------------------------

The linear-critic estimator solves

.. math::
   :label: minimax-sparse-est1

   \min_{\|\alpha\|_1\leq B}
   \max_{\|\theta\|_1\leq1}
   \theta^\top m(\alpha)+\lambda\|\alpha\|_1.

The critic best response gives :math:`\|m(\alpha)\|_\infty`; the learner and
critic updates use entropy on their lifted :math:`\ell_1` balls.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse_l1vsl1

Estimator 2 - empirical-L2 penalty
-----------------------------------

The ridge variant replaces the coefficient penalty by fitted-value
regularization:

.. math::
   :label: minimax-sparse-est2

   \min_{\|\alpha\|_1\leq B}
   \max_{\|\theta\|_1\leq1}
   \theta^\top m(\alpha)
   +\frac{\lambda}{2}\alpha^\top Q_X\alpha.

Its duality gap minimizes the quadratic learner response over the actual
:math:`\ell_1` ball. This includes boundary solutions and does not replace the
quadratic program by an eigenvalue bound.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse_ridge_l1vsl1

Nested moment notation
----------------------

For simultaneous estimation, let :math:`\mathbb E_p` and
:math:`\mathbb E_q` be the normalized empirical means used for the first and
second bridge moments. With rowwise multiplication in :math:`WA`, define

.. math::

   \begin{aligned}
   y_D&=\mathbb E_p[YD], & M_{AD}&=\mathbb E_p[AD^\top],\\
   M_{AC}&=\mathbb E_q[(WA)C^\top], &
   M_{BC}&=\mathbb E_q[BC^\top],\\
   r_1(\alpha)&=y_D-M_{AD}^\top\alpha, &
   r_2(\alpha,\beta)&=M_{AC}^\top\alpha-M_{BC}^\top\beta.
   \end{aligned}

Here ``W`` is an observation-level bridge multiplier, not a sample weight. If
``W=None``, it is one. Without subsetting,
:math:`\mathbb E_p=\mathbb E_q=\mathbb E_n`. With ``subsetted=True``, the two
binary masks select the normalized stage means. ``subset_ind1`` is required;
if ``subset_ind2`` is omitted, its complement is used. Two explicitly supplied
masks must each be nonempty but need not be disjoint or cover the sample.

Estimator 3 - ridge learners and quadratic critics
---------------------------------------------------

The quadratic-adversary estimator additionally defines

.. math::

   \begin{aligned}
   Q_A&=\mathbb E_n[AA^\top], & Q_B&=\mathbb E_n[BB^\top],\\
   Q_D&=\mathbb E_p[DD^\top], & Q_C&=\mathbb E_q[CC^\top].
   \end{aligned}

It minimizes over :math:`\|\alpha\|_1\leq V_1` and
:math:`\|\beta\|_1\leq V_2`, and maximizes over unit :math:`\ell_1` critic
balls, the objective

.. math::
   :label: minimax-sparse-est3-quadratic

   \begin{aligned}
   L(\alpha,\beta,\theta_1,\theta_2)
   ={}&2r_1^\top\theta_1-\theta_1^\top Q_D\theta_1
      +2r_2^\top\theta_2-\theta_2^\top Q_C\theta_2\\
     &+\mu'\alpha^\top Q_A\alpha
      +\mu\beta^\top Q_B\beta.
   \end{aligned}

The code uses the equivalent half-scaled objective. Its four update directions
are

.. math::

   \begin{aligned}
   \nabla_\alpha(L/2)&=-M_{AD}\theta_1+M_{AC}\theta_2+\mu'Q_A\alpha,\\
   \nabla_\beta(L/2)&=-M_{BC}\theta_2+\mu Q_B\beta,\\
   \nabla_{\theta_1}(L/2)&=r_1-Q_D\theta_1,\\
   \nabla_{\theta_2}(L/2)&=r_2-Q_C\theta_2.
   \end{aligned}

Let :math:`\|M\|_{\max}=\max_{jk}|M_{jk}|` and
:math:`m=\max\{\|M_{AD}\|_{\max},\|M_{AC}\|_{\max},
\|M_{BC}\|_{\max}\}`. Automatic tuning uses
:math:`\eta=(16m)^{-1}` when :math:`m>0`, learner rates
:math:`2\eta/V_1,2\eta/V_2`, and critic rates :math:`2\eta`. A zero learner
radius uses a zero rate for that learner. If :math:`m=0`, the base rate is one.
Wider learner balls therefore generally require more iterations.

By default, all ``n_iter`` played feasible iterates are averaged. Setting a
nonnegative ``tol`` checks ``duality_gap_upper_bound_`` every 50 iterations.
This quantity is a conservative upper bound on the full-scale saddle gap, not
an exact gap.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse2_ridge_quadratic_l1vsl1

Linear-critic ridge variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The linear-critic ridge class instead solves

.. math::

   \min_{\substack{\|\alpha\|_1\leq V_1\\
                   \|\beta\|_1\leq V_2}}
   \max_{\substack{\|\theta_1\|_1\leq1\\
                   \|\theta_2\|_1\leq1}}
   \theta_1^\top r_1+\theta_2^\top r_2
   +\frac{\mu}{2}
      (\alpha^\top Q_A\alpha+\beta^\top Q_B\beta).

It is a distinct objective: it does not contain the critic covariance terms
in :eq:`minimax-sparse-est3-quadratic`. Its gap calculation now solves the two
constrained quadratic learner responses exactly.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse2_ridge_l1vsl1

Estimator 4 - coefficient :math:`\ell_1` penalties
---------------------------------------------------

The nested coefficient-penalty variant uses linear critics:

.. math::

   \min_{\substack{\|\alpha\|_1\leq V_1\\
                   \|\beta\|_1\leq V_2}}
   \max_{\substack{\|\theta_1\|_1\leq1\\
                   \|\theta_2\|_1\leq1}}
   \theta_1^\top r_1+\theta_2^\top r_2
   +\mu(\|\alpha\|_1+\|\beta\|_1).

The two learner radii are applied separately, and the returned second critic
uses the complete positive-negative slice. Its duality gap uses the joint
learner gradient in the first bridge, rather than separate norm bounds.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse2_l1vsl1

Related :math:`\ell_2-\ell_2` variants
---------------------------------------

The :math:`\ell_2` classes replace both player balls by Euclidean balls.
``sparse_l2vsl2`` and ``sparse2_l2vsl2`` use coefficient-L2 penalties;
the two ``ridge`` classes use empirical fitted-value penalties. Their moment
violations and exact best-response gap calculations use the matching
:math:`\ell_2` geometry, including active trust-region boundaries and singular
learner covariance matrices.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.sparse_l2vsl2
   nnpiv.linear.sparse_ridge_l2vsl2
   nnpiv.linear.sparse2_l2vsl2
   nnpiv.linear.sparse2_ridge_l2vsl2

First-order solver variants
---------------------------

The public first-order classes in ``nnpiv.linear.sparse_linear`` implement the
same linear moment games with alternative online updates. Learner steps that
must satisfy :math:`\|\alpha\|_1\leq B` use the Euclidean projection onto the
entire :math:`\ell_1` ball, rather than clipping each coordinate. The
:math:`\ell_2` critic variants define the critic as zero when the empirical
moment is zero and report ``max_violation_`` in the matching
:math:`\ell_2` norm. ``L2OptimisticHedgeVsOGD`` also uses the immediately
preceding critic score in its optimistic update.

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.linear.SubGradientVsHedge
   nnpiv.linear.ProxGradientVsHedge
   nnpiv.linear.L2SubGradient
   nnpiv.linear.L2ProxGradient
   nnpiv.linear.L2OptimisticHedgeVsOGD
