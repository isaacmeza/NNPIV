Regularized Kernel Hilbert Space
================================
.. _rkhs_estimators:

In this section, the function classes :math:`\mathcal{G}`, :math:`\mathcal{H}`,
:math:`\mathcal{F}`, and :math:`\mathcal{F}'` are RKHSs. Let
:math:`\Phi_A:\mathcal{G}\rightarrow\mathbb{R}^n` have :math:`i` th row
:math:`\langle\phi(A_i),\cdot\rangle_{\mathcal G}`, with kernel matrix
:math:`K_A`; define the remaining feature operators and kernel matrices
analogously.

Classes ending in ``L2`` use empirical-:math:`L_2` learner regularization;
the other classes use RKHS-norm regularization. ``CV`` classes select the
regularization scale by cross-validation, and ``Approx`` classes replace full
kernel matrices with Nyström or random Fourier features.

The regularization sequence is based on

.. math::

   \delta_n=\texttt{delta\_scale}\,n^{-\texttt{delta\_exp}},
   \qquad \alpha=\texttt{alpha\_scale}\,\delta_n^4.

The automatic values of ``delta_scale`` and ``delta_exp`` are 5 and 0.4.


Closed form - Estimator 1
-------------------------

The RKHS-norm sequential estimator has the representer form
:math:`\hat g=\Phi_A^*\hat a`. With instruments :math:`C'`, define

.. math::

   M_{\delta}
   =K_{C'}^{1/2}
   \left\{\frac{K_{C'}}{2n\delta_n^2}+\frac{I_n}{2}\right\}^{-1}
   K_{C'}^{1/2}.

.. admonition:: Formula of minimizers

   The fitted coefficients solve

   .. math::

      \left(K_AM_{\delta}K_A+\alpha K_A\right)\hat a
      =K_AM_{\delta}Y.

   The regularized instrument system is solved directly; its inverse is not
   formed explicitly.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.rkhs.RKHSIV
   nnpiv.rkhs.RKHSIVCV

**Remark (Nyström/RFF approximation)**

For instrument and learner feature matrices :math:`F_{C'}` and :math:`F_A`, let

.. math::

   Q_{C'}=\frac{F_{C'}^\top F_{C'}}{2n\delta_n^2}+\frac{I}{2}.

The approximate estimator solves the feature-sized equation

.. math::

   \left(F_A^\top F_{C'}Q_{C'}^{-1}F_{C'}^\top F_A+\alpha I\right)\hat\theta
   =F_A^\top F_{C'}Q_{C'}^{-1}F_{C'}^\top Y.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.rkhs.ApproxRKHSIV
   nnpiv.rkhs.ApproxRKHSIVCV


Closed form - Estimator 2
-------------------------

The empirical-:math:`L_2` sequential estimator uses the orthogonal projector
onto the numerical range of the instrument kernel. If :math:`U_{C'}` contains
the retained positive eigenvectors of :math:`K_{C'}`, then
:math:`P_{C'}=U_{C'}U_{C'}^\top`.

.. admonition:: Formula of minimizers

   The fitted coefficients solve

   .. math::

      \left(K_AP_{C'}K_A+\alpha K_A^2\right)\hat a
      =K_AP_{C'}Y.

   Constructing :math:`P_{C'}` spectrally is the symmetric form of
   :math:`K_{C'}^\dagger K_{C'}` and retains the pseudoinverse's numerical-rank
   convention.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.rkhs.RKHSIVL2
   nnpiv.rkhs.RKHSIVL2CV

**Remark (Nyström/RFF approximation)**

The approximate estimator obtains the instrument range from a thin SVD of
:math:`F_{C'}` and contracts the empirical-:math:`L_2` normal equation through
a thin SVD of :math:`F_A`. It therefore solves only a learner-feature-sized
system.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.rkhs.ApproxRKHSIVL2
   nnpiv.rkhs.ApproxRKHSIVL2CV


Closed form - Estimator 3
-------------------------

The simultaneous empirical-:math:`L_2` estimator jointly estimates
:math:`g(A)` and :math:`h(B)`. In the public API, argument ``D`` denotes the
first-stage instruments :math:`C'`. Let

.. math::

   G=K_Aa,\qquad H=K_Bb,\qquad D_W=\operatorname{diag}(W),

where ``W`` is an observation-level multiplier in the bridge residual, not a
loss weight. It defaults to one. Ignoring a common positive scale, the fitted
criterion is

.. math::

   (Y-G)^\top P_{C'}(Y-G)
   +(H-D_WG)^\top P_C(H-D_WG)
   +\alpha\left(G^\top G+H^\top H\right),

where :math:`P_C` and :math:`P_{C'}` are symmetric numerical-range projectors.

Maximizers
^^^^^^^^^^

For each residual vector, the inner adversarial problem depends only on its
projection onto the corresponding empirical instrument range. The package
constructs these actions from symmetric spectral range bases rather than from
products of kernel pseudoinverses.

Minimizers
^^^^^^^^^^

.. admonition:: Formula of minimizers

   Write :math:`K_A^W=D_WK_A`. Both first-order conditions are solved in one
   symmetric block:

   .. math::

      \begin{bmatrix}
      K_AP_{C'}K_A+(K_A^W)^\top P_CK_A^W+\alpha K_A^2
      & -(K_A^W)^\top P_CK_B \\
      -K_BP_CK_A^W
      & K_BP_CK_B+\alpha K_B^2
      \end{bmatrix}
      \begin{bmatrix}\hat a\\\hat b\end{bmatrix}
      =
      \begin{bmatrix}K_AP_{C'}Y\\0\end{bmatrix}.

   The Moore--Penrose solution of this joint system defines the fitted
   coefficients.

``RKHS2IVL2`` and ``RKHS2IVL2CV`` implement this estimator. Their
finite-feature counterparts are ``ApproxRKHS2IVL2`` and
``ApproxRKHS2IVL2CV``.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.rkhs.RKHS2IVL2
   nnpiv.rkhs.RKHS2IVL2CV
   nnpiv.rkhs.ApproxRKHS2IVL2
   nnpiv.rkhs.ApproxRKHS2IVL2CV


Remark (Subsetted estimator)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``subsetted=True``, ``subset_ind1`` selects the :math:`C'` (argument
``D``) moment rows and ``subset_ind2`` selects the :math:`C` moment rows. If
``subset_ind2`` is omitted, the complement of ``subset_ind1`` is used. Each
subset action is lifted to the full sample and scaled by :math:`n/n_s`, after
which the same joint system applies.

The call ``predict(B_test)`` returns :math:`\hat h(B_{\rm test})`. Supplying
``A_test`` also returns the pair
:math:`(\hat h(B_{\rm test}),\hat g(A_{\rm test}))`.


Nyström approximation
^^^^^^^^^^^^^^^^^^^^^^

Full-kernel estimators require sample-sized kernel matrices. The approximate
estimators instead use finite feature matrices. For Nyström features,

.. math::

   \phi(x)\ \mapsto\ \check\phi(x)
   =K_{\mathcal SS}^{-1/2}K_{\mathcal Sx},

where :math:`\mathcal S` contains the landmarks. Random Fourier features are
also available through ``kernel_approx='rbfsampler'`` when ``kernel='rbf'``.
The feature maps are deterministic for fixed inputs and parameters.

The ``n_components`` parameter is interpreted as follows:

- a value in :math:`(0,1]` is a fraction of the fitting sample, rounded with a
  floor of 10 and capped by the available sample size;
- a value greater than 1 must be integer-like and is a fixed component count,
  again capped by the available sample size.

Thus ``n_components=1`` means 100 percent of the fitting sample.

For simultaneous estimators, let :math:`F_A,F_B,F_C,F_{C'}` denote the finite
feature matrices. The empirical-:math:`L_2` variants use the range action
:math:`F_CF_C^\dagger`; the RKHS-norm variants use the ridge action

.. math::

   F_C(F_C^\top F_C+I)^{-1}F_C^\top.

These actions are evaluated by feature contractions without forming the
corresponding :math:`n\times n` projectors. The two learner equations are
assembled as one :math:`(r_A+r_B)` block. For a common feature dimension
:math:`m`, the normal-equation work is therefore of order
:math:`O(nm^2+m^3)` in time and :math:`O(nm+m^2)` in memory.

Cross-validation
^^^^^^^^^^^^^^^^

All CV estimators use the actual training and validation fold sizes. Automatic
bandwidths are resolved from each training fold. Approximate CV estimators fit
Nyström/RFF maps on training rows and only transform held-out rows; the selected
maps are then refitted on the full sample.

The simultaneous CV classes accept a scalar, ``'auto'``, or a candidate grid
for ``gamma``. Their approximate versions also accept an ``n_components``
grid and jointly compare the bandwidth, component, and regularization grids.
Component candidates are scored in a common validation instrument space.
When ``expand_alpha_grid=True``, a simultaneous CV estimator expands the
regularization grid once if the best value is on its boundary.

Constructor parameters are left unchanged after fitting. The evaluated grid
is available as ``alpha_scales_``; selected values are stored in
``best_alpha_scale``, ``best_alpha``, ``best_gamma_``, and, for approximate
simultaneous CV, ``best_n_components_``. Detailed results are available in the
``cv_*_`` attributes.

Approximate simultaneous estimators accept dense arrays and SciPy sparse
matrices for ``A``, ``B``, ``C``, and ``D``. For empirical-:math:`L_2`
variants, choosing a component dimension close to the sample size can expose
near-null directions; a genuinely finite-rank choice, jointly tuned with the
regularization scale, is recommended.


Closed form - Estimator 3 (RKHS norm)
-------------------------------------

The simultaneous RKHS-norm variant replaces the empirical-:math:`L_2` learner
penalties by :math:`\alpha(\|g\|_{\mathcal G}^2+\|h\|_{\mathcal H}^2)` and
uses the package's unit-ridge instrument actions

.. math::

   P_C=(K_C+I_n)^{-1}K_C,\qquad
   P_{C'}=(K_{C'}+I_n)^{-1}K_{C'}.

Let :math:`F_A=U_A\Lambda_A^{1/2}` and
:math:`F_B=U_B\Lambda_B^{1/2}` be retained empirical RKHS coordinates and let
:math:`F_A^W=D_WF_A`.

.. admonition:: Formula of minimizers

   The implementation solves

   .. math::

      \begin{bmatrix}
      F_A^\top P_{C'}F_A+(F_A^W)^\top P_CF_A^W+\alpha I
      & -(F_A^W)^\top P_CF_B \\
      -F_B^\top P_CF_A^W
      & F_B^\top P_CF_B+\alpha I
      \end{bmatrix}
      \begin{bmatrix}\hat\theta_A\\\hat\theta_B\end{bmatrix}
      =
      \begin{bmatrix}F_A^\top P_{C'}Y\\0\end{bmatrix}.

   Solving jointly in the retained coordinates avoids eliminating either
   bridge equation. The approximate classes use the same block form with
   Nyström/RFF learner features and feature-contracted instrument actions.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.rkhs.RKHS2IV
   nnpiv.rkhs.RKHS2IVCV
   nnpiv.rkhs.ApproxRKHS2IV
   nnpiv.rkhs.ApproxRKHS2IVCV

The subsetted RKHS-norm estimator uses the same row selection and
:math:`n/n_s` scaling described above.
