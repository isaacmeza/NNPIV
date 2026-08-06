.. _neural-networks:

Neural Networks
===============

We now consider learner and critic classes represented by neural networks. The
joint estimator implemented by ``AGMM2L2`` approximately solves

.. math::

    (\hat g,\hat h)
    = \arg\min_{g,h}\max_{f_1,f_2}
    \left\{
    \mathbb E_n[2\{g(A)-Y\}f_1(D)-f_1(D)^2]
    + \mathbb E_n[2\{h(B)-Wg(A)\}f_2(C)-f_2(C)^2]
    + \mu\mathbb E_n[g(A)^2+h(B)^2]
    \right\}.

Here, :math:`A` and :math:`B` are the inputs to the first and second bridge
functions, while :math:`D` and :math:`C` are the instruments for the first and
second moment equations. The multiplier :math:`W` defaults to one. Thus the
first critic targets the residual :math:`g(A)-Y`, and the second targets
:math:`h(B)-Wg(A)`.

The ``learner_norm_reg`` argument supplies the common coefficient :math:`\mu`
on the empirical squared learner outputs. This is distinct from
``learner_l2`` and ``adversary_l2``, which apply weight decay to network
parameters during optimization. The minimax problem is trained with the
Optimistic Adam algorithm of `Daskalakis et al. (2017)
<https://arxiv.org/abs/1711.00141>`_, as proposed for adversarial conditional
moment estimation by `Dikkala et al. (2020)
<https://arxiv.org/abs/2006.07201>`_.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.neuralnet.oadam.OAdam

Subsetted Estimator
-------------------

With ``subsetted=True``, ``subset_ind1`` and ``subset_ind2`` select the
observations used for the first and second moment equations. If their sizes are
:math:`p` and :math:`q`, the corresponding masked full-sample losses are scaled
by :math:`n/p` and :math:`n/q`. This is equivalent to averaging each moment
loss over its own subset. The empirical-L2 penalties remain full-sample
averages.

Both indicators must be binary and nonempty. If ``subset_ind2`` is omitted,
the complement of ``subset_ind1`` is used. When both indicators are supplied,
they need not form a partition.

Predictions
-----------

``AGMM2L2.predict`` returns ``(h(B), g(A))``. By default, each prediction is
averaged across saved epochs after an optional burn-in. A final model or a
specific epoch can also be selected. If ``alpha`` is supplied for an averaged
prediction, the additional quantiles summarize variation across retained
epochs; they are heuristic stability bands, not sampling confidence
intervals.

Single estimator
----------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   nnpiv.neuralnet.AGMM
   nnpiv.neuralnet.KernelLayerMMDGMM
   nnpiv.neuralnet.CentroidMMDGMM
   nnpiv.neuralnet.KernelLossAGMM
   nnpiv.neuralnet.MMDGMM


Joint estimator
---------------

.. autosummary::
   :toctree: _autosummary
   :template: estimator_class

   nnpiv.neuralnet.AGMM2L2
