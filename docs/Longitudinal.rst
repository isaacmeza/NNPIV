Estimators for Sequential and Simultaneous Nested NPIV
======================================================

In this section, we analyze the closed-form or approximate solutions under different function classes for the following estimators:

**Sequential Nested NPIV:**

Given observations :math:`(A_i, B_i, C_i)` in \tr, an initial estimator :math:`\hat{g}` which may be estimated in \tr, and hyperparameter values :math:`(\lambda, \mu)`, estimate 

.. math::
   \hat{h} = \arg\min_{h \in \mathcal{H}} \left[ \sup_{f \in \mathcal{F}} \left( 2 \cdot \text{loss}(f, \hat{g}, h) - \text{penalty}(f, \lambda) \right) + \text{penalty}(h, \mu) \right]

where :math:`\text{penalty}(f, \lambda) = \mathbb{E}_m\{f(C)^2\} + \lambda \cdot \|f\|^2_{\mathcal{F}}` and :math:`\text{penalty}(h, \mu) = \mu \cdot \|h\|^2_{\mathcal{H}}`.

**Sequential Nested NPIV: Ridge:**

Given observations :math:`(A_i, B_i, C_i)` in \tr, an initial estimator :math:`\hat{g}` which may be estimated in \tr, and a hyperparameter :math:`\mu`, estimate 

.. math::
   \hat{h} = \arg\min_{h \in \mathcal{H}} \left[ \sup_{f \in \mathcal{F}} \left( 2 \cdot \text{loss}(f, \hat{g}, h) - \text{penalty}(f) \right) + \text{penalty}(h, \mu) \right]

where :math:`\text{penalty}(f) = \mathbb{E}_m\{f(C)^2\}` and :math:`\text{penalty}(h, \mu) = \mu \cdot \mathbb{E}_m\{h(B)^2\}`.

**Simultaneous Nested NPIV:**

Given observations :math:`(A_i, B_i, C_i, C_i')` in \tr\, and hyperparameter values :math:`(\mu', \mu)`, estimate 

.. math::
   (\hat{g}, \hat{h}) = \arg\min_{g \in \mathcal{G}, h \in \mathcal{H}} \left[ \sup_{f' \in \mathcal{F}} \left( 2 \cdot \text{loss}(f', Y, g) - \text{penalty}(f') \right) + \text{penalty}(g, \mu') \right. \\
    \left. + \sup_{f \in \mathcal{F}} \left( 2 \cdot \text{loss}(f, g, h) - \text{penalty}(f) \right) + \text{penalty}(h, \mu) \right]

using analogous :math:`\text{penalty}` notation to the Sequential estimators.


.. toctree::
   :maxdepth: 2

   longitudinal/RKHS
   longitudinal/Random Forest
   longitudinal/Neural Network
   longitudinal/Sparse Linear
   longitudinal/Regularized Linear
   longitudinal/Linear



