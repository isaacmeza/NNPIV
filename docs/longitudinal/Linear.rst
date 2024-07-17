Linear Class
============

If we consider the space of linear functions, then the sequential and joint estimators coincide. Without regularization the joint estimator takes the form:

.. math::
   \min _{\alpha ,\beta } \max _{\theta_1,\theta_2} \ell(\{\alpha, \beta\}, \{\theta_1,\theta_2\})

where

.. math::
   \ell(\{\alpha, \beta\}, \{\theta_1,\theta_2\}) := 2\theta_1^{\top} \mathbb{E}_n[c' y]-2\theta_1^{\top} \mathbb{E}_n\left[c' a^{\top}\right] \alpha + 2\theta_2^{\top} \mathbb{E}_n[c a^{\top}]\alpha-2\theta_2^{\top} \mathbb{E}_n\left[c b^{\top}\right] \beta

Note that the saddle point is given by the system:

.. math::
   \begin{aligned}
   \mathbb{E}_n[(y-\langle \alpha, a\rangle)c'] &= 0 \\
   \mathbb{E}_n[(\langle \alpha, a\rangle-\langle\beta, b\rangle)c] &= 0
   \end{aligned}

Solving first for :math:`\alpha` in the first equation, and then for :math:`\beta` in the second equation gives the same solution as in the sequential procedure.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   tsls.tsls
   tsls.regtsls
