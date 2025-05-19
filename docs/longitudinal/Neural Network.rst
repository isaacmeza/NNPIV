.. _neural-networks:

Neural Networks
===============

We now consider the case where the function classes correspond to neural networks. In such case, the (joint) estimator takes the form:

.. math::

    (\hat{g}, \hat{h}) = \arg \min _{\theta_1, \theta_2} 
    \max_{\omega_1, \omega_2} \left\{ 
    \mathbb{E}_n\left[2\left\{g_{\theta_1}(A) - Y\right\} f_{\omega_1}'(C') - f_{\omega_1}'(C')^2\right] 
    + \mu' \mathbb{E}_n\{g_{\theta_1}(A)^2\} \right. \\
    \left. + \mathbb{E}_n\left[2\left\{h_{\theta_2}(B) - g_{\theta_1}(A)\right\} f_{\omega_2}(C) - f_{\omega_2}(C)^2\right] 
    + \mu \mathbb{E}_n\{h_{\theta_2}(B)^2\} 
    \right\}


where :math:`\theta_1, \theta_2, \omega_1, \omega_2` are weights of the neural networks.

We use the Optimistic Adam algorithm of `Daskalakis et al. (2017) <http://arxiv.org/abs/1711.00141>`_ to solve the previous minimax problem as was also proposed in `Dikkala et al. (2020) <https://arxiv.org/abs/2006.07201>`_.

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   oadam.OAdam

Subsetted Estimator
-------------------

Modify the computation of the loss for the adversary to be zero for the observations outside the restriction:

.. code-block:: python

    test = self.adversary(zb)
    test[indices_] = 0 
    G_loss = - torch.mean((yb - pred) * test) + torch.mean(test**2)


Single estimator
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   agmm.AGMM
   agmm.KernelLayerMMDGMM
   agmm.CentroidMMDGMM
   agmm.KernelLossAGMM
   agmm.MMDGMM


Joint estimator
^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   .. autosummary::
   :toctree: _autosummary
   :template: class.rst

   agmm2.AGMM2L2
