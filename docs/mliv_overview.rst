Longitudinal API Overview
=========================

This page lists public estimator APIs for sequential and simultaneous nested
NPIV estimation. Use the domain pages for mathematical development and this
page for direct class/function lookup.

RKHS Estimators
---------------

.. autosummary::
   :toctree: longitudinal/_autosummary
   :template: class.rst

   nnpiv.rkhs.RKHSIV
   nnpiv.rkhs.RKHSIVCV
   nnpiv.rkhs.ApproxRKHSIV
   nnpiv.rkhs.ApproxRKHSIVCV
   nnpiv.rkhs.RKHSIVL2
   nnpiv.rkhs.RKHSIVL2CV
   nnpiv.rkhs.ApproxRKHSIVL2
   nnpiv.rkhs.ApproxRKHSIVL2CV
   nnpiv.rkhs.RKHS2IV
   nnpiv.rkhs.RKHS2IVCV
   nnpiv.rkhs.RKHS2IVL2
   nnpiv.rkhs.RKHS2IVL2CV
   nnpiv.rkhs.ApproxRKHS2IV
   nnpiv.rkhs.ApproxRKHS2IVCV
   nnpiv.rkhs.ApproxRKHS2IVL2
   nnpiv.rkhs.ApproxRKHS2IVL2CV

Ensemble and Random Forest Estimators
-------------------------------------

.. autosummary::
   :toctree: longitudinal/_autosummary
   :template: class.rst

   nnpiv.ensemble.EnsembleIV
   nnpiv.ensemble.EnsembleIVStar
   nnpiv.ensemble.EnsembleIVL2
   nnpiv.ensemble.Ensemble2IV
   nnpiv.ensemble.Ensemble2IVL2

Neural Network Estimators
-------------------------

.. autosummary::
   :toctree: longitudinal/_autosummary
   :template: class.rst

   nnpiv.neuralnet.AGMM
   nnpiv.neuralnet.KernelLayerMMDGMM
   nnpiv.neuralnet.CentroidMMDGMM
   nnpiv.neuralnet.KernelLossAGMM
   nnpiv.neuralnet.MMDGMM
   nnpiv.neuralnet.AGMM2L2

Sparse and Regularized Linear Estimators
----------------------------------------

.. autosummary::
   :toctree: longitudinal/_autosummary
   :template: class.rst

   nnpiv.linear.sparse_l1vsl1
   nnpiv.linear.sparse_ridge_l1vsl1
   nnpiv.linear.sparse2_l1vsl1
   nnpiv.linear.sparse2_ridge_l1vsl1
   nnpiv.linear.sparse_l2vsl2
   nnpiv.linear.sparse_ridge_l2vsl2
   nnpiv.linear.sparse2_l2vsl2
   nnpiv.linear.sparse2_ridge_l2vsl2

TSLS Baselines
--------------

.. autosummary::
   :toctree: longitudinal/_autosummary
   :template: function.rst

   nnpiv.tsls.tsls
   nnpiv.tsls.regtsls
