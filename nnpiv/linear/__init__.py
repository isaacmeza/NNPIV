# Licensed under the MIT License.

"""
Public linear/sparse estimator API for nested NPIV.

This module re-exports optimization-based sparse linear estimators and their
ridge/L1/L2 variants used in sequential and simultaneous formulations.
"""

from .sparse_linear import OptimisticHedgeVsOptimisticHedge,\
    StochasticOptimisticHedgeVsOptimisticHedge,\
    ProxGradientVsHedge,\
    SubGradientVsHedge,\
    TSLasso,\
    L2SubGradient, L2ProxGradient, L2OptimisticHedgeVsOGD
from .sparse_l1_l1 import sparse_l1vsl1, sparse_ridge_l1vsl1
from .sparse2_l1_l1 import sparse2_l1vsl1, sparse2_ridge_l1vsl1
from . sparse_l2_l2 import sparse_l2vsl2, sparse_ridge_l2vsl2
from .sparse2_l2_l2 import sparse2_l2vsl2, sparse2_ridge_l2vsl2

__all__ = ['OptimisticHedgeVsOptimisticHedge',
           'StochasticOptimisticHedgeVsOptimisticHedge',
           'ProxGradientVsHedge',
           'SubGradientVsHedge',
           'TSLasso',
           'L2SubGradient', 'L2ProxGradient', 'L2OptimisticHedgeVsOGD',
           'sparse_l1vsl1', 'sparse_ridge_l1vsl1',
           'sparse2_l1vsl1', 'sparse2_ridge_l1vsl1',
           'sparse_l2vsl2', 'sparse_ridge_l2vsl2',
           'sparse2_l2vsl2', 'sparse2_ridge_l2vsl2']
