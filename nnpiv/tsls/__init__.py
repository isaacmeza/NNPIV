# Licensed under the MIT License.

"""
Public TSLS baseline API for nested NPIV comparisons.

This module re-exports standard and regularized two-stage least squares
estimators used as linear benchmarks.
"""

from .tsls import regtsls, tsls


__all__ = ['regtsls',
           'tsls']
