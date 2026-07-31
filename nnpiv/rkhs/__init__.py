# Licensed under the MIT License.

"""
Public RKHS estimator API for nested NPIV.

This module re-exports sequential and simultaneous RKHS estimators, including
cross-validated and low-rank approximation variants.
"""

from .rkhsiv import RKHSIV, RKHSIVCV, ApproxRKHSIV, ApproxRKHSIVCV, \
    RKHSIVL2, RKHSIVL2CV, ApproxRKHSIVL2, ApproxRKHSIVL2CV
from .rkhs2iv import RKHS2IV, RKHS2IVCV, RKHS2IVL2, RKHS2IVL2CV, \
    ApproxRKHS2IV, ApproxRKHS2IVCV, ApproxRKHS2IVL2, ApproxRKHS2IVL2CV
from .rkhsiv_legacy import RKHSIV_legacy, RKHSIVCV_legacy, \
    ApproxRKHSIV_legacy, ApproxRKHSIVCV_legacy, RKHSIVL2_legacy, \
    RKHSIVL2CV_legacy, ApproxRKHSIVL2_legacy, ApproxRKHSIVL2CV_legacy
from .rkhs2iv_legacy import RKHS2IV_legacy, RKHS2IVCV_legacy, \
    RKHS2IVL2_legacy, RKHS2IVL2CV_legacy, ApproxRKHS2IV_legacy, \
    ApproxRKHS2IVCV_legacy, ApproxRKHS2IVL2_legacy, \
    ApproxRKHS2IVL2CV_legacy

__all__ = ['RKHSIV',
           'RKHSIVCV',
           'ApproxRKHSIV',
           'ApproxRKHSIVCV',
           'RKHSIVL2',
           'RKHSIVL2CV',
           'ApproxRKHSIVL2',
           'ApproxRKHSIVL2CV',
           'RKHS2IV',
           'RKHS2IVCV',
           'RKHS2IVL2',
           'RKHS2IVL2CV',
           'ApproxRKHS2IV',
           'ApproxRKHS2IVCV',
           'ApproxRKHS2IVL2',
           'ApproxRKHS2IVL2CV',
           'RKHSIV_legacy',
           'RKHSIVCV_legacy',
           'ApproxRKHSIV_legacy',
           'ApproxRKHSIVCV_legacy',
           'RKHSIVL2_legacy',
           'RKHSIVL2CV_legacy',
           'ApproxRKHSIVL2_legacy',
           'ApproxRKHSIVL2CV_legacy',
           'RKHS2IV_legacy',
           'RKHS2IVCV_legacy',
           'RKHS2IVL2_legacy',
           'RKHS2IVL2CV_legacy',
           'ApproxRKHS2IV_legacy',
           'ApproxRKHS2IVCV_legacy',
           'ApproxRKHS2IVL2_legacy',
           'ApproxRKHS2IVL2CV_legacy']
