# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .agmm_earlystop import AGMMEarlyStop, KernelLayerMMDGMMEarlyStop,\
    CentroidMMDGMMEarlyStop, KernelLossAGMMEarlyStop
from .agmm import AGMM, KernelLayerMMDGMM, CentroidMMDGMM, KernelLossAGMM, MMDGMM
from .agmm2 import AGMM2, AGMM2L2

from .architectures import CNN_Z_agmm, CNN_Z_kernel, CNN_X, CNN_X_bn, fc_z_agmm, fc_z_kernel, fc_x


__all__ = ['AGMMEarlyStop',
           'KernelLayerMMDGMMEarlyStop',
           'CentroidMMDGMMEarlyStop',
           'KernelLossAGMMEarlyStop',
           'AGMM',
           'AGMM2',
           'AGMM2L2',
           'KernelLayerMMDGMM',
           'CentroidMMDGMM',
           'KernelLossAGMM',
           'MMDGMM',
           'CNN_Z_agmm',
           'CNN_Z_kernel',
           'CNN_X',
           'CNN_X_bn',
           'fc_x',
           'fc_z_agmm',
           'fc_z_kernel']
