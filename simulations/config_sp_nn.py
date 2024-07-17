# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from simulations.dgps_mediated import fn_dict, fn_dict_test 

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from nnpiv.neuralnet.rbflayer import gaussian, inverse_multiquadric
from nnpiv.neuralnet import AGMM, KernelLayerMMDGMM, CentroidMMDGMM, KernelLossAGMM, MMDGMM

p = 0.1  # dropout prob of dropout layers throughout notebook
n_hidden = 100  # width of hidden layers throughout notebook

# For any method that use a projection of z into features g(z)
g_features = 100

# The kernel function
kernel_fn = gaussian
# kernel_fn = inverse_multiquadric

# Training params
learner_lr = 1e-4
adversary_lr = 1e-4
learner_l2 = 1e-3
adversary_l2 = 1e-4
adversary_norm_reg = 1e-3
n_epochs = 300
bs = 100
sigma = 2.0 / g_features
n_centers = 100
device = torch.cuda.current_device() if torch.cuda.is_available() else None
print("GPU:", torch.cuda.is_available())

def _get_learner(n_t):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, 1))

def _get_adversary(n_z):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, 1))

def _get_adversary_g(n_z):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, g_features), nn.ReLU())


agmm_1 = AGMM(_get_learner(4),_get_adversary(4))
agmm_2 = AGMM(_get_learner(3),_get_adversary(3))

klayerfixed_1 = KernelLayerMMDGMM(_get_learner(4), lambda x: x, 4, n_centers, kernel_fn, centers = np.tile(
        np.linspace(-4, 4, n_centers).reshape(-1, 1), (1, 4)), sigmas = np.ones((n_centers,)) * 2 / 4, trainable=False)
klayerfixed_2 = KernelLayerMMDGMM(_get_learner(3), lambda x: x, 3, n_centers, kernel_fn, centers = np.tile(
        np.linspace(-4, 4, n_centers).reshape(-1, 1), (1, 3)), sigmas = np.ones((n_centers,)) * 2 / 3, trainable=False)

klayertrained_1 = KernelLayerMMDGMM(_get_learner(4), _get_adversary_g(4), g_features,
                                     n_centers, kernel_fn, centers=np.random.uniform(-4, 4, size=(n_centers, g_features)), sigmas=np.ones((n_centers,)) * sigma)
klayertrained_2 = KernelLayerMMDGMM(_get_learner(3), _get_adversary_g(3), g_features,
                                     n_centers, kernel_fn, centers=np.random.uniform(-4, 4, size=(n_centers, g_features)), sigmas=np.ones((n_centers,)) * sigma)

centroidmmd_1 = CentroidMMDGMM(_get_learner(4), _get_adversary_g(4), kernel_fn, np.tile(np.linspace(-4, 4, n_centers).reshape(-1, 1), (1, 4)) ,np.ones(n_centers) * sigma)
centroidmmd_2 = CentroidMMDGMM(_get_learner(3), _get_adversary_g(3), kernel_fn, np.tile(np.linspace(-4, 4, n_centers).reshape(-1, 1), (1, 3)) ,np.ones(n_centers) * sigma)

klossgmm_1 = KernelLossAGMM(_get_learner(4), _get_adversary_g(4), kernel_fn, sigma)
klossgmm_2 = KernelLossAGMM(_get_learner(3), _get_adversary_g(3), kernel_fn, sigma)


CONFIG = {
    "target_dir": "sp_nn",
    "reload_results": True,
    "dgp_opts": {
        'dgp_name': 'nn',
        'fn': list(iter(fn_dict.values())),
        'n_samples': 2000
    },
    "methods": {
    'AGMM' : [agmm_1, agmm_2, agmm_2, agmm_1],
    'KLF' : [klayerfixed_1, klayerfixed_2, klayerfixed_2, klayerfixed_1],
    'KLT' : [klayertrained_1, klayertrained_2, klayertrained_2, klayertrained_1],
    'cMMD' : [centroidmmd_1, centroidmmd_2, centroidmmd_2, centroidmmd_1],
    'klMMD' : [klossgmm_1, klossgmm_2, klossgmm_2, klossgmm_1]           
    },
    "method_opts": {
        'nn_1' : True,
        'nn_2' : True,
        'nn_q1' : True,
        'nn_q2' : True,
        'CHIM' : False,
        'fitargs' : {'n_epochs': 300, 'bs': 100, 'learner_lr': 1e-4, 'adversary_lr': 1e-4, 
                      'learner_l2': 1e-3, 'adversary_l2': 1e-4, 'model_dir' : str(Path.home()), 'device' : device },
        'opts' : {'burnin': 200}
    },
    "mc_opts": {
        'n_experiments': 100,  # number of monte carlo experiments
        "seed": 123,
    }
}
