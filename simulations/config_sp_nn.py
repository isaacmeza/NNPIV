# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from simulations.dgps_mediated import fn_dict, fn_dict_test 

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from nnpiv.neuralnet.rbflayer import gaussian, inverse_multiquadric
from nnpiv.neuralnet import AGMM, AGMM2

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


agmm_1 = AGMM2(learnerh = _get_learner(3), learnerg = _get_learner(4),
                     adversary1 = _get_adversary(4), adversary2 = _get_adversary(3))

agmm_q1 = AGMM2(learnerh = _get_learner(4), learnerg = _get_learner(3),
                     adversary1 = _get_adversary(3), adversary2 = _get_adversary(4))



CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        'dgp_name': 'nn',
        'fn': [0,1],
        'n_samples': 2000
    },
    "methods": {
    'AGMM2' : [agmm_1, agmm_q1]      
    },
    "method_opts": {
        'nn_1' : True,
        'nn_q1' : True,
        'CHIM' : False,
        'fitargs' : {'n_epochs': 600, 'bs': 100, 'learner_lr': 1e-4, 'adversary_lr': 1e-4, 
                      'learner_l2': 1e-3, 'adversary_l2': 1e-4, 'model_dir' : str(Path.home()), 'device' : device },
        'opts' : {'burnin': 400}
    },
    "estimator": 'joint',
    "mc_opts": {
        'n_experiments': 5,  # number of monte carlo experiments
        "seed": 123,
    }
}
