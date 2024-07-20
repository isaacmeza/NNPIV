# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from simulations.dgps_mediated import fn_dict, fn_dict_test 

import numpy as np
from nnpiv.linear import sparse2_l1vsl1

CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        'dgp_name': 'sparse',
        'fn': [0,1],
        'n_samples': 2000
    },
    "methods": {
    'Sparse2_2' : [sparse2_l1vsl1(mu=0.05, V1=3, V2=3,
                 eta_alpha=0.1, eta_w1=0.1, eta_beta=0.1, eta_w2=0.1,
                 n_iter=20000, tol=.00001/2, sparsity=None, fit_intercept=True)
                 for _ in range(2)]
    },
    "method_opts": {
        'nn_1' : False,
        'nn_q1' : False,
        'CHIM' : False,
        'opts' : {'lin_degree': 3}
    },
    "estimator": 'joint',
    "mc_opts": {
        'n_experiments': 5,  # number of monte carlo experiments
        "seed": 123,
    }
}
