# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from simulations.dgps_mediated import fn_dict, fn_dict_test 

import numpy as np
from nnpiv.rkhs import ApproxRKHSIVCV
from nnpiv.tsls import tsls, regtsls


CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        'dgp_name': 'benchmark',
        'fn': [0,1,2,3,4],
        'n_samples': 2000
    },
    "methods": {
    '2SLS' : [tsls(), tsls(), tsls(), tsls()] ,
    'Reg2SLS' : [regtsls(), regtsls(), regtsls(), regtsls()]
    },
    "method_opts": {
        'nn_1' : False,
        'nn_2' : False,
        'nn_q1' : False,
        'nn_q2' : False,
        'CHIM' : False,
        'opts' : {'lin_degree': 3}
    },
    "estimator": 'sequential',
    "mc_opts": {
        'n_experiments': 500,  # number of monte carlo experiments
        "seed": 123,
    }
}
