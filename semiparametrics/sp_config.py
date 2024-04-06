# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from mliv.dgps_mediated import fn_dict, fn_dict_test 

import numpy as np
from mliv.rkhs import ApproxRKHSIVCV
from mliv.tsls import tsls, regtsls, exptsls
from mliv.linear import OptimisticHedgeVsOptimisticHedge


CONFIG = {
    "target_dir": "sp",
    "reload_results": True,
    "dgp_opts": {
        'dgp_name': 'kernel',
        'fn': list(iter(fn_dict.values())),
        'n_samples': 2000
    },
    "methods": {
    '2SLS' : [tsls(), tsls(), tsls(), tsls()] ,
    '2SLS_exp' : [tsls(), tsls(), exptsls(), tsls()],
    'Reg2SLS' : [regtsls(), regtsls(), regtsls(), regtsls()],
    'RKHS' : [ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5) for _ in range(4)],
    'SpLin' : [OptimisticHedgeVsOptimisticHedge(B=3, lambda_theta=.05,
                                             eta_theta=.1,
                                             eta_w=.1,
                                             n_iter=10000, tol=.0001, sparsity=None) for _ in range(4)]                  
    },
    "method_opts": {
        'nn_1' : False,
        'nn_2' : False,
        'nn_q1' : False,
        'nn_q2' : False,
        'CHIM' : False,
        'opts' : None
    },
    "mc_opts": {
        'n_experiments': 100,  # number of monte carlo experiments
        "seed": 123,
    }
}
