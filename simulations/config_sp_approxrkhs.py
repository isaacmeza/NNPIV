from simulations.dgps_mediated import fn_dict, fn_dict_test 

import numpy as np
from nnpiv.rkhs import ApproxRKHSIVCV

model = ApproxRKHSIVCV(
    kernel_approx="nystrom", n_components=200,
    kernel="rbf", gamma=0.1, delta_scale="auto",
    delta_exp=0.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5
)

CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        'dgp_name': 'approxrkhs',
        'fn': [0,1,2,4],
        'n_samples': 2000
    },
    "methods": {
    'ApproxRKHS' : [[model, model], [model, model]]
    },
    "method_opts": {
        'nn_1' : [False, False],
        'nn_q1' : [False, False],
        'fitargs' : [None, None],
        'fitargsq1' : [None, None],
        'opts' : {'lin_degree': 1}
    },
    "estimator": 'joint',
    "mc_opts": {
        'n_experiments': 500,  # number of monte carlo experiments
        "seed": 123,
    }
}
