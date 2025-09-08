from simulations.dgps_mediated import fn_dict, fn_dict_test 

import numpy as np
from nnpiv.rkhs import RKHS2IVL2

CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        'dgp_name': 'rkhs',
        'fn': [0,1,2,4],
        'n_samples': 2000
    },
    "methods": {
    'RKHS2IV' : [RKHS2IVL2(kernel='rbf', gamma=.05, delta_scale='auto', delta_exp=.4)
                 for _ in range(2)]
    },
    "method_opts": {
        'nn_1' : False,
        'nn_q1' : False,
        'CHIM' : False,
        'opts' : {'lin_degree': 1}
    },
    "estimator": 'joint',
    "mc_opts": {
        'n_experiments': 500,  # number of monte carlo experiments
        "seed": 123,
    }
}
