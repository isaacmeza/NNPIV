# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from simulations.dgps_mediated import fn_dict, fn_dict_test 

import numpy as np
from nnpiv.rkhs import ApproxRKHSIVCV
from nnpiv.tsls import tsls, regtsls
from nnpiv.linear import OptimisticHedgeVsOptimisticHedge

from nnpiv.linear import sparse2_l1vsl1
from nnpiv.rkhs import RKHS2IVL2CV
from nnpiv.ensemble import Ensemble2IV, EnsembleIV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

adversary = RandomForestRegressor(n_estimators=50, max_depth=2,
                                    bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001)
learnerg = RandomForestClassifier(n_estimators=10, max_depth=2, criterion='gini',
                                        bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001)
learnerh = RandomForestClassifier(n_estimators=10, max_depth=2, criterion='gini',
                                        bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001)


CONFIG = {
    "target_dir": "semiparametric_cov",
    "reload_results": True,
    "dgp_opts": {
        'dgp_name': 'rf_qseq',
        'fn': [0,1,2,3,4],
        'n_samples': 2000
    },
    "methods": {  
    'RFIV2' : [Ensemble2IV(n_iter=500, max_abs_value=2, 
                              adversary=adversary, learnerg=learnerg, learnerh=learnerh, n_burn_in=10),
                [EnsembleIV(n_iter=300, max_abs_value=2, 
                              adversary=adversary, learner=learnerg), EnsembleIV(n_iter=300, max_abs_value=2, 
                              adversary=adversary, learner=learnerg)]]
    },
    "method_opts": {
        'nn_1' : False,
        'nn_q1' : [False, False],
        'CHIM' : False,
        'opts' : {'lin_degree': 1}
    },
    "estimator": 'joint',
    "mc_opts": {
        'n_experiments': 500,  # number of monte carlo experiments
        "seed": 123,
    }
}
