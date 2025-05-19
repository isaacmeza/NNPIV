# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions_nested
import ivfunctions_nested_nn
from simulations.dgps_nested import fn_dict_paper 
from simulations.printplots import raw_metric, plot_ind
import printtable


CONFIG = {
    "target_dir": "nonparametric_fit",
    "reload_results": True,
    "dgps": {
        "dgp2": ivfunctions_nested.gen_data
    },
    "dgp_opts": {
        'dgp_num': 2,
        'fn': list(iter(fn_dict_paper.values())),
        'n_samples': 2000,
        'n_a': 10,
        'n_b': 10,
        'n_test': 1000,
        'gridtest': 1
    },
    "methods": { 
        "RKHS2L2": ivfunctions_nested.rkhs2_ridge
    },
    "method_opts": {
        'lin_degree': 3,
        'lin_l1': 0.1,
        'lin_nit': 10000,
        'budget': 10,
        'rf_iter': 250
    },
    "metrics": {
        'rmse': ivfunctions_nested.mse,
        'raw': raw_metric
    },
    "plots": {
        'est': plot_ind,
        'print_metrics': lambda x, y, z: printtable.print_table(x, y, z,
                                                                 filename='nonparametric_fit/table_1.csv')
    },
    "subplots": {
        'fn_plot': [8,15,2,7,16]
    },
    "sweep_plots": {
    },
    "mc_opts": {
        'n_experiments': 100,  # number of monte carlo experiments
        "seed": 123,
    },
    "cluster_opts": {
        "node_id": __NODEID__,
        "n_nodes": __NNODES__
    },
}
