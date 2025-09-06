# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions_nested_nn
from simulations.dgps_nested import fn_dict_paper 
from simulations.printplots import raw_metric, plot_ind
import printtable


CONFIG = {
    "target_dir": "nonparametric_fit",
    "reload_results": True,
    "dgps": {
        "dgp2": ivfunctions_nested_nn.gen_data
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
        "AGMM2L2": ivfunctions_nested_nn.agmm2l2    
    },
    "method_opts": {
        'n_epochs': 300,
        'model': 0,  # 0 is avg, 1 is final
        'burnin': 200
    },
    "metrics": {
        'rmse': ivfunctions_nested_nn.mse,
        'raw': raw_metric
    },
    "plots": {
        'est': plot_ind,
        'print_metrics': lambda x, y, z: printtable.print_table(x, y, z,
                                                                 filename='nonparametric_fit/table1_nn.csv')
    },
    "subplots": {
        'fn_plot': [8,15,2,16]
    },
    "sweep_plots": {
    },
    "mc_opts": {
        'n_experiments': 500,  # number of monte carlo experiments
        "seed": 123,
    },
    "cluster_opts": {
        "node_id": __NODEID__,
        "n_nodes": __NNODES__
    },
}
