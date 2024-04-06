# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions_nested_nn
from mcpy import metrics
from mcpy import plotting
from mliv.dgps_nested import fn_dict 
from raw_plots import raw_metric, plot_raw, plot_raw_ind
import papertables_nested


CONFIG = {
    "target_dir": "nested_md_2_nn",
    "reload_results": True,
    "dgps": {
        "dgp2": ivfunctions_nested_nn.gen_data
    },
    "dgp_opts": {
        'dgp_num': 2,
        'fn': list(iter(fn_dict.values())),
        'n_samples': 2000,
        'n_a': 10,
        'n_b': 10,
        'n_test': 1000,
        'gridtest': 1
    },
    "methods": {
        "AGMM": ivfunctions_nested_nn.agmm    
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
        'est': plot_raw_ind,
        'print_metrics': lambda x, y, z: papertables_nested.paper_table(x, y, z,
                                                                 filename='nested_md_nn_dgp2.csv',
                                                                 nn=True)
    },
    "subplots": {
        'fn_plot': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
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
