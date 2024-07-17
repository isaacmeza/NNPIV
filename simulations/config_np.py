# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions_nested
from simulations.dgps_nested import fn_dict 
from simulations.printplots import raw_metric, plot_ind
import printtable


CONFIG = {
    "target_dir": "nested_md_2",
    "reload_results": True,
    "dgps": {
        "dgp2": ivfunctions_nested.gen_data
    },
    "dgp_opts": {
        'dgp_num': 2,
        'fn': [0,1],
        'n_samples': 1000,
        'n_a': 1,
        'n_b': 1,
        'n_test': 1000,
        'gridtest': 1
    },
    "methods": {
        "2SLS": ivfunctions_nested.tsls,
        "Reg2SLS": ivfunctions_nested.regtsls    
    },
    "method_opts": {
        'nstrm_n_comp': 100,
        'lin_degree': 3
    },
    "metrics": {
        'rmse': ivfunctions_nested.mse,
        'raw': raw_metric
    },
    "plots": {
        'est': plot_ind,
        'print_metrics': lambda x, y, z: printtable.print_table(x, y, z,
                                                                 filename='nested_md_2/nested_sims_md_dgp2.csv',
                                                                 nn=False)
    },
    "subplots": {
        'fn_plot': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    },
    "sweep_plots": {
    },
    "mc_opts": {
        'n_experiments': 2,  # number of monte carlo experiments
        "seed": 123,
    },
    "cluster_opts": {
        "node_id": __NODEID__,
        "n_nodes": __NNODES__
    },
}
