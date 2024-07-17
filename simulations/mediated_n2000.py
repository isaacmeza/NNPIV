# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions_mediated
from mcpy import metrics
from mcpy import plotting
from mliv.dgps_mediated import fn_dict, fn_dict_test 
from raw_plots import raw_metric, plot_raw, plot_raw_ind
import papertables_mediated


CONFIG = {
    "target_dir": "mediated",
    "reload_results": True,
    "dgps": {
        "dgp1": ivfunctions_mediated.gen_data
    },
    "dgp_opts": {
        'fn': list(iter(fn_dict.values())),
        'n_samples': 2000,
        'n_test': 1000
    },
    "methods": {
        "2SLS": ivfunctions_mediated.tsls,
        "Reg2SLS": ivfunctions_mediated.regtsls,
        "NystromRKHS": ivfunctions_mediated.nystromrkhsfit,        
        "RFIV": ivfunctions_mediated.ensembleiv 
    },
    "method_opts": {
        'nstrm_n_comp': 100,
        'shiv_L': 2,
        'shiv_mon': None,
        'lin_degree': 3
    },
    "metrics": {
        'rmse': ivfunctions_mediated.mse,
        'raw': raw_metric
    },
    "plots": {
        'est': plot_raw_ind,
        'print_metrics': lambda x, y, z: papertables_mediated.paper_table(x, y, z,
                                                                 filename='mediated_sims_dgp1.csv',
                                                                 nn=False)
    },
    "subplots": {
        'fn_plot': [0,1,2,3,4]
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
