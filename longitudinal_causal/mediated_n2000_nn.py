# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ivfunctions_mediated_nn
from mcpy import metrics
from mcpy import plotting
from mliv.dgps_mediated import fn_dict, fn_dict_test 
from raw_plots import raw_metric, plot_raw, plot_raw_ind
import papertables_mediated


CONFIG = {
    "target_dir": "mediated_nn",
    "reload_results": True,
    "dgps": {
        "dgp1": ivfunctions_mediated_nn.gen_data
    },
    "dgp_opts": {
        'fn': list(iter(fn_dict.values())),
        'n_samples': 2000,
        'n_test': 1000
    },
    "methods": {
        "AGMM": ivfunctions_mediated_nn.agmm,
        "KLayerFixed": ivfunctions_mediated_nn.klayerfixed,
        "KLayerTrained": ivfunctions_mediated_nn.klayertrained,
        "CentroidMMD": ivfunctions_mediated_nn.centroidmmd,
        "KLossMMD": ivfunctions_mediated_nn.klossgmm      
    },
    "method_opts": {
        'n_epochs': 300,
        'model': 0,  # 0 is avg, 1 is final
        'burnin': 200
    },
    "metrics": {
        'rmse': ivfunctions_mediated_nn.mse,
        'raw': raw_metric
    },
    "plots": {
        'est': plot_raw_ind,
        'print_metrics': lambda x, y, z: papertables_mediated.paper_table(x, y, z,
                                                                 filename='mediated_sims_nn_dgp1.csv',
                                                                 nn=True)
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
