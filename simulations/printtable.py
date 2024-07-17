# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import simulations.dgps_nested as dgps
import ivfunctions_nested


def print_table(param_estimates, metric_results, config, filename="", nn=False):
    out = open(filename, 'a')
    if nn:
        method_list = ["AGMM", "KLayerFixed",
                       "KLayerTrained", "CentroidMMD", "KLossMMD"]
    else:
        method_list = ['2SLS', 'Reg2SLS',
                        'NystromRKHS',
                        'SpLinL1', 'StSpLinL1',
                        'ConvexIV', 'TVIV', 'LipTVIV',
                        'RFIV']
    for dgp_name, mdgp in metric_results.items():
        print(
            ivfunctions_nested._key(dgps.fn_dict, config['dgp_opts']['fn']).replace("_", ""), end=" ", file=out)
        for metric_name in ['rmse']:
            min_metric = np.inf
            for method_name in mdgp.keys():
                if method_name in method_list:
                    res = mdgp[method_name][metric_name]
                    mean_res = res.mean()
                    if mean_res <= min_metric:
                        best = method_name
                        min_metric = mean_res
            for method_name in mdgp.keys():
                if method_name in method_list:
                    res = mdgp[method_name][metric_name]
                    mean_res = res.mean()
                    std_res = res.std() / np.sqrt(len(res))
                    if method_name == best:
                        print(r"& {{\bf {:.3f} $\pm$ {:.3f} }}".format(
                            mean_res, 2 * std_res), end=" ", file=out)
                    else:
                        print(r"& {:.3f} $\pm$ {:.3f}".format(
                            mean_res, 2 * std_res), end=" ", file=out)
            print("\\\\", file=out)
    out.close()
    return
