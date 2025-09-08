from mcpy.utils import filesafe
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def raw_metric(param_estimates, true_params):
    return np.hstack([param_estimates, true_params])


def plot_ind(param_estimates, metric_results, config):
    for dgp_name, mdgp in metric_results.items():
        metric_name = 'raw'
        for it, method_name in enumerate(mdgp.keys()):
            if config['dgp_opts']['fn'] in config['subplots']['fn_plot'] :
                plt.figure()
                true_params = np.array(mdgp[method_name][metric_name][0, :, 1])
                plt.plot(np.arange(true_params.shape[0]),
                            true_params, '--', label='true')

                res = np.array(mdgp[method_name][metric_name][:, :, 0])
                med_res = np.median(res, axis=0)
                lb_res = np.percentile(res, 5, axis=0)
                ub_res = np.percentile(res, 95, axis=0)
                line = plt.plot(
                    np.arange(med_res.shape[0]), med_res, label=method_name)
                plt.fill_between(
                    np.arange(med_res.shape[0]), lb_res, ub_res, alpha=.4, color=line[0].get_color())
                plt.legend()
                plt.tight_layout()
                plots_dir = os.path.join(config['target_dir'], 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                
                # Generate the filename
                filename = '{}_dgp_{}_{}_{}.png'.format(
                    filesafe('plot'), dgp_name, method_name, config['dgp_opts']['fn'])
                filepath = os.path.join(plots_dir, filename)

                plt.savefig(filepath, dpi=300)
                plt.close()
    return



