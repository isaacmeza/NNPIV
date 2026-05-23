import sys
import os
from numbers import Integral

# Add the simulations/mcpy directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../simulations')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from joblib import Parallel, delayed
import joblib
import argparse
import importlib
from itertools import product
import collections
from copy import deepcopy
from mcpy.utils import filesafe
import simulations.dgps_mediated as dgps
from nnpiv.semiparametrics import DML_mediated
import time

def _get(opts, key, default):
    return opts[key] if (key in opts) else default


def _require_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
    if int(value) < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
    return int(value)


def _resolve_inner_n_jobs(config, outer_effective_n_jobs, n_folds):
    method_opts = config.get("method_opts", {})
    explicit_inner_n_jobs = method_opts.get("inner_n_jobs")
    if explicit_inner_n_jobs is not None:
        inner_n_jobs = _require_positive_int(explicit_inner_n_jobs, "method_opts.inner_n_jobs")
        return min(inner_n_jobs, n_folds)

    if outer_effective_n_jobs > 1:
        return 1

    available_cores = max(1, int(joblib.cpu_count()))
    return min(n_folds, available_cores)


def _finite_1d(values):
    """Return finite numeric values as a flat float array."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _percentile_slice(values, lower_q=None, upper_q=None):
    """Filter values by percentile bounds, falling back to finite values if empty."""
    arr = _finite_1d(values)
    if arr.size == 0:
        return arr

    mask = np.ones(arr.shape, dtype=bool)
    if lower_q is not None:
        lower = np.percentile(arr, lower_q)
        mask &= arr >= lower
    if upper_q is not None:
        upper = np.percentile(arr, upper_q)
        mask &= arr <= upper

    filtered = arr[mask]
    return filtered if filtered.size > 0 else arr


def _safe_mean_std_median(values):
    arr = _finite_1d(values)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr)), float(np.median(arr))


def _warn_if_nonfinite_method_outputs(results, fn_number, model_name):
    """Emit a warning when a method run produced only non-finite estimates."""
    if not results:
        print(
            f"[WARN] Empty results for method={model_name}, fn={fn_number}. "
            "No experiments were returned."
        )
        return

    theta_vals = _finite_1d([run[0] for run in results])
    variance_vals = _finite_1d([run[1] for run in results])

    ci_finite = 0
    for run in results:
        try:
            ci = run[2]
            lo, hi = ci
            if np.isfinite(lo) and np.isfinite(hi):
                ci_finite += 1
        except Exception:
            continue

    n_runs = len(results)
    if theta_vals.size == 0 and variance_vals.size == 0 and ci_finite == 0:
        print(
            f"[WARN] All estimates are non-finite for method={model_name}, fn={fn_number} "
            f"(theta 0/{n_runs}, variance 0/{n_runs}, CI 0/{n_runs}). "
            "Results CSV will contain NaNs for summary statistics."
        )
    elif theta_vals.size == 0 or variance_vals.size == 0 or ci_finite == 0:
        print(
            f"[WARN] Partially non-finite outputs for method={model_name}, fn={fn_number} "
            f"(theta {theta_vals.size}/{n_runs}, variance {variance_vals.size}/{n_runs}, CI {ci_finite}/{n_runs})."
        )


def _check_valid_config(config):
    assert 'dgp_opts' in config, "config dict must contain dgp_opts"
    assert 'method_opts' in config, "config dict must contain method_opts"
    assert 'mc_opts' in config, "config dict must contain mc_opts"
    assert 'methods' in config, "config dict must contain methods"
    assert 'target_dir' in config, "config must contain target_dir"
    assert 'reload_results' in config, "config must contain reload_results"
    assert 'n_experiments' in config['mc_opts'], "config[mc_opts] must contain n_experiments"
    assert 'seed' in config['mc_opts'], "config[mc_opts] must contain seed"


class SemiParametricsMonteCarlo:

    def __init__(self, config):
        self.config = config
        _check_valid_config(self.config)
        config['param_str'] = '_'.join(
            ['{}_{}'.format(filesafe(k), v) for k, v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(
            ['{}_{}'.format(filesafe(k), v) for k, v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join([str(k) for k, _ in self.config['methods'].items()])
        return

    def experiment(self, exp_id, fn_number, model_instance, n_folds, inner_n_jobs):
        ''' Runs an experiment on a single randomly generated instance and sample and returns
        the parameter estimates for each method 
        '''
        np.random.seed(exp_id)
        
        tau_fn = dgps.get_tau_fn(fn_number)
        W, Z, X, M, D, Y, tau_fn = dgps.get_data(_get(self.config['dgp_opts'], 'n_samples', 2000), tau_fn)

        dml_model = DML_mediated(Y=Y, D=D, M=M, W=W, Z=Z, X1=X,
                                estimator='MR',
                                estimand='E[Y(1,M(0))]',
                                model1 = model_instance[0],
                                modelq1 = model_instance[1],
                                n_folds=n_folds, n_rep=1, verbose=False,
                                inner_n_jobs=inner_n_jobs,
                                CHIM = _get(self.config['method_opts'], 'CHIM', False),
                                nn_1 = _get(self.config['method_opts'], 'nn_1', False),
                                nn_q1 = _get(self.config['method_opts'], 'nn_q1', False),
                                fitargs1 = _get(self.config['method_opts'], 'fitargs', None),
                                fitargsq1 = _get(self.config['method_opts'], 'fitargs', None),
                                opts = _get(self.config['method_opts'], 'opts', None))
                  
        start_time = time.time()
        theta, var, ci = dml_model.dml()
        end_time = time.time()
        elapsed_time = end_time - start_time

        return (theta, var, ci, elapsed_time)

    def run(self):
        ''' Runs multiple experiments in parallel on randomly generated instances and samples and returns
        the parameter estimates for each method across all experiments
        '''
        random_seed = self.config['mc_opts']['seed']
        n_folds = 5
        mc_n_jobs = _get(self.config['mc_opts'], 'n_jobs', -1)
        if mc_n_jobs is None:
            mc_n_jobs = -1
        try:
            outer_effective_n_jobs = joblib.effective_n_jobs(mc_n_jobs)
        except Exception as exc:
            raise ValueError(
                f"mc_opts.n_jobs must be an integer compatible with joblib, got {mc_n_jobs!r}."
            ) from exc
        inner_n_jobs = _resolve_inner_n_jobs(self.config, outer_effective_n_jobs, n_folds)

        if not os.path.exists(self.config['target_dir']):
            os.makedirs(self.config['target_dir'])

        result_distributions = np.empty((4, len(self.config['dgp_opts']['fn']), 
            len(self.config['methods']), self.config['mc_opts']['n_experiments']), dtype=object)
            
        ii = 0
        for fn_number in self.config['dgp_opts']['fn']:
            j = 0
            for model_name, model_instance in self.config['methods'].items():
                filename_jbl = '_'.join(
                    ['{}_{}'.format(filesafe(k), v) for k, v in self.config['mc_opts'].items()])
                filename_jbl += f"_{self.config['dgp_opts']['dgp_name']}_{self.config['dgp_opts']['n_samples']}"
                filename_jbl += f'_fn_{fn_number}_{model_name}'
                

                results_file = os.path.join(self.config['target_dir'], 'results_{}.jbl'.format(filename_jbl))
                if self.config['reload_results'] and os.path.exists(results_file):
                    results = joblib.load(results_file)
                else:
                    # Parallelize the loop
                    results = Parallel(n_jobs=mc_n_jobs, verbose=1)(
                        delayed(self.experiment)(
                            random_seed + exp_id,
                            fn_number,
                            model_instance,
                            n_folds,
                            inner_n_jobs,
                        )
                        for exp_id in range(self.config['mc_opts']['n_experiments']))

                    joblib.dump(results, results_file)

                _warn_if_nonfinite_method_outputs(results, fn_number, model_name)

                k = 0
                for sim_run in results:    
                    # [m][ii][j][k] : parameter | fn number | method | run 
                    for m in range(4) :  
                        result_distributions[m][ii][j][k] = sim_run[m]
                    k += 1
                j += 1
            ii += 1

        #---------------------------------------------------------------------------------------

        # Initialize arrays to store the calculated values
        num_i = len(self.config['dgp_opts']['fn'])
        num_j = len(self.config['methods'])

        mean_estimate = np.zeros((num_i, num_j))
        sd_mean_estimate = np.zeros((num_i, num_j))
        median_estimate = np.zeros((num_i, num_j))

        mean_variance = np.zeros((num_i, num_j))
        sd_mean_variance = np.zeros((num_i, num_j))
        median_variance = np.zeros((num_i, num_j))

        bias = np.zeros((num_i, num_j))
        sd_bias = np.zeros((num_i, num_j))
        median_bias = np.zeros((num_i, num_j))

        mse = np.zeros((num_i, num_j))
        sd_mse = np.zeros((num_i, num_j))

        studentized = {}

        coverage = np.zeros((num_i, num_j))

        interval_lengths = np.zeros((num_i, num_j))
        sd_interval_lengths = np.zeros((num_i, num_j))
        interval_median = np.zeros((num_i, num_j))

        average_time = np.zeros((num_i, num_j))
        sd_average_time = np.zeros((num_i, num_j))

        for i in range(num_i):
            studentized[i] = {}
            for j in range(num_j):

                # Mean variance: winsorize upper tail and guard against non-finite entries.
                variance_vals = _percentile_slice(result_distributions[1][i][j], upper_q=99)
                mean_variance[i][j], sd_mean_variance[i][j], median_variance[i][j] = _safe_mean_std_median(variance_vals)

                # Mean estimate: trim extremes and guard against empty slices.
                estimate_vals = _percentile_slice(result_distributions[0][i][j], lower_q=1, upper_q=99)
                mean_estimate[i][j], sd_mean_estimate[i][j], median_estimate[i][j] = _safe_mean_std_median(estimate_vals)

                # Bias / MSE
                bias_vals = estimate_vals - 4.05
                if bias_vals.size == 0:
                    bias[i][j] = np.nan
                    sd_bias[i][j] = np.nan
                    median_bias[i][j] = np.nan
                    mse[i][j] = np.nan
                    sd_mse[i][j] = np.nan
                else:
                    bias[i][j] = float(np.mean(bias_vals))
                    sd_bias[i][j] = float(np.std(bias_vals))
                    median_bias[i][j] = float(np.median(bias_vals))
                    mse[i][j] = float(np.mean(bias_vals**2))
                    sd_mse[i][j] = float(np.std(bias_vals**2))

                # Studentized values: avoid divide-by-zero / invalid values.
                raw_estimates = _finite_1d(result_distributions[0][i][j])
                if raw_estimates.size > 0 and np.isfinite(sd_bias[i][j]) and sd_bias[i][j] > 0:
                    studentized_vals = (raw_estimates - 4.05) / sd_bias[i][j]
                    studentized[i][j] = _finite_1d(studentized_vals)
                else:
                    studentized[i][j] = np.array([], dtype=float)

                #Coverage
                c_i = result_distributions[2][i][j]
                valid_intervals = []
                for interval in c_i:
                    if interval is None or len(interval) != 2:
                        continue
                    lo, hi = interval
                    if np.isfinite(lo) and np.isfinite(hi):
                        valid_intervals.append((lo, hi))

                if len(valid_intervals) > 0:
                    ub = np.array([interval[1] for interval in valid_intervals], dtype=float)
                    lb = np.array([interval[0] for interval in valid_intervals], dtype=float)
                    coverage[i][j] = float(np.mean((ub >= 4.05) * (lb <= 4.05)))
                    lengths = ub - lb
                    interval_lengths[i][j] = float(np.mean(lengths))
                    sd_interval_lengths[i][j] = float(np.std(lengths))
                    interval_median[i][j] = float(np.median(lengths))
                else:
                    coverage[i][j] = np.nan
                    interval_lengths[i][j] = np.nan
                    sd_interval_lengths[i][j] = np.nan
                    interval_median[i][j] = np.nan

                #Average time
                time_vals = _finite_1d(result_distributions[3][i][j])
                if time_vals.size > 0:
                    average_time[i][j] = float(np.mean(time_vals))
                    sd_average_time[i][j] = float(np.std(time_vals))
                else:
                    average_time[i][j] = np.nan
                    sd_average_time[i][j] = np.nan
                
        #---------------------------------------------------------------------------------------

        results_dict = {
            "DGP function": [],
            "Method": [],
            "Mean Estimate": [],
            "SD Estimate": [],
            "Median Estimate": [],
            "Mean Variance": [],
            "SD Variance": [],
            "Median Variance": [],
            "Bias": [],
            "SD Bias": [],
            "Median Bias": [],
            "MSE": [],
            "SD MSE": [],
            "Coverage": [],
            "Interval Length": [],
            "SD Interval Length": [],
            "Interval Median": [],
            "Average Time": [],
            "SD Average Time": []
        }

        i = 0
        for fn_number in self.config['dgp_opts']['fn']:
            j = 0
            for model_name, model_instance in self.config['methods'].items():
                results_dict["DGP function"].append(fn_number)
                results_dict["Method"].append(model_name)
                results_dict["Mean Estimate"].append(mean_estimate[i][j])
                results_dict["SD Estimate"].append(sd_mean_estimate[i][j])
                results_dict["Median Estimate"].append(median_estimate[i][j])
                results_dict["Mean Variance"].append(mean_variance[i][j])
                results_dict["SD Variance"].append(sd_mean_variance[i][j])
                results_dict["Median Variance"].append(median_variance[i][j])
                results_dict["Bias"].append(bias[i][j])
                results_dict["SD Bias"].append(sd_bias[i][j])
                results_dict["Median Bias"].append(median_bias[i][j])
                results_dict["MSE"].append(mse[i][j])
                results_dict["SD MSE"].append(sd_mse[i][j])
                results_dict["Coverage"].append(coverage[i][j])
                results_dict["Interval Length"].append(interval_lengths[i][j])
                results_dict["SD Interval Length"].append(sd_interval_lengths[i][j])
                results_dict["Interval Median"].append(interval_median[i][j])
                results_dict["Average Time"].append(average_time[i][j])
                results_dict["SD Average Time"].append(sd_average_time[i][j])

                j += 1
            i += 1

        # Convert the dictionary to a pandas DataFrame
        results_df = pd.DataFrame(results_dict)

        # Save the DataFrame to a CSV file
        self.config['param_str'] = '_'.join(
            ['{}_{}'.format(filesafe(k), v) for k, v in self.config['mc_opts'].items()])
        self.config['param_str'] += '_' + '_'.join(
                    ['{}_{}'.format(filesafe(k), v) for k, v in self.config['dgp_opts'].items()])
        self.config['param_str'] += '_' + '_'.join([str(k) for k, _ in self.config['methods'].items()])
        results_csv_filename = os.path.join(self.config['target_dir'], 'results_{}.csv'.format(self.config['param_str']))
        results_df.to_csv(results_csv_filename, index=False)
        print("Results saved to", results_csv_filename)

        #---------------------------------------------------------------------------------------

        df = self.config['mc_opts']['n_experiments'] - 1
        x = np.linspace(t.ppf(0.01, df), t.ppf(0.99, df), 100)

        dgp_name = self.config['dgp_opts']['dgp_name']
        folder_plots = os.path.join(self.config['target_dir'], f'plots_{dgp_name}')
        os.makedirs(folder_plots, exist_ok=True)

        i = 0
        for fn_number in self.config['dgp_opts']['fn']:
            j = 0
            for model_name, model_instance in self.config['methods'].items():
                studentized_vals = _finite_1d(studentized[i][j])
                filtered_data = _percentile_slice(studentized_vals, lower_q=2.5, upper_q=97.5)

                fig, ax = plt.subplots(1, 1)
                ax.plot(x, t.pdf(x, df), 'r-', lw=2, label='t pdf')
                if filtered_data.size > 0:
                    ax.hist(filtered_data, density=True, bins=max(1, int(np.sqrt(self.config['mc_opts']['n_experiments']))), histtype='stepfilled')
                else:
                    ax.text(0.5, 0.5, 'No finite studentized values', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.set_title(f'Studentized Distribution (DGP function {fn_number}, Method {model_name})')
                ax.legend()
                
                # Save the plot
                plot_filename = f"{folder_plots}/plot_{self.config['dgp_opts']['dgp_name']}_{self.config['dgp_opts']['n_samples']}_fn_{fn_number}_method_{model_name}.png"
                plt.savefig(plot_filename)
                print(f'Plot saved as {plot_filename}')
                
                plt.close()  # Close the plot to avoid memory leaks

                j += 1
            i += 1

                        

def semiparametrics_main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Ignore cached .jbl results and recompute, overwriting cache files.',
    )
    args = parser.parse_args(sys.argv[1:])

    config = importlib.import_module(args.config)
    runtime_config = deepcopy(config.CONFIG)
    if args.force_rerun:
        runtime_config['reload_results'] = False
    SemiParametricsMonteCarlo(runtime_config).run()


if __name__ == "__main__":
    semiparametrics_main()
