import sys
import os
import json
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
from nnpiv.diagnostics import relative_wellposedness_diagnostic
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


def _valid_interval(interval):
    if interval is None or len(interval) != 2:
        return None
    lo, hi = interval
    lo = float(np.asarray(lo).reshape(-1)[0])
    hi = float(np.asarray(hi).reshape(-1)[0])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return lo, hi


def _summarize_interval_metrics(intervals, true_param):
    valid = []
    for interval in intervals:
        parsed = _valid_interval(interval)
        if parsed is not None:
            valid.append(parsed)
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    lb = np.array([v[0] for v in valid], dtype=float)
    ub = np.array([v[1] for v in valid], dtype=float)
    lengths = ub - lb
    coverage = float(np.mean((ub >= true_param) * (lb <= true_param)))
    return coverage, float(np.mean(lengths)), float(np.std(lengths))


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

    def _diagnostics_enabled(self):
        diag_opts = self.config.get("diagnostics_opts", {})
        return bool(_get(diag_opts, "enabled", False))

    def _run_pre_estimation_diagnostic_A(self, fn_number):
        diag_opts = self.config.get("diagnostics_opts", {})
        n_aux = int(_get(diag_opts, "n_aux_samples", _get(self.config["dgp_opts"], "n_samples", 2000)))
        seed_base = int(_get(diag_opts, "seed", _get(self.config["mc_opts"], "seed", 123)))
        aux_seed = seed_base + 100_003 * int(fn_number)

        np_state = np.random.get_state()
        try:
            np.random.seed(aux_seed)
            tau_fn = dgps.get_tau_fn(fn_number)
            W, Z, X, M, D, _, _ = dgps.get_data(n_aux, tau_fn)
        finally:
            np.random.set_state(np_state)

        A = np.column_stack((M, X, W))
        C = np.column_stack((X, Z))
        C_prime = np.column_stack((M, X, Z))

        d_flat = np.asarray(D).reshape(-1)
        mask_s = (d_flat == 1).astype(int)
        mask_t = (d_flat == 0).astype(int)

        diag = relative_wellposedness_diagnostic(
            A=A,
            C=C,
            C_prime=C_prime,
            feature_map=_get(diag_opts, "feature_map", "rff"),
            n_features=int(_get(diag_opts, "n_features", 300)),
            gamma=_get(diag_opts, "gamma", "auto"),
            poly_degree=int(_get(diag_opts, "poly_degree", 3)),
            poly_include_bias=bool(_get(diag_opts, "poly_include_bias", False)),
            ridge_alpha=float(_get(diag_opts, "ridge_alpha", 1.0)),
            eta=float(_get(diag_opts, "eta", 1e-6)),
            random_state=int(_get(diag_opts, "random_state", 123)),
            mask_s=mask_s,
            mask_t=mask_t,
            return_details=False,
        )

        feature_meta = diag.get("feature_meta", {})
        return {
            "diagnostic_name": "relative_wellposedness_A",
            "runner": "semiparametric",
            "dgp_name": _get(self.config["dgp_opts"], "dgp_name", ""),
            "fn_number": int(fn_number),
            "stage": "mediated_joint_default",
            "aux_seed": int(aux_seed),
            "n_aux_samples": int(n_aux),
            "feature_map": feature_meta.get("feature_map", ""),
            "feature_gamma": feature_meta.get("gamma", None),
            "kappa2": diag["kappa2"],
            "kappa": diag["kappa"],
            "eta": diag["eta"],
            "ridge_alpha": diag["ridge_alpha"],
            "n_total": diag["n_total"],
            "n_s": diag["n_s"],
            "n_t": diag["n_t"],
            "n_features": diag["n_features"],
            "null_like_dim_sigma_s": diag["null_like_dim_sigma_s"],
            "max_diag_ratio_sigma_t_over_sigma_s": diag["max_diag_ratio_sigma_t_over_sigma_s"],
            "unstable_flag": diag["unstable_flag"],
            "min_eig_sigma_s_eta": diag["min_eig_sigma_s_eta"],
            "max_eig_whitened": diag["max_eig_whitened"],
        }

    def _save_pre_diagnostics(self, rows):
        if len(rows) == 0:
            return None, None

        diag_opts = self.config.get("diagnostics_opts", {})
        if not bool(_get(diag_opts, "save_csv", True)):
            return None, None

        df = pd.DataFrame(rows)
        csv_path = os.path.join(
            self.config["target_dir"],
            f"pre_diagnostics_relative_wellposedness_{self.config['param_str']}.csv",
        )
        json_path = os.path.join(
            self.config["target_dir"],
            f"pre_diagnostics_relative_wellposedness_{self.config['param_str']}.json",
        )
        df.to_csv(csv_path, index=False)

        payload = {"rows": []}
        for row in rows:
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, np.generic):
                    clean_row[key] = value.item()
                else:
                    clean_row[key] = value
            payload["rows"].append(clean_row)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=True)

        print("Pre-estimation diagnostics saved to", csv_path)
        return csv_path, json_path

    def _build_dml_model(self, model_instance, n_folds, inner_n_jobs, Y, D, M, W, Z, X):
        return DML_mediated(
            Y=Y, D=D, M=M, W=W, Z=Z, X1=X,
            estimator='MR',
            estimand='E[Y(1,M(0))]',
            model1=model_instance[0],
            modelq1=model_instance[1],
            n_folds=n_folds, n_rep=1, verbose=False,
            inner_n_jobs=inner_n_jobs,
            CHIM=_get(self.config['method_opts'], 'CHIM', False),
            nn_1=_get(self.config['method_opts'], 'nn_1', False),
            nn_q1=_get(self.config['method_opts'], 'nn_q1', False),
            fitargs1=_get(self.config['method_opts'], 'fitargs', None),
            fitargsq1=_get(self.config['method_opts'], 'fitargs', None),
            opts=_get(self.config['method_opts'], 'opts', None),
        )

    def _bootstrap_cis(self, rng, model_instance, n_folds, inner_n_jobs,
                       Y, D, M, W, Z, X, theta_hat, var_hat, n_bootstrap):
        n = Y.shape[0]
        theta_boot = []
        se_boot = []
        for _ in range(n_bootstrap):
            inds = rng.integers(0, n, size=n)
            dml_boot = self._build_dml_model(
                model_instance, n_folds=n_folds, inner_n_jobs=inner_n_jobs,
                Y=Y[inds], D=D[inds], M=M[inds], W=W[inds], Z=Z[inds], X=X[inds],
            )
            try:
                theta_b, var_b, _ = dml_boot.dml()
            except Exception:
                continue
            theta_b = float(np.asarray(theta_b).reshape(-1)[0])
            var_b = float(np.asarray(var_b).reshape(-1)[0])
            if not (np.isfinite(theta_b) and np.isfinite(var_b) and var_b >= 0):
                continue
            theta_boot.append(theta_b)
            se_boot.append(np.sqrt(var_b / n))

        theta_boot = np.asarray(theta_boot, dtype=float)
        se_boot = np.asarray(se_boot, dtype=float)
        if theta_boot.size < 2:
            return None, None

        pct_lo = float(np.percentile(theta_boot, 2.5))
        pct_hi = float(np.percentile(theta_boot, 97.5))
        percentile_ci = (pct_lo, pct_hi)

        se_hat = np.sqrt(max(var_hat, 0.0) / n)
        if not np.isfinite(se_hat) or se_hat <= 0:
            return percentile_ci, None

        mask = np.isfinite(se_boot) & (se_boot > 0)
        if np.sum(mask) < 2:
            return percentile_ci, None

        t_boot = (theta_boot[mask] - theta_hat) / se_boot[mask]
        t_lo = float(np.percentile(t_boot, 2.5))
        t_hi = float(np.percentile(t_boot, 97.5))
        studentized_ci = (float(theta_hat - t_hi * se_hat), float(theta_hat - t_lo * se_hat))
        return percentile_ci, studentized_ci

    def experiment(self, exp_id, fn_number, model_instance, n_folds, inner_n_jobs):
        ''' Runs an experiment on a single randomly generated instance and sample and returns
        the parameter estimates for each method 
        '''
        np.random.seed(exp_id)
        
        tau_fn = dgps.get_tau_fn(fn_number)
        W, Z, X, M, D, Y, tau_fn = dgps.get_data(_get(self.config['dgp_opts'], 'n_samples', 2000), tau_fn)

        dml_model = self._build_dml_model(
            model_instance, n_folds=n_folds, inner_n_jobs=inner_n_jobs,
            Y=Y, D=D, M=M, W=W, Z=Z, X=X
        )
                  
        start_time = time.time()
        theta, var, ci = dml_model.dml()
        end_time = time.time()
        elapsed_time = end_time - start_time
        theta = float(np.asarray(theta).reshape(-1)[0])
        var = float(np.asarray(var).reshape(-1)[0])
        ci = _valid_interval(ci)

        bootstrap_reps = _get(self.config['mc_opts'], 'bootstrap_reps', 0)
        bootstrap_percentile_ci = None
        bootstrap_studentized_ci = None
        if int(bootstrap_reps) > 0:
            rng = np.random.default_rng(exp_id + 100_000)
            bootstrap_percentile_ci, bootstrap_studentized_ci = self._bootstrap_cis(
                rng=rng,
                model_instance=model_instance,
                n_folds=n_folds,
                inner_n_jobs=inner_n_jobs,
                Y=Y, D=D, M=M, W=W, Z=Z, X=X,
                theta_hat=theta,
                var_hat=var,
                n_bootstrap=int(bootstrap_reps),
            )

        return (theta, var, ci, elapsed_time, bootstrap_percentile_ci, bootstrap_studentized_ci)

    def run(self):
        ''' Runs multiple experiments in parallel on randomly generated instances and samples and returns
        the parameter estimates for each method across all experiments
        '''
        random_seed = self.config['mc_opts']['seed']
        n_folds = int(_get(self.config['mc_opts'], 'n_folds', 5))
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

        pre_diagnostics_rows = []

        result_distributions = np.empty((6, len(self.config['dgp_opts']['fn']), 
            len(self.config['methods']), self.config['mc_opts']['n_experiments']), dtype=object)
            
        ii = 0
        for fn_number in self.config['dgp_opts']['fn']:
            if self._diagnostics_enabled():
                try:
                    pre_diagnostics_rows.append(self._run_pre_estimation_diagnostic_A(fn_number))
                except Exception as exc:
                    pre_diagnostics_rows.append({
                        "diagnostic_name": "relative_wellposedness_A",
                        "runner": "semiparametric",
                        "dgp_name": _get(self.config["dgp_opts"], "dgp_name", ""),
                        "fn_number": int(fn_number),
                        "stage": "mediated_joint_default",
                        "error": str(exc),
                    })
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
                    for m in range(6) :  
                        result_distributions[m][ii][j][k] = sim_run[m]
                    k += 1
                j += 1
            ii += 1

        #---------------------------------------------------------------------------------------

        # Initialize arrays to store the calculated values
        true_param = float(_get(self.config['dgp_opts'], 'true_param', 4.05))
        n_samples = int(_get(self.config['dgp_opts'], 'n_samples', 2000))
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

        empirical_sd = np.zeros((num_i, num_j))
        mean_reported_se = np.zeros((num_i, num_j))
        se_ratio = np.zeros((num_i, num_j))

        coverage_normal = np.zeros((num_i, num_j))
        interval_lengths_normal = np.zeros((num_i, num_j))
        sd_interval_lengths_normal = np.zeros((num_i, num_j))

        coverage_bootstrap_percentile = np.zeros((num_i, num_j))
        interval_lengths_bootstrap_percentile = np.zeros((num_i, num_j))
        sd_interval_lengths_bootstrap_percentile = np.zeros((num_i, num_j))

        coverage_bootstrap_studentized = np.zeros((num_i, num_j))
        interval_lengths_bootstrap_studentized = np.zeros((num_i, num_j))
        sd_interval_lengths_bootstrap_studentized = np.zeros((num_i, num_j))

        average_time = np.zeros((num_i, num_j))
        sd_average_time = np.zeros((num_i, num_j))

        for i in range(num_i):
            studentized[i] = {}
            for j in range(num_j):

                # Mean variance: winsorize upper tail and guard against non-finite entries.
                variance_vals = _percentile_slice(result_distributions[1][i][j], upper_q=99)
                mean_variance[i][j], sd_mean_variance[i][j], median_variance[i][j] = _safe_mean_std_median(variance_vals)

                # Mean estimate and empirical SD from finite values.
                estimate_vals = _finite_1d(result_distributions[0][i][j])
                mean_estimate[i][j], sd_mean_estimate[i][j], median_estimate[i][j] = _safe_mean_std_median(estimate_vals)
                empirical_sd[i][j] = float(np.std(estimate_vals, ddof=1)) if estimate_vals.size > 1 else np.nan

                # Bias / MSE
                bias_vals = estimate_vals - true_param
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
                    studentized_vals = (raw_estimates - true_param) / sd_bias[i][j]
                    studentized[i][j] = _finite_1d(studentized_vals)
                else:
                    studentized[i][j] = np.array([], dtype=float)

                # Mean reported SE and SD/SE ratio.
                var_vals = _finite_1d(result_distributions[1][i][j])
                se_vals = np.sqrt(np.maximum(var_vals, 0.0) / max(1, n_samples))
                mean_reported_se[i][j] = float(np.mean(se_vals)) if se_vals.size > 0 else np.nan
                if np.isfinite(empirical_sd[i][j]) and np.isfinite(mean_reported_se[i][j]) and mean_reported_se[i][j] > 0:
                    se_ratio[i][j] = float(empirical_sd[i][j] / mean_reported_se[i][j])
                else:
                    se_ratio[i][j] = np.nan

                # Coverage and CI length summaries for normal and bootstrap CIs.
                cov, mean_len, sd_len = _summarize_interval_metrics(result_distributions[2][i][j], true_param)
                coverage_normal[i][j] = cov
                interval_lengths_normal[i][j] = mean_len
                sd_interval_lengths_normal[i][j] = sd_len

                cov, mean_len, sd_len = _summarize_interval_metrics(result_distributions[4][i][j], true_param)
                coverage_bootstrap_percentile[i][j] = cov
                interval_lengths_bootstrap_percentile[i][j] = mean_len
                sd_interval_lengths_bootstrap_percentile[i][j] = sd_len

                cov, mean_len, sd_len = _summarize_interval_metrics(result_distributions[5][i][j], true_param)
                coverage_bootstrap_studentized[i][j] = cov
                interval_lengths_bootstrap_studentized[i][j] = mean_len
                sd_interval_lengths_bootstrap_studentized[i][j] = sd_len

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
            "Empirical SD": [],
            "Mean Variance": [],
            "SD Variance": [],
            "Median Variance": [],
            "Mean Reported SE": [],
            "SE Ratio": [],
            "Bias": [],
            "SD Bias": [],
            "Median Bias": [],
            "MSE": [],
            "SD MSE": [],
            "Normal Coverage": [],
            "Normal CI Length": [],
            "SD Normal CI Length": [],
            "Percentile Bootstrap Coverage": [],
            "Percentile Bootstrap CI Length": [],
            "SD Percentile Bootstrap CI Length": [],
            "Studentized Bootstrap Coverage": [],
            "Studentized Bootstrap CI Length": [],
            "SD Studentized Bootstrap CI Length": [],
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
                results_dict["Empirical SD"].append(empirical_sd[i][j])
                results_dict["Mean Variance"].append(mean_variance[i][j])
                results_dict["SD Variance"].append(sd_mean_variance[i][j])
                results_dict["Median Variance"].append(median_variance[i][j])
                results_dict["Mean Reported SE"].append(mean_reported_se[i][j])
                results_dict["SE Ratio"].append(se_ratio[i][j])
                results_dict["Bias"].append(bias[i][j])
                results_dict["SD Bias"].append(sd_bias[i][j])
                results_dict["Median Bias"].append(median_bias[i][j])
                results_dict["MSE"].append(mse[i][j])
                results_dict["SD MSE"].append(sd_mse[i][j])
                results_dict["Normal Coverage"].append(coverage_normal[i][j])
                results_dict["Normal CI Length"].append(interval_lengths_normal[i][j])
                results_dict["SD Normal CI Length"].append(sd_interval_lengths_normal[i][j])
                results_dict["Percentile Bootstrap Coverage"].append(coverage_bootstrap_percentile[i][j])
                results_dict["Percentile Bootstrap CI Length"].append(interval_lengths_bootstrap_percentile[i][j])
                results_dict["SD Percentile Bootstrap CI Length"].append(sd_interval_lengths_bootstrap_percentile[i][j])
                results_dict["Studentized Bootstrap Coverage"].append(coverage_bootstrap_studentized[i][j])
                results_dict["Studentized Bootstrap CI Length"].append(interval_lengths_bootstrap_studentized[i][j])
                results_dict["SD Studentized Bootstrap CI Length"].append(sd_interval_lengths_bootstrap_studentized[i][j])
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

        if self._diagnostics_enabled():
            self._save_pre_diagnostics(pre_diagnostics_rows)

        #---------------------------------------------------------------------------------------

        if not _get(self.config['mc_opts'], 'skip_plots', False):
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
