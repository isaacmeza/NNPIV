import os
import sys
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import joblib
import argparse
import importlib
from itertools import product
import collections
from copy import deepcopy
from mcpy.utils import filesafe
from nnpiv.diagnostics import relative_wellposedness_diagnostic


def _get(opts, key, default):
    return opts[key] if (key in opts) else default


def _is_failed_np_run(run):
    if run is None:
        return True
    if not isinstance(run, (tuple, list)) or len(run) != 2:
        return True
    return False


def _check_valid_config(config):
    assert 'dgps' in config, "config dict must contain dgps"
    assert 'dgp_opts' in config, "config dict must contain dgp_opts"
    assert 'method_opts' in config, "config dict must contain method_opts"
    assert 'mc_opts' in config, "config dict must contain mc_opts"
    assert 'metrics' in config, "config dict must contain metrics"
    assert 'methods' in config, "config dict must contain methods"
    assert 'plots' in config, "config dict must contain plots"
    assert 'target_dir' in config, "config must contain target_dir"
    assert 'reload_results' in config, "config must contain reload_results"
    assert 'n_experiments' in config['mc_opts'], "config[mc_opts] must contain n_experiments"
    assert 'seed' in config['mc_opts'], "config[mc_opts] must contain seed"


def _get(opts, key, default):
    return opts[key] if (key in opts) else default


class MonteCarlo:

    def __init__(self, config):
        self.config = config
        _check_valid_config(self.config)
        config['param_str'] = '_'.join(
            ['{}_{}'.format(filesafe(k), v) for k, v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(
            ['{}_{}'.format(filesafe(k), v) for k, v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join(
            [filesafe(method_name) for method_name in self.config['methods'].keys()])
        return

    def _diagnostics_enabled(self):
        diag_opts = self.config.get("diagnostics_opts", {})
        return bool(_get(diag_opts, "enabled", False))

    def _skip_failed_runs_enabled(self):
        mc_opts = self.config.get("mc_opts", {})
        return bool(_get(mc_opts, "skip_failed_runs", False))

    def _extract_pre_diag_arrays(self, data):
        diag_opts = self.config.get("diagnostics_opts", {})

        # Dict pathway: either explicit A/C/C_prime, or nested aliases A1/A2/B2.
        if isinstance(data, dict):
            if all(k in data for k in ("A", "C", "C_prime")):
                return np.asarray(data["A"]), np.asarray(data["C"]), np.asarray(data["C_prime"]), "dict_A_C_Cprime"
            if all(k in data for k in ("A1", "A2", "B2")):
                return np.asarray(data["A1"]), np.asarray(data["B2"]), np.asarray(data["A2"]), "dict_A1_A2_B2"
            raise ValueError(
                "Diagnostic A data dict must contain either keys (A, C, C_prime) "
                "or (A1, A2, B2)."
            )

        # Tuple/list pathway defaulting to nested_npiv data layout:
        # (B1_test, A1, A2, B1, B2, Y) -> A=A1, C_prime=A2, C=B2
        if isinstance(data, (tuple, list)):
            a_idx = int(_get(diag_opts, "a_index", 1))
            cprime_idx = int(_get(diag_opts, "cprime_index", 2))
            c_idx = int(_get(diag_opts, "c_index", 4))
            max_idx = max(a_idx, cprime_idx, c_idx)
            if len(data) <= max_idx:
                raise ValueError(
                    "Diagnostic A tuple indexing is out of bounds. "
                    f"Got len(data)={len(data)} and required max index={max_idx}."
                )
            return (
                np.asarray(data[a_idx]),
                np.asarray(data[c_idx]),
                np.asarray(data[cprime_idx]),
                "tuple_indexed",
            )

        raise ValueError(
            "Unsupported DGP data type for diagnostics. "
            f"Expected dict/tuple/list, got {type(data).__name__}."
        )

    def _run_pre_estimation_diagnostic_A(self, dgp_name, dgp_fn):
        diag_opts = self.config.get("diagnostics_opts", {})
        n_aux = int(_get(diag_opts, "n_aux_samples", _get(self.config["dgp_opts"], "n_samples", 2000)))
        seed_base = int(_get(diag_opts, "seed", _get(self.config["mc_opts"], "seed", 123)))
        fn_value = _get(self.config["dgp_opts"], "fn", -1)
        try:
            fn_scalar = int(fn_value)
        except Exception:
            fn_scalar = -1
        name_hash = sum(ord(ch) for ch in str(dgp_name))
        aux_seed = int(seed_base + 100_003 * max(0, fn_scalar) + 1_009 * name_hash)

        aux_opts = deepcopy(self.config["dgp_opts"])
        aux_opts["n_samples"] = n_aux

        np_state = np.random.get_state()
        try:
            np.random.seed(aux_seed)
            data, _ = dgp_fn(aux_opts)
        finally:
            np.random.set_state(np_state)

        A, C, C_prime, extraction_mode = self._extract_pre_diag_arrays(data)

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
            return_details=False,
        )

        feature_meta = diag.get("feature_meta", {})
        return {
            "diagnostic_name": "relative_wellposedness_A",
            "runner": "nonparametric",
            "dgp_name": str(dgp_name),
            "fn_number": fn_scalar,
            "stage": "nested_npiv_default",
            "aux_seed": int(aux_seed),
            "n_aux_samples": int(n_aux),
            "data_extraction_mode": extraction_mode,
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

        os.makedirs(self.config["target_dir"], exist_ok=True)
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

    def experiment(self, exp_id):
        ''' Runs an experiment on a single randomly generated instance and sample and returns
        the parameter estimates for each method and the evaluated metrics for each method
        '''
        try:
            np.random.seed(exp_id)

            param_estimates = {}
            true_params = {}
            for dgp_name, dgp_fn in self.config['dgps'].items():
                data, true_param = dgp_fn(self.config['dgp_opts'])
                true_params[dgp_name] = true_param
                param_estimates[dgp_name] = {}
                for method_name, method in self.config['methods'].items():
                    param_estimates[dgp_name][method_name] = method(
                        data, self.config['method_opts'])

            return param_estimates, true_params
        except Exception as exc:
            if self._skip_failed_runs_enabled():
                print(
                    f"[WARN] Skipping failed nonparametric run: exp_id={exp_id}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                return None
            raise

    def run(self):
        ''' Runs multiple experiments in parallel on randomly generated instances and samples and returns
        the parameter estimates for each method and the evaluated metrics for each method across all
        experiments
        '''
        random_seed = self.config['mc_opts']['seed']

        if not os.path.exists(self.config['target_dir']):
            os.makedirs(self.config['target_dir'])

        pre_diagnostics_rows = []
        if self._diagnostics_enabled():
            for dgp_name, dgp_fn in self.config['dgps'].items():
                try:
                    pre_diagnostics_rows.append(self._run_pre_estimation_diagnostic_A(dgp_name, dgp_fn))
                except Exception as exc:
                    pre_diagnostics_rows.append({
                        "diagnostic_name": "relative_wellposedness_A",
                        "runner": "nonparametric",
                        "dgp_name": str(dgp_name),
                        "fn_number": int(_get(self.config["dgp_opts"], "fn", -1)) if str(_get(self.config["dgp_opts"], "fn", "-1")).lstrip("-").isdigit() else -1,
                        "stage": "nested_npiv_default",
                        "error": str(exc),
                    })

        results_file = os.path.join(
            self.config['target_dir'], 'results_{}.jbl'.format(self.config['param_str']))
        if self.config['reload_results'] and os.path.exists(results_file):
            results = joblib.load(results_file)
        else:
            results = Parallel(n_jobs=_get(self.config['mc_opts'], 'n_jobs', -1), verbose=1)(
                delayed(self.experiment)(random_seed + exp_id)
                for exp_id in range(self.config['mc_opts']['n_experiments']))
            joblib.dump(results, results_file)

        failed_runs = sum(1 for run in results if _is_failed_np_run(run))
        if failed_runs > 0:
            print(
                f"[WARN] skipped {failed_runs}/{len(results)} failed nonparametric runs."
            )
        results = [run for run in results if not _is_failed_np_run(run)]
        if len(results) == 0:
            raise RuntimeError(
                "All nonparametric runs failed; no valid results available for aggregation."
            )

        param_estimates = {}
        metric_results = {}
        n_valid = len(results)
        for dgp_name in self.config['dgps'].keys():
            param_estimates[dgp_name] = {}
            metric_results[dgp_name] = {}
            for method_name in self.config['methods'].keys():
                param_estimates[dgp_name][method_name] = np.array(
                    [results[i][0][dgp_name][method_name] for i in range(n_valid)])
                metric_results[dgp_name][method_name] = {}
                for metric_name, metric_fn in self.config['metrics'].items():
                    metric_results[dgp_name][method_name][metric_name] = np.array([
                        metric_fn(results[i][0][dgp_name][method_name], results[i][1][dgp_name])
                        for i in range(n_valid)
                    ])

        for plot_name, plot_fn in self.config['plots'].items():
            plot_fn(param_estimates, metric_results, self.config)

        if self._diagnostics_enabled():
            self._save_pre_diagnostics(pre_diagnostics_rows)

        return param_estimates, metric_results


class NonParametricsMonteCarlo:

    def __init__(self, config):
        self.config = config
        _check_valid_config(self.config)
        config['param_str'] = '_'.join(['{}_{}'.format(filesafe(
            k), self._stringify_param(v)) for k, v in self.config['mc_opts'].items()])
        config['param_str'] += '_' + '_'.join(['{}_{}'.format(filesafe(
            k), self._stringify_param(v)) for k, v in self.config['dgp_opts'].items()])
        config['param_str'] += '_' + '_'.join(
            [filesafe(method_name) for method_name in self.config['methods'].keys()])
        return

    def _stringify_param(self, param):
        if hasattr(param, "__len__"):
            return '{}_to_{}'.format(np.min(param), np.max(param))
        else:
            return param

    def run(self):
        dgp_sweep_params = []
        dgp_sweep_param_vals = []
        for dgp_key, dgp_val in self.config['dgp_opts'].items():
            if hasattr(dgp_val, "__len__"):
                dgp_sweep_params.append(dgp_key)
                dgp_sweep_param_vals.append(dgp_val)

        n_sweeps = len(list(product(*dgp_sweep_param_vals)))
        if 'cluster_opts' in self.config:
            n_nodes = _get(self.config['cluster_opts'], 'n_nodes', 1)
            node_id = _get(self.config['cluster_opts'], 'node_id', 0)
        else:
            n_nodes = 1
            node_id = 0
        start_sweep, end_sweep = 0, 0
        if node_id < n_nodes - 1:
            node_splits = np.array_split(np.arange(n_sweeps), n_nodes - 1)
            start_sweep, end_sweep = node_splits[node_id][0], node_splits[node_id][-1]

        sweep_keys = []
        sweep_params = []
        sweep_metrics = []
        inst_config = deepcopy(self.config)
        # This is the node that loads results and plots sweep plots
        if (n_nodes > 1) and (node_id == n_nodes - 1):
            inst_config['reload_results'] = True
            inst_config['plots'] = {}
        for it, vec in enumerate(product(*dgp_sweep_param_vals)):
            if (node_id == n_nodes - 1) or ((it >= start_sweep) and (it <= end_sweep)):
                setting = list(zip(dgp_sweep_params, vec))
                for k, v in setting:
                    inst_config['dgp_opts'][k] = v
                params, metrics = MonteCarlo(inst_config).run()
                sweep_keys.append(setting)
                sweep_params.append(params)
                sweep_metrics.append(metrics)

        if node_id == n_nodes - 1:
            for plot_key, plot_fn in self.config['sweep_plots'].items():
                plot_fn(plot_key, sweep_keys, sweep_params,
                            sweep_metrics, self.config)

        return sweep_keys, sweep_params, sweep_metrics


def nonparametrics_main():
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
    MonteCarlo(runtime_config).run()


if __name__ == "__main__":
    nonparametrics_main()
