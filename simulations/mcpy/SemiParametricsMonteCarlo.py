import sys
import os

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
from nnpiv.semiparametrics import DML_mediated, DML_joint_mediated
import time

def _get(opts, key, default):
    return opts[key] if (key in opts) else default


def _check_valid_config(config):
    assert 'dgp_opts' in config, "config dict must contain dgp_opts"
    assert 'estimator' in config, "config dict must contain estimator"
    assert config['estimator'] in ['joint', 'sequential'], "config dict must contain estimator as 'joint' or 'sequential'"
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

    def experiment(self, exp_id, fn_number, model_instance):
        ''' Runs an experiment on a single randomly generated instance and sample and returns
        the parameter estimates for each method 
        '''
        np.random.seed(exp_id)
        
        tau_fn = dgps.get_tau_fn(fn_number)
        W, Z, X, M, D, Y, tau_fn = dgps.get_data(_get(self.config['dgp_opts'], 'n_samples', 2000), tau_fn)

        if self.config['estimator'] == 'sequential':
            dml_model = DML_mediated(Y=Y, D=D, M=M, W=W, Z=Z, X1=X,
                                estimator='MR',
                                estimand='E[Y(1,M(0))]',
                                model1 = model_instance[0],
                                model2 = model_instance[1],
                                modelq1 = model_instance[2],
                                modelq2 = model_instance[3],
                                n_folds=5, n_rep=1, verbose=False,
                                CHIM = _get(self.config['method_opts'], 'CHIM', False),
                                nn_1 = _get(self.config['method_opts'], 'nn_1', False),
                                nn_2 = _get(self.config['method_opts'], 'nn_2', False),
                                nn_q1 = _get(self.config['method_opts'], 'nn_q1', False),
                                nn_q2 = _get(self.config['method_opts'], 'nn_q2', False),
                                fitargs1 = _get(self.config['method_opts'], 'fitargs', None),
                                fitargs2 = _get(self.config['method_opts'], 'fitargs', None),
                                fitargsq1 = _get(self.config['method_opts'], 'fitargs', None),
                                fitargsq2 = _get(self.config['method_opts'], 'fitargs', None),
                                opts = _get(self.config['method_opts'], 'opts', None))
        else :
            dml_model = DML_joint_mediated(Y=Y, D=D, M=M, W=W, Z=Z, X1=X,
                                estimator='MR',
                                estimand='E[Y(1,M(0))]',
                                model1 = model_instance[0],
                                modelq1 = model_instance[1],
                                n_folds=5, n_rep=1, verbose=False,
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
                    results = Parallel(n_jobs=_get(self.config['mc_opts'], 'n_jobs', -1), verbose=1)(
                        delayed(self.experiment)(random_seed + exp_id , fn_number, model_instance) 
                        for exp_id in range(self.config['mc_opts']['n_experiments']))

                    joblib.dump(results, results_file)

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

                #Mean variance
                # Calculate the 99th percentiles
                percentile_99 = np.percentile(result_distributions[1][i][j], 99)
                # Filter the data to include values less than 99th percentile range
                filtered_data = result_distributions[1][i][j][(result_distributions[1][i][j] <= percentile_99)]

                mean_variance[i][j] = np.mean(filtered_data)
                sd_mean_variance[i][j] = np.std(filtered_data)
                median_variance[i][j] = np.median(filtered_data)

                #Mean estimate
                # Calculate the 1st and 99th percentiles
                percentile_1 = np.percentile(result_distributions[0][i][j], 1)
                percentile_99 = np.percentile(result_distributions[0][i][j], 99)
                # Filter the data to include values within the 1st to 99th percentile range
                filtered_data = result_distributions[0][i][j][(result_distributions[0][i][j] >= percentile_1) & (result_distributions[0][i][j] <= percentile_99)]

                mean_estimate[i][j] = np.mean(filtered_data)
                sd_mean_estimate[i][j] = np.std(filtered_data)
                median_estimate[i][j] = np.median(filtered_data)

                #Bias
                bias[i][j] = np.mean(filtered_data-4.05)
                sd_bias[i][j] = np.std(filtered_data-4.05)
                median_bias[i][j] = np.median(filtered_data-4.05)
                #MSE
                mse[i][j] = np.mean((filtered_data-4.05)**2)
                sd_mse[i][j] = np.std((filtered_data-4.05)**2)
                #Studentized
                studentized[i][j] = (result_distributions[0][i][j]-4.05)/(sd_bias[i][j])

                #Coverage
                c_i = result_distributions[2][i][j]
                ub = np.array([interval[1] for interval in c_i])
                lb = np.array([interval[0] for interval in c_i])
                coverage[i][j] =  np.mean((ub >= 4.05) * (lb <= 4.05))
                #Length
                lengths = ub-lb
                interval_lengths[i][j] = np.mean(lengths)
                sd_interval_lengths[i][j] = np.std(lengths)
                interval_median[i][j] = np.median(lengths)

                #Average time
                average_time[i][j] = np.mean(result_distributions[3][i][j])
                sd_average_time[i][j] = np.std(result_distributions[3][i][j])
                
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

                # Calculate the 2.5th and 97.5th percentiles
                percentile_25 = np.percentile(studentized[i][j], 2.5)
                percentile_975 = np.percentile(studentized[i][j], 97.5)

                # Create a filtered dataset containing values within the 2.5th and 97.5th percentiles
                filtered_data = studentized[i][j][(studentized[i][j] >= percentile_25) & (studentized[i][j] <= percentile_975)]

                fig, ax = plt.subplots(1, 1)
                ax.plot(x, t.pdf(x, df), 'r-', lw=2, label='t pdf')
                ax.hist(filtered_data, density=True, bins=int(np.sqrt(self.config['mc_opts']['n_experiments'])), histtype='stepfilled')
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
    args = parser.parse_args(sys.argv[1:])

    config = importlib.import_module(args.config)
    SemiParametricsMonteCarlo(config.CONFIG).run()


if __name__ == "__main__":
    semiparametrics_main()
