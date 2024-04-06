# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import numpy as np
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, LinearRegression,\
    ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import mliv.dgps_mediated as dgps
from mliv.ensemble import EnsembleIV, EnsembleIVStar
from mliv.rkhs import ApproxRKHSIVCV
from mliv.shape import LipschitzShapeIV, ShapeIV
from mliv.linear import OptimisticHedgeVsOptimisticHedge, StochasticOptimisticHedgeVsOptimisticHedge
from mliv.linear import L2OptimisticHedgeVsOGD, L2ProxGradient
from sklearn.pipeline import Pipeline
from mcpy.utils import filesafe

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
#from mliv.neuralnet.deepiv_fit import deep_iv_fit
from mliv.neuralnet.rbflayer import gaussian, inverse_multiquadric
from mliv.neuralnet import AGMM, KernelLayerMMDGMM, CentroidMMDGMM, KernelLossAGMM, MMDGMM


p = 0.1  # dropout prob of dropout layers throughout notebook
n_hidden = 100  # width of hidden layers throughout notebook

# For any method that use a projection of z into features g(z)
g_features = 100

# The kernel function
kernel_fn = gaussian
# kernel_fn = inverse_multiquadric

# Training params
learner_lr = 1e-4
adversary_lr = 1e-4
learner_l2 = 1e-3
adversary_l2 = 1e-4
adversary_norm_reg = 1e-3
n_epochs = 300
bs = 100
sigma = 2.0 / g_features
n_centers = 100
device = torch.cuda.current_device() if torch.cuda.is_available() else None


def _get_learner(n_t):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, 1))


def _get_adversary(n_z):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, 1))


def _get_adversary_g(n_z):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, g_features), nn.ReLU())


def _get(opts, key, default):
    return opts[key] if (key in opts) else default


def _get_model_opt(opts, key, default):
    model_enc = _get(opts, 'model', default)
    return ('avg' if model_enc == 0 else 'final')


def gen_data(opts):
    """
    opts : the dgp_opts from the config file
    """
    tau_fn = dgps.get_tau_fn(opts['fn'])
    W, Z, X, M, D, Y, tau_fn = dgps.get_data(opts['n_samples'], tau_fn)

    ind = np.where(D==0)[0]
    W0 = W[ind]
    X0 = X[ind,:]
    W0_test = np.zeros((opts['n_test'], 1+X.shape[1]))
    W0_test += np.median(np.column_stack((X0,W0)), axis=0, keepdims=True)
    W0_test[:, 2] = np.linspace(np.percentile(
            W0[:, 0], 5), np.percentile(W0[:, 0], 95), 1000)

    # True parameters
    b_yd = 2.0; b_ym = 1.0; b_yx = np.array([[-1.0],[-1.0]]); b_yu = -1.0; b_yw = 2.0; b_y0 = 2.0
    b_wx = np.array([[0.2],[0.2]]); b_wu = -0.6; b_w0 = 0.3
    b_md = -0.3; b_mx = np.array([[-0.5],[-0.5]]); b_mu = 0.4; b_m0 = 0.0
    
    gamma_1w = (b_yw*b_wu + b_yu)/b_wu
    gamma_1x = b_yw*b_wx + b_yx - gamma_1w*b_wx
    gamma_1m = b_ym
    gamma_10 = b_y0 + b_yd + b_yw*b_w0 - gamma_1w*b_w0

    gamma_0w = (gamma_1m*b_mu + gamma_1w*b_wu)/b_wu
    gamma_0x = gamma_1m*b_mx + gamma_1w*b_wx + gamma_1x - gamma_0w*b_wx
    gamma_00 = gamma_10 + gamma_1m*b_m0 + gamma_1w*b_w0 - gamma_0w*b_w0

    # True nuisance function
    tauinv_fn = dgps.get_tauinv_fn(opts['fn'])
    expected_te = gamma_00 + tauinv_fn(W0_test)@np.row_stack((gamma_0x, gamma_0w))

    # data, true_param
    return (W0_test, W, Z, X, M, D, Y), expected_te


def agmm(data, opts):
    print("GPU:", torch.cuda.is_available())
    W0_test, W, Z, X, M, D, Y = map(lambda x: torch.Tensor(x), data)
    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = torch.cat((M1,X1,Z1), dim=1)
    A1 = torch.cat((M1,X1,W1), dim=1)
    
    learner = _get_learner(A1.shape[1])
    adversary_fn = _get_adversary(A2.shape[1])
    agmm = AGMM(learner, adversary_fn).fit(A2, A1, Y1, learner_lr=learner_lr, adversary_lr=adversary_lr,
                                           learner_l2=learner_l2, adversary_l2=adversary_l2,
                                           n_epochs=_get(
                                               opts, 'n_epochs', n_epochs),
                                           bs=_get(opts, 'bs', bs),
                                           model_dir=str(Path.home()),
                                           device=device)
                    
    bridge_fs = torch.Tensor(agmm.predict(torch.cat((M,X,W), dim=1).to(device),
                        model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = torch.cat((X0,Z0), dim=1)
    B1 = torch.cat((X0,W0), dim=1)

    learner = _get_learner(B1.shape[1])
    adversary_fn = _get_adversary(B2.shape[1])
    agmm = AGMM(learner, adversary_fn).fit(B2, B1, bridge_fs, learner_lr=learner_lr, adversary_lr=adversary_lr,
                                            learner_l2=learner_l2, adversary_l2=adversary_l2,
                                            n_epochs=_get(
                                                opts, 'n_epochs', n_epochs),
                                            bs=_get(opts, 'bs', bs),
                                            model_dir=str(Path.home()),
                                            device=device)
    
    return agmm.predict(W0_test.to(device),
                        model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def klayerfixed(data, opts):
    W0_test, W, Z, X, M, D, Y = map(lambda x: torch.Tensor(x), data)
    
    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = torch.cat((M1,X1,Z1), dim=1)
    A1 = torch.cat((M1,X1,W1), dim=1)

    n_a = A2.shape[1]
    centers = np.tile(
        np.linspace(-4, 4, n_centers).reshape(-1, 1), (1, n_a))
    sigmas = np.ones((n_centers,)) * 2 / n_a

    learner = _get_learner(A1.shape[1])

    mmdgmm_fixed = KernelLayerMMDGMM(learner, lambda x: x, n_a, n_centers, kernel_fn,
                                     centers=centers, sigmas=sigmas, trainable=False)
    mmdgmm_fixed.fit(A2, A1, Y1, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    bridge_fs = torch.Tensor(mmdgmm_fixed.predict(torch.cat((M,X,W), dim=1).to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = torch.cat((X0,Z0), dim=1)
    B1 = torch.cat((X0,W0), dim=1)

    n_b = B2.shape[1]
    centers = np.tile(
        np.linspace(-4, 4, n_centers).reshape(-1, 1), (1, n_b))
    sigmas = np.ones((n_centers,)) * 2 / n_b

    learner = _get_learner(B1.shape[1])

    mmdgmm_fixed = KernelLayerMMDGMM(learner, lambda x: x, n_b, n_centers, kernel_fn,
                                     centers=centers, sigmas=sigmas, trainable=False)
    mmdgmm_fixed.fit(B2, B1, bridge_fs, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    
    return mmdgmm_fixed.predict(W0_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def klayertrained(data, opts):
    W0_test, W, Z, X, M, D, Y = map(lambda x: torch.Tensor(x), data)
    centers = np.random.uniform(-4, 4, size=(n_centers, g_features))
    sigmas = np.ones((n_centers,)) * sigma

    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = torch.cat((M1,X1,Z1), dim=1)
    A1 = torch.cat((M1,X1,W1), dim=1)

    learner = _get_learner(A1.shape[1])
    adversary_g = _get_adversary_g(A2.shape[1])
    klayermmdgmm = KernelLayerMMDGMM(learner, adversary_g, g_features,
                                     n_centers, kernel_fn, centers=centers, sigmas=sigmas)
    klayermmdgmm.fit(A2, A1, Y1, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    bridge_fs = torch.Tensor(klayermmdgmm.predict(torch.cat((M,X,W), dim=1).to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = torch.cat((X0,Z0), dim=1)
    B1 = torch.cat((X0,W0), dim=1)

    learner = _get_learner(B1.shape[1])
    adversary_g = _get_adversary_g(B2.shape[1])
    klayermmdgmm = KernelLayerMMDGMM(learner, adversary_g, g_features,
                                     n_centers, kernel_fn, centers=centers, sigmas=sigmas)
    klayermmdgmm.fit(B2, B1, bridge_fs, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    return klayermmdgmm.predict(W0_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def centroidmmd(data, opts):
    _, _, Z, X, M, D, _ = data
    ind = np.where(D==1)[0]
    M1 = M[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    A2 = np.column_stack((M1,X1,Z1))

    ind = np.where(D==0)[0]
    X0 = X[ind,:]
    Z0 = Z[ind]
    B2 = np.column_stack((X0,Z0))

    centers_A = KMeans(n_clusters=n_centers).fit(A2).cluster_centers_
    centers_B = KMeans(n_clusters=n_centers).fit(B2).cluster_centers_
    W0_test, W, Z, X, M, D, Y = map(lambda x: torch.Tensor(x), data)

    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = torch.cat((M1,X1,Z1), dim=1)
    A1 = torch.cat((M1,X1,W1), dim=1)

    learner = _get_learner(A1.shape[1])
    adversary_g = _get_adversary_g(A2.shape[1])
    centroid_mmd = CentroidMMDGMM(
        learner, adversary_g, kernel_fn, centers_A, np.ones(n_centers) * sigma)
    centroid_mmd.fit(A2, A1, Y1, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    bridge_fs = torch.Tensor(centroid_mmd.predict(torch.cat((M,X,W), dim=1).to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = torch.cat((X0,Z0), dim=1)
    B1 = torch.cat((X0,W0), dim=1)

    learner = _get_learner(B1.shape[1])
    adversary_g = _get_adversary_g(B2.shape[1])
    centroid_mmd = CentroidMMDGMM(
        learner, adversary_g, kernel_fn, centers_B, np.ones(n_centers) * sigma)
    centroid_mmd.fit(B2, B1, bridge_fs, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    return centroid_mmd.predict(W0_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def klossgmm(data, opts):
    W0_test, W, Z, X, M, D, Y = map(lambda x: torch.Tensor(x), data)

    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = torch.cat((M1,X1,Z1), dim=1)
    A1 = torch.cat((M1,X1,W1), dim=1)

    learner = _get_learner(A1.shape[1])
    adversary_g = _get_adversary_g(A2.shape[1])
    kernelgmm = KernelLossAGMM(learner, adversary_g, kernel_fn, sigma)
    kernelgmm.fit(A2, A1, Y1, learner_l2=learner_l2**2, adversary_l2=adversary_l2,
                  learner_lr=learner_lr, adversary_lr=adversary_lr,
                  n_epochs=_get(opts, 'n_epochs', n_epochs),
                  bs=_get(opts, 'bs', bs),
                  model_dir=str(Path.home()),
                  device=device)
    bridge_fs = torch.Tensor(kernelgmm.predict(torch.cat((M,X,W), dim=1).to(device),
                            model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = torch.cat((X0,Z0), dim=1)
    B1 = torch.cat((X0,W0), dim=1)

    learner = _get_learner(B1.shape[1])
    adversary_g = _get_adversary_g(B2.shape[1])
    kernelgmm = KernelLossAGMM(learner, adversary_g, kernel_fn, sigma)
    kernelgmm.fit(B2, B1, bridge_fs, learner_l2=learner_l2**2, adversary_l2=adversary_l2,
                  learner_lr=learner_lr, adversary_lr=adversary_lr,
                  n_epochs=_get(opts, 'n_epochs', n_epochs),
                  bs=_get(opts, 'bs', bs),
                  model_dir=str(Path.home()),
                  device=device)
    return kernelgmm.predict(W0_test.to(device),
                             model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))



def mse(param_estimates, true_params):
    return np.mean((np.array(param_estimates) - np.array(true_params))**2)


def rsquare(param_estimates, true_params):
    return 1 - mse(param_estimates, true_params) / np.var(true_params)


def _key(dic, value):
    return list(iter(dic.keys()))[np.argwhere(np.array(list(iter(dic.values()))) == value)[0, 0]]


def print_metrics(param_estimates, metric_results, config):
    out = open(os.path.join(config['target_dir'],
                            'print_metrics.csv'), 'a')
    methods = list(next(iter(metric_results.values())).keys())
    metrics = list(
        next(iter(next(iter(metric_results.values())).values())).keys())
    print(config['param_str'], file=out)
    for metric_name in metrics:
        if metric_name != 'raw':
            print(metric_name, file=out)
            print("&", "&".join(methods), file=out)
            for dgp_name, mdgp in metric_results.items():
                print(dgp_name, _key(dgps.fn_dict,
                                     config['dgp_opts']['fn']), end=" ", file=out)
                for method_name in mdgp.keys():
                    res = mdgp[method_name][metric_name]
                    mean_res = res.mean()
                    std_res = res.std() / np.sqrt(len(res))
                    print(r"& {:.3f} $\pm$ {:.3f}".format(
                        mean_res, 2 * std_res), end=" ", file=out)
            print(" ", file=out)
    out.close()
    return
