# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import numpy as np
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, LinearRegression,\
    ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import simulations.dgps_nested as dgps
from nnpiv.ensemble import EnsembleIV, EnsembleIVStar
from nnpiv.rkhs import ApproxRKHSIVCV
from nnpiv.shape import LipschitzShapeIV, ShapeIV
from nnpiv.linear import OptimisticHedgeVsOptimisticHedge, StochasticOptimisticHedgeVsOptimisticHedge
from nnpiv.linear import L2OptimisticHedgeVsOGD, L2ProxGradient
from sklearn.pipeline import Pipeline
from mcpy.utils import filesafe

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
#from nnpiv.neuralnet.deepiv_fit import deep_iv_fit
from nnpiv.neuralnet.rbflayer import gaussian, inverse_multiquadric
from nnpiv.neuralnet import AGMM, KernelLayerMMDGMM, CentroidMMDGMM, KernelLossAGMM, MMDGMM

from nnpiv.neuralnet import AGMM2, AGMM2L2

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
    A1, A2, B1, B2, Y, tau_fn = dgps.get_data(opts['n_samples'], opts['n_a'],
                                    opts['n_b'], tau_fn, opts['dgp_num'])

    if opts['gridtest']:
        B1_test = np.zeros((opts['n_test'], B1.shape[1]))
        B1_test += np.median(B1, axis=0, keepdims=True)
        B1_test[:, 0] = np.linspace(np.percentile(
            B1[:, 0], 5), np.percentile(B1[:, 0], 95), 1000)
    else:
        _, _, B1_test, _, _, _ = dgps.get_data(opts['n_test'], opts['n_a'],
                                    opts['n_b'], tau_fn, opts['dgp_num'])
        B1_test = B1_test[np.argsort(B1_test[:, 0])]
    expected_te = tau_fn(B1_test)

    # data, true_param
    return (B1_test, A1, A2, B1, B2, Y), expected_te


def agmm(data, opts):
    print("GPU:", torch.cuda.is_available())
    B1_test, A1, A2, B1, B2, Y = map(lambda x: torch.Tensor(x), data)
    #First stage
    learner = _get_learner(A1.shape[1])
    adversary_fn = _get_adversary(A2.shape[1])
    agmm = AGMM(learner, adversary_fn).fit(A2, A1, Y, learner_lr=learner_lr, adversary_lr=adversary_lr,
                                           learner_l2=learner_l2, adversary_l2=adversary_l2,
                                           n_epochs=_get(
                                               opts, 'n_epochs', n_epochs),
                                           bs=_get(opts, 'bs', bs),
                                           model_dir=str(Path.home()),
                                           device=device)
    bridge_fs = torch.Tensor(agmm.predict(A1.to(device),
                        model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
    learner = _get_learner(B1.shape[1])
    adversary_fn = _get_adversary(B2.shape[1])
    agmm = AGMM(learner, adversary_fn).fit(B2, B1, bridge_fs, learner_lr=learner_lr, adversary_lr=adversary_lr,
                                            learner_l2=learner_l2, adversary_l2=adversary_l2,
                                            n_epochs=_get(
                                                opts, 'n_epochs', n_epochs),
                                            bs=_get(opts, 'bs', bs),
                                            model_dir=str(Path.home()),
                                            device=device)
    
    return agmm.predict(B1_test.to(device),
                        model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def agmm2(data, opts):
    print("GPU:", torch.cuda.is_available())
    B1_test, A1, A2, B1, B2, Y = map(lambda x: torch.Tensor(x), data)

    model =  AGMM2(learnerh = _get_learner(B1.shape[1]), learnerg = _get_learner(A1.shape[1]),
                     adversary1 = _get_adversary(A2.shape[1]), adversary2 = _get_adversary(B2.shape[1]))
    
      
    agmm2 = model.fit(A1, B1, B2, A2, Y, learner_l2=learner_l2, adversary_l2=adversary_l2, adversary_norm_reg=adversary_norm_reg, learner_norm_reg=adversary_norm_reg,
            learner_lr=learner_lr, adversary_lr=adversary_lr, 
            n_epochs=_get(opts, 'n_epochs', n_epochs)*2, bs=_get(opts, 'bs', bs), 
            train_learner_every=1, train_adversary_every=1,
            model_dir=str(Path.home()), device=device)
    pred, _ = agmm2.predict(B1_test.to(device), A1.to(device),
                        model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))
    return pred


def agmm2l2(data, opts):
    print("GPU:", torch.cuda.is_available())
    B1_test, A1, A2, B1, B2, Y = map(lambda x: torch.Tensor(x), data)

    model =  AGMM2L2(learnerh = _get_learner(B1.shape[1]), learnerg = _get_learner(A1.shape[1]),
                     adversary1 = _get_adversary(A2.shape[1]), adversary2 = _get_adversary(B2.shape[1]))
    
      
    agmm2l2 = model.fit(A1, B1, B2, A2, Y, learner_l2=learner_l2, adversary_l2=adversary_l2, adversary_norm_reg=adversary_norm_reg, learner_norm_reg=adversary_norm_reg,
            learner_lr=learner_lr, adversary_lr=adversary_lr, 
            n_epochs=_get(opts, 'n_epochs', n_epochs)*2, bs=_get(opts, 'bs', bs), 
            train_learner_every=1, train_adversary_every=1,
            model_dir=str(Path.home()), device=device)
    pred, _ = agmm2l2.predict(B1_test.to(device), A1.to(device),
                        model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))
    return pred


def klayerfixed(data, opts):
    B1_test, A1, A2, B1, B2, Y = map(lambda x: torch.Tensor(x), data)
    
    #First stage
    n_a = A2.shape[1]
    centers = np.tile(
        np.linspace(-4, 4, n_centers).reshape(-1, 1), (1, n_a))
    sigmas = np.ones((n_centers,)) * 2 / n_a

    learner = _get_learner(A1.shape[1])

    mmdgmm_fixed = KernelLayerMMDGMM(learner, lambda x: x, n_a, n_centers, kernel_fn,
                                     centers=centers, sigmas=sigmas, trainable=False)
    mmdgmm_fixed.fit(A2, A1, Y, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    bridge_fs = torch.Tensor(mmdgmm_fixed.predict(A1.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
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
    
    return mmdgmm_fixed.predict(B1_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def klayertrained(data, opts):
    B1_test, A1, A2, B1, B2, Y = map(lambda x: torch.Tensor(x), data)
    centers = np.random.uniform(-4, 4, size=(n_centers, g_features))
    sigmas = np.ones((n_centers,)) * sigma

    #First stage
    learner = _get_learner(A1.shape[1])
    adversary_g = _get_adversary_g(A2.shape[1])
    klayermmdgmm = KernelLayerMMDGMM(learner, adversary_g, g_features,
                                     n_centers, kernel_fn, centers=centers, sigmas=sigmas)
    klayermmdgmm.fit(A2, A1, Y, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    bridge_fs = torch.Tensor(klayermmdgmm.predict(A1.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
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
    return klayermmdgmm.predict(B1_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def centroidmmd(data, opts):
    _, _, A2, _, B2, _ = data
    centers_A = KMeans(n_clusters=n_centers).fit(A2).cluster_centers_
    centers_B = KMeans(n_clusters=n_centers).fit(B2).cluster_centers_
    B1_test, A1, A2, B1, B2, Y = map(lambda x: torch.Tensor(x), data)

    #First stage
    learner = _get_learner(A1.shape[1])
    adversary_g = _get_adversary_g(A2.shape[1])
    centroid_mmd = CentroidMMDGMM(
        learner, adversary_g, kernel_fn, centers_A, np.ones(n_centers) * sigma)
    centroid_mmd.fit(A2, A1, Y, learner_l2=learner_l2, adversary_l2=adversary_l2,
                     adversary_norm_reg=adversary_norm_reg,
                     learner_lr=learner_lr, adversary_lr=adversary_lr,
                     n_epochs=_get(opts, 'n_epochs', n_epochs),
                     bs=_get(opts, 'bs', bs),
                     model_dir=str(Path.home()),
                     device=device)
    bridge_fs = torch.Tensor(centroid_mmd.predict(A1.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
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
    return centroid_mmd.predict(B1_test.to(device),
                                model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))


def klossgmm(data, opts):
    B1_test, A1, A2, B1, B2, Y = map(lambda x: torch.Tensor(x), data)

    #First stage
    learner = _get_learner(A1.shape[1])
    adversary_g = _get_adversary_g(A2.shape[1])
    kernelgmm = KernelLossAGMM(learner, adversary_g, kernel_fn, sigma)
    kernelgmm.fit(A2, A1, Y, learner_l2=learner_l2**2, adversary_l2=adversary_l2,
                  learner_lr=learner_lr, adversary_lr=adversary_lr,
                  n_epochs=_get(opts, 'n_epochs', n_epochs),
                  bs=_get(opts, 'bs', bs),
                  model_dir=str(Path.home()),
                  device=device)
    bridge_fs = torch.Tensor(kernelgmm.predict(A1.to(device),
                            model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0)))
    
    #Second stage
    learner = _get_learner(B1.shape[1])
    adversary_g = _get_adversary_g(B2.shape[1])
    kernelgmm = KernelLossAGMM(learner, adversary_g, kernel_fn, sigma)
    kernelgmm.fit(B2, B1, bridge_fs, learner_l2=learner_l2**2, adversary_l2=adversary_l2,
                  learner_lr=learner_lr, adversary_lr=adversary_lr,
                  n_epochs=_get(opts, 'n_epochs', n_epochs),
                  bs=_get(opts, 'bs', bs),
                  model_dir=str(Path.home()),
                  device=device)
    return kernelgmm.predict(B1_test.to(device),
                             model=_get_model_opt(opts, 'model', 0), burn_in=_get(opts, 'burnin', 0))



def mse(param_estimates, true_params):
    return np.mean((np.array(param_estimates) - np.array(true_params))**2)


def rsquare(param_estimates, true_params):
    return 1 - mse(param_estimates, true_params) / np.var(true_params)


def _key(dic, value):
    return list(iter(dic.keys()))[np.argwhere(np.array(list(iter(dic.values()))) == value)[0, 0]]
