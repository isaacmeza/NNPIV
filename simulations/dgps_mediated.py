# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from scipy.special import expit, logit

# all functions
fn_dict = {'linear': 0, 'pwlinear': 1, 'sigmoid': 2,
           'cubic': 3, 'exponential': 4}
fn_dict_test = {'linear': 0, 'exponential': 4}


def get_tau_fn(func):
    # func describes the relation between response and treatment
    if func == fn_dict['linear']:
        def tau_fn(x): return x
    elif func == fn_dict['pwlinear']:
        def tau_fn(x): return (0.1*x-0.9)*(x < -1) + (x)*(x >= -1)*(x <= 1) + (0.05*x+0.95)*(x > 1)
    elif func == fn_dict['sigmoid']:
        def tau_fn(x): return 2 / (1 + np.exp(-2 * x))
    elif func == fn_dict['cubic']:
        def tau_fn(x): return -np.power(x,3)
    elif func == fn_dict['exponential']:
        def tau_fn(x): return np.exp(x)
    else:
        raise NotImplementedError()

    return tau_fn

def get_tauinv_fn(func):
    # func describes the relation between response and treatment
    if func == fn_dict['linear']:
        def tauinv_fn(x): return x
    elif func == fn_dict['pwlinear']:
        def tauinv_fn(x): return (10*x+9)*(x < -1) + (x)*(x >= -1)*(x <= 1) + (20*x-19)*(x > 1)
    elif func == fn_dict['sigmoid']:
        def tauinv_fn(x): return (np.log(x)-np.log(2-x))/2
    elif func == fn_dict['cubic']:
        def tauinv_fn(x): return np.cbrt(-x)
    elif func == fn_dict['exponential']:
        def tauinv_fn(x): return np.log(x)
    else:
        raise NotImplementedError()

    return tauinv_fn


def get_data(n_samples, tau_fn):
    # Construct dataset
    # W :- negative control outcome
    # Z :- negative control action
    # X :- covariates
    # M :- mediation variable
    # D :- treatment dummy
    # Y :- response (is a scalar always)

    fn = tau_fn
    #1 
    cov = np.array([[0.25,0,0.05],
        [0,0.25,0.05],
        [0.05,0.05,1]])   
    XU = np.random.multivariate_normal([0.25,0.25,0], cov, n_samples)
    X = XU[:,0:2]
    U = XU[:,2:]
    #2
    p = expit(X@np.array([[-0.5],[-0.5]])-0.4*U)
    D = np.random.binomial(1,p)
    #3
    Z = 0.2 - 0.52*D + X@np.array([[0.2],[0.2]]) - U + np.random.normal(0,1,size=(n_samples,1))
    #4
    W = 0.3 + X@np.array([[0.2],[0.2]]) - 0.6*U + np.random.normal(0,1,size=(n_samples,1))
    #5
    M = -0.3*D - X@np.array([[0.5],[0.5]]) + 0.4*U + np.random.normal(0,1,size=(n_samples,1))
    #6
    Y = 2 + 2*D + M + 2*W - X@np.array([[1],[1]]) - U + 2*np.random.normal(0,1,size=(n_samples,1))	
    #7
    W = fn(W)
    M = fn(M)
    Z = fn(Z)
    X = fn(X)

    return W, Z, X, M, D, Y, fn
