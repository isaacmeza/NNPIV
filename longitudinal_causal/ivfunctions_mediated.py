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


def _get(opts, key, default):
    return opts[key] if (key in opts) else default


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

def nested_npivfit(data, model, fitargs=None):
    W0_test, W, Z, X, M, D, Y = data

    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = np.column_stack((M1,X1,Z1))
    A1 = np.column_stack((M1,X1,W1))

    if fitargs is not None:
        model.fit(A2, A1, Y1, **fitargs)
    else:
        model.fit(A2, A1, Y1)

    A1 = np.column_stack((M,X,W))    
    bridge_fs = model.predict(A1)
    bridge_fs = bridge_fs.reshape(A1.shape[:1] + Y.shape[1:])

    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = np.column_stack((X0,Z0))
    B1 = np.column_stack((X0,W0))

    if fitargs is not None:
        model.fit(B2, B1, bridge_fs, **fitargs)
    else:
        model.fit(B2, B1, bridge_fs)
    
    y_pred = model.predict(W0_test)
    return y_pred.reshape(W0_test.shape[:1] + Y.shape[1:])

def nystromrkhsfit(data, opts):
    """
    data: the data returned by gen_data
    opts: the method_opts from the config file
    """
    alpha_scales = np.geomspace(1, 10000, 10)
    model = ApproxRKHSIVCV(kernel_approx='nystrom', n_components=_get(opts, 'nstrm_n_comp', 100),
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=alpha_scales, cv=5)
    return nested_npivfit(data, model)


def ensembleiv(data, opts):
    model = EnsembleIV(n_iter=200, max_abs_value=2)
    return nested_npivfit(data, model)


def ensemblestariv(data, opts):
    model = EnsembleIVStar(n_iter=200, max_abs_value=2)
    return nested_npivfit(data, model)


def l1sparselinear(data, opts):
    W0_test, W, Z, X, M, D, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = OptimisticHedgeVsOptimisticHedge(B=3, lambda_theta=_get(opts, 'lin_l1', .05),
                                             eta_theta=.1,
                                             eta_w=.1,
                                             n_iter=_get(opts, 'lin_nit', 10000), tol=.0001, sparsity=None)
    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = np.column_stack((M1,X1,Z1))
    A1 = np.column_stack((M1,X1,W1))

    model.fit(trans.fit_transform(A2), trans.fit_transform(A1), Y1)
    
    A1 = np.column_stack((M,X,W))
    bridge_fs = model.predict(trans.fit_transform(A1)).reshape(A1.shape[:1] + Y.shape[1:])

    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = np.column_stack((X0,Z0))
    B1 = np.column_stack((X0,W0))

    model.fit(trans.fit_transform(B2), trans.fit_transform(B1), bridge_fs)

    return model.predict(trans.fit_transform(W0_test)).reshape(W0_test.shape[:1] + Y.shape[1:])


def stochasticl1sparselinear(data, opts):
    W0_test, W, Z, X, M, D, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = StochasticOptimisticHedgeVsOptimisticHedge(B=3, lambda_theta=_get(opts, 'lin_l1', .05),
                                                       eta_theta=.05, eta_w=.05,
                                                       n_iter=_get(opts, 'lin_nit', 20000), tol=0.0001)
    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = np.column_stack((M1,X1,Z1))
    A1 = np.column_stack((M1,X1,W1))

    model.fit(trans.fit_transform(A2), trans.fit_transform(A1), Y, L=100)
    
    A1 = np.column_stack((M,X,W))
    bridge_fs = model.predict(trans.fit_transform(A1)).reshape(A1.shape[:1] + Y.shape[1:])

    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = np.column_stack((X0,Z0))
    B1 = np.column_stack((X0,W0))

    model.fit(trans.fit_transform(B2), trans.fit_transform(B1), bridge_fs, L=100)

    return model.predict(trans.fit_transform(W0_test)).reshape(W0_test.shape[:1] + Y.shape[1:])


def l2sparselinear(data, opts):
    W0_test, W, Z, X, M, D, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = L2OptimisticHedgeVsOGD(B=3, tol=0.0001, lambda_theta=_get(opts, 'lin_l1', .05),
                                   n_iter=_get(opts, 'lin_nit', 20000), eta_theta=.001, eta_w=.001, sparsity=None)
    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = np.column_stack((M1,X1,Z1))
    A1 = np.column_stack((M1,X1,W1))

    model.fit(trans.fit_transform(A2), trans.fit_transform(A1), Y)

    A1 = np.column_stack((M,X,W))
    bridge_fs = model.predict(trans.fit_transform(A1)).reshape(A1.shape[:1] + Y.shape[1:])

    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = np.column_stack((X0,Z0))
    B1 = np.column_stack((X0,W0))

    model.fit(trans.fit_transform(B2), trans.fit_transform(B1), bridge_fs)

    return model.predict(trans.fit_transform(W0_test)).reshape(W0_test.shape[:1] + Y.shape[1:])


def tsls(data, opts):
    W0_test, W, Z, X, M, D, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = np.column_stack((M1,X1,Z1))
    A1 = np.column_stack((M1,X1,W1))

    polyA1 = trans.fit_transform(A1)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', LinearRegression())])
    first.fit(A2, polyA1)
    second = LinearRegression()
    second.fit(first.predict(A2), Y1)
    bridge_fs = second.predict(trans.fit_transform(np.column_stack((M,X,W))))

    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = np.column_stack((X0,Z0))
    B1 = np.column_stack((X0,W0))

    polyB1 = trans.fit_transform(B1)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', LinearRegression())])
    first.fit(B2, polyB1)
    second = LinearRegression()
    second.fit(first.predict(B2), bridge_fs)

    polyB1_test = trans.fit_transform(W0_test)
    return second.predict(polyB1_test)


def regtsls(data, opts):
    W0_test, W, Z, X, M, D, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    #First stage
    ind = np.where(D==1)[0]
    M1 = M[ind]
    W1 = W[ind]
    X1 = X[ind,:]
    Z1 = Z[ind]
    Y1 = Y[ind]

    A2 = np.column_stack((M1,X1,Z1))
    A1 = np.column_stack((M1,X1,W1))

    polyA1 = trans.fit_transform(A1)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', MultiTaskElasticNetCV(cv=3))])
    first.fit(A2, polyA1)
    second = ElasticNetCV(cv=3)
    second.fit(first.predict(A2), Y1.ravel())
    bridge_fs = second.predict(trans.fit_transform(np.column_stack((M,X,W))))

    #Second stage
    ind = np.where(D==0)[0]
    M0 = M[ind]
    W0 = W[ind]
    X0 = X[ind,:]
    Z0 = Z[ind]
    Y0 = Y[ind]
    bridge_fs = bridge_fs[ind]

    B2 = np.column_stack((X0,Z0))
    B1 = np.column_stack((X0,W0))

    polyB1 = trans.fit_transform(B1)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', MultiTaskElasticNetCV(cv=3))])
    first.fit(B2, polyB1)
    second = ElasticNetCV(cv=3)
    second.fit(first.predict(B2), bridge_fs.ravel())

    polyB1_test = trans.fit_transform(W0_test)
    return second.predict(polyB1_test).reshape(W0_test.shape[:1] + Y.shape[1:])


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
