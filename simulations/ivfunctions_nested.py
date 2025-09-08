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

from nnpiv.linear import sparse2_l1vsl1, sparse2_ridge_l1vsl1
from nnpiv.rkhs import RKHS2IVL2
from nnpiv.ensemble import Ensemble2IV

from sklearn.pipeline import Pipeline
from mcpy.utils import filesafe


def _get(opts, key, default):
    return opts[key] if (key in opts) else default

def gen_data(opts):
    """
    opts : the dgp_opts from the config file
    """
    tau_fn = dgps.get_tau_fn(opts['fn'])
    if _get(opts, 'sparse', 0) == 1:
        n_a = opts['n_a']
        n_b = n_a
    else:
        n_a = opts['n_a']
        n_b = opts['n_b']
        
    A1, A2, B1, B2, Y, tau_fn = dgps.get_data(opts['n_samples'], n_a,
                                    n_b, tau_fn, opts['dgp_num'])

    if opts['gridtest']:
        B1_test = np.zeros((opts['n_test'], B1.shape[1]))
        B1_test += np.median(B1, axis=0, keepdims=True)
        B1_test[:, 0] = np.linspace(np.percentile(
            B1[:, 0], 5), np.percentile(B1[:, 0], 95), 1000)
    else:
        _, _, B1_test, _, _, _ = dgps.get_data(opts['n_test'], n_a,
                                    n_b, tau_fn, opts['dgp_num'])
        B1_test = B1_test[np.argsort(B1_test[:, 0])]
    expected_te = tau_fn(B1_test)

    # data, true_param
    return (B1_test, A1, A2, B1, B2, Y), expected_te

def nested_npivfit(data, model, fitargs=None):
    B1_test, A1, A2, B1, B2, Y = data

    #First stage
    if fitargs is not None:
        model.fit(A2, A1, Y, **fitargs)
    else:
        model.fit(A2, A1, Y)
    bridge_fs = model.predict(A1)
    bridge_fs = bridge_fs.reshape(A1.shape[:1] + Y.shape[1:])

    #Second stage
    if fitargs is not None:
        model.fit(B2, B1, bridge_fs, **fitargs)
    else:
        model.fit(B2, B1, bridge_fs)
    
    y_pred = model.predict(B1_test)
    return y_pred.reshape(B1_test.shape[:1] + Y.shape[1:])

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


def tviv(data, opts):
    n_iter = 5000
    eta_theta = 2 / np.sqrt(n_iter)
    eta_w = 2 / np.sqrt(n_iter)
    est = ShapeIV(n_iter=n_iter, eta_theta=eta_theta, eta_w=eta_w,
                  y_min=-5, y_max=5, lambda_w=2, monotonic=_get(opts, 'shiv_mon', None))
    return nested_npivfit(data, est)


def lipschitztviv(data, opts):
    n_iter = 5000
    eta_theta = 2 / np.sqrt(n_iter)
    eta_w = 2 / np.sqrt(n_iter)
    est = LipschitzShapeIV(L=_get(opts, 'shiv_L', 2), convexity=None, n_iter=n_iter,
                           eta_theta=eta_theta, eta_w=eta_w,
                           y_min=-5, y_max=5, lambda_w=2,
                           n_projection_subsamples=50, max_projection_iters=20)
    return nested_npivfit(data, est)


def convexiv(data, opts):
    n_iter = 5000
    eta_theta = 2 / np.sqrt(n_iter)
    eta_w = 2 / np.sqrt(n_iter)
    est = LipschitzShapeIV(L=_get(opts, 'shiv_L', 2), convexity='convex', n_iter=n_iter,
                           eta_theta=eta_theta, eta_w=eta_w,
                           y_min=-5, y_max=5, lambda_w=2,
                           n_projection_subsamples=50, max_projection_iters=20)
    return nested_npivfit(data, est)


def l1sparselinear(data, opts):
    B1_test, A1, A2, B1, B2, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = OptimisticHedgeVsOptimisticHedge(B=3, lambda_theta=_get(opts, 'lin_l1', .05),
                                             eta_theta=.1,
                                             eta_w=.1,
                                             n_iter=_get(opts, 'lin_nit', 10000), tol=.0001, sparsity=None)
    #First stage
    model.fit(trans.fit_transform(A2), trans.fit_transform(A1), Y)
    bridge_fs = model.predict(trans.fit_transform(A1)).reshape(A1.shape[:1] + Y.shape[1:])

    #Second stage
    model.fit(trans.fit_transform(B2), trans.fit_transform(B1), bridge_fs)

    return model.predict(trans.fit_transform(B1_test)).reshape(B1_test.shape[:1] + Y.shape[1:])


def stochasticl1sparselinear(data, opts):
    B1_test, A1, A2, B1, B2, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = StochasticOptimisticHedgeVsOptimisticHedge(B=3, lambda_theta=_get(opts, 'lin_l1', .05),
                                                       eta_theta=.05, eta_w=.05,
                                                       n_iter=_get(opts, 'lin_nit', 20000), tol=0.0001)
    #First stage
    model.fit(trans.fit_transform(A2), trans.fit_transform(A1), Y, L=100)
    bridge_fs = model.predict(trans.fit_transform(A1)).reshape(A1.shape[:1] + Y.shape[1:])

    #Second stage
    model.fit(trans.fit_transform(B2), trans.fit_transform(B1), bridge_fs, L=100)

    return model.predict(trans.fit_transform(B1_test)).reshape(B1_test.shape[:1] + Y.shape[1:])


def l2sparselinear(data, opts):
    B1_test, A1, A2, B1, B2, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    model = L2OptimisticHedgeVsOGD(B=3, tol=0.0001, lambda_theta=_get(opts, 'lin_l1', .05),
                                   n_iter=_get(opts, 'lin_nit', 20000), eta_theta=.001, eta_w=.001, sparsity=None)
    #First stage
    model.fit(trans.fit_transform(A2), trans.fit_transform(A1), Y)
    bridge_fs = model.predict(trans.fit_transform(A1)).reshape(A1.shape[:1] + Y.shape[1:])

    #Second stage
    model.fit(trans.fit_transform(B2), trans.fit_transform(B1), bridge_fs)

    return model.predict(trans.fit_transform(B1_test)).reshape(B1_test.shape[:1] + Y.shape[1:])


def tsls(data, opts):
    B1_test, A1, A2, B1, B2, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    #First stage
    polyA1 = trans.fit_transform(A1)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', LinearRegression())])
    first.fit(A2, polyA1)
    second = LinearRegression()
    second.fit(first.predict(A2), Y)
    bridge_fs = second.predict(trans.fit_transform(A1))

    #Second stage
    polyB1 = trans.fit_transform(B1)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', LinearRegression())])
    first.fit(B2, polyB1)
    second = LinearRegression()
    second.fit(first.predict(B2), bridge_fs)

    polyB1_test = trans.fit_transform(B1_test)
    return second.predict(polyB1_test)


def regtsls(data, opts):
    B1_test, A1, A2, B1, B2, Y = data
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    #First stage
    polyA1 = trans.fit_transform(A1)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', MultiTaskElasticNetCV(cv=3))])
    first.fit(A2, polyA1)
    second = ElasticNetCV(cv=3)
    second.fit(first.predict(A2), Y.ravel())
    bridge_fs = second.predict(trans.fit_transform(A1))

    #Second stage
    polyB1 = trans.fit_transform(B1)
    first = Pipeline([('poly', PolynomialFeatures(degree=_get(opts, 'lin_degree', 1))),
                      ('elasticnet', MultiTaskElasticNetCV(cv=3))])
    first.fit(B2, polyB1)
    second = ElasticNetCV(cv=3)
    second.fit(first.predict(B2), bridge_fs.ravel())

    polyB1_test = trans.fit_transform(B1_test)
    return second.predict(polyB1_test).reshape(B1_test.shape[:1] + Y.shape[1:])


def sparse_joint_l1vsl1(data, opts):
    B1_test, A1, A2, B1, B2, Y = data
    
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    B1_test, A1, A2, B1, B2 = map(trans.fit_transform, [B1_test, A1, A2, B1, B2])

    model = sparse2_l1vsl1(mu=_get(opts, 'lin_l1', .05), V1=_get(opts, 'budget', 100), V2=_get(opts, 'budget', 100),
                 eta_alpha='auto', eta_w1='auto', eta_beta='auto', eta_w2='auto',
                 n_iter=_get(opts, 'lin_nit', 10000)*2, tol=.00001/2, sparsity=None, fit_intercept=True)
    
    return model.fit(A1, B1, B2, A2, Y).predict(B1_test).reshape(B1_test.shape[:1] + Y.shape[1:])


def sparse_joint_ridgel1vsl1(data, opts):
    B1_test, A1, A2, B1, B2, Y = data
    
    trans = PolynomialFeatures(degree=_get(
        opts, 'lin_degree', 1), include_bias=False)
    B1_test, A1, A2, B1, B2 = map(trans.fit_transform, [B1_test, A1, A2, B1, B2])

    model = sparse2_ridge_l1vsl1(mu=_get(opts, 'lin_l1', .05), V1=_get(opts, 'budget', 100), V2=_get(opts, 'budget', 100),
                 eta_alpha='auto', eta_w1='auto', eta_beta='auto', eta_w2='auto',
                 n_iter=_get(opts, 'lin_nit', 10000)*2, tol=.00001/2, sparsity=None, fit_intercept=True)
    
    return model.fit(A1, B1, B2, A2, Y).predict(B1_test).reshape(B1_test.shape[:1] + Y.shape[1:])


def rkhs2_ridge(data, opts):
    B1_test, A1, A2, B1, B2, Y = data

    model = RKHS2IVL2(kernel='rbf', gamma=.0013, delta_scale='auto', delta_exp=.4)

    return model.fit(A1, B1, B2, A2, Y).predict(B1_test).reshape(B1_test.shape[:1] + Y.shape[1:])


def rfiv2(data, opts):
    B1_test, A1, A2, B1, B2, Y = data

    adversary = RandomForestRegressor(n_estimators=50, max_depth=2,
                                    bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001)
    learnerg = RandomForestClassifier(n_estimators=10, max_depth=2, criterion='gini',
                                        bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001)
    learnerh = RandomForestClassifier(n_estimators=10, max_depth=2, criterion='gini',
                                        bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001)

    model = Ensemble2IV(n_iter=_get(opts, 'rf_iter', 200)*2, max_abs_value=2, 
                              adversary=adversary, learnerg=learnerg, learnerh=learnerh, n_burn_in=10)

    return model.fit(A1, B1, B2, A2, Y).predict(B1_test).reshape(B1_test.shape[:1] + Y.shape[1:])



def mse(param_estimates, true_params):
    return np.mean((np.array(param_estimates) - np.array(true_params))**2)


def rsquare(param_estimates, true_params):
    return 1 - mse(param_estimates, true_params) / np.var(true_params)


def _key(dic, value):
    return list(iter(dic.keys()))[np.argwhere(np.array(list(iter(dic.values()))) == value)[0, 0]]
