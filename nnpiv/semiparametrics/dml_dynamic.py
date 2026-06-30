"""
Debiased Machine Learning for dynamic treatment effects (DML-dynamic).

This module estimates path-specific dynamic treatment means E[Y(d1, d2)] for two sequential binary treatments. 
The implementation follows the orthogonal score used for dynamic treatment effects in Bradic, Ji, and Zhang (2024) and Bodory, Huber, and Laffers (2022), while keeping the same cross-fitting, localization, and confidence interval architecture used by the other DML classes in this package.

Classes:
    DML_dynamic: Main class for performing DML for two-period dynamic treatment effects.

DML_dynamic Methods:
    __init__: Initialize the DML_dynamic instance with data and model configurations.

    _calculate_confidence_interval: Calculate confidence intervals for the estimates.

    _localization: Perform localization using kernel density estimation.

    _npivfit_outcome: Fit the path-specific outcome regression delta.

    _fit_propensity_models: Fit the sequential propensity score models.

    _fit_state_regression: Fit the first-period state regression nu.

    _process_fold: Process a single fold for cross-validation.

    _split_and_estimate: Split the data and estimate the model for each fold.

    dml: Perform Debiased Machine Learning for the dynamic treatment mean.
"""

import copy
import warnings

import numpy as np
from joblib import Parallel, delayed, cpu_count
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.kde import kernel_switch
from tqdm import tqdm
import torch

from nnpiv.rkhs import RKHSIVL2


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
toT = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)


def _get(opts, key, default):
    """
    Retrieve the value associated with 'key' in 'opts', or return 'default' if not present.

    Parameters:
    opts : dict
        Dictionary of options.
    key : str
        Key to look up in 'opts'.
    default : any
        Default value to return if 'key' is not found.

    Returns
    -------
    any
        Value associated with 'key' or 'default'.
    """
    return opts[key] if (opts is not None and key in opts) else default


def _transform_poly(X, opts):
    """
    Transform the input data X using polynomial features.

    Parameters:
    X : array-like
        Input data.
    opts : dict
        Options dictionary containing the polynomial degree ('lin_degree').

    Returns
    -------
    array-like
        Transformed data.
    """
    degree = _get(opts, 'lin_degree', 1)
    if degree == 1:
        return X
    else:
        trans = PolynomialFeatures(degree=degree, include_bias=False)
        return trans.fit_transform(X)


def _fun_threshold_alpha(alpha, g):
    """
    Auxiliary function for computation of optimal alpha for improvement in overlap: CHIM 
    (Dealing with limited overlap in estimation of average treatment effects, Crump et al., Biometrika, 2009).

    Parameters:
    alpha : float
        Alpha value.
    g : array-like
        Input array.

    Returns
    -------
    float
        Result of the threshold function.
    """
    lambda_val = 1 / (alpha * (1 - alpha))
    ind = (g <= lambda_val)
    den = sum(ind)
    num = ind * g
    result = (2 * sum(num) / den - lambda_val) ** 2
    return result


def _as_2d(X):
    """
    Convert an array-like object to a two-dimensional numpy array.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def _as_col(Y):
    """
    Convert an array-like object to a single-column numpy array.
    """
    return np.asarray(Y).reshape(-1, 1)


class DML_dynamic:
    """
    Debiased Machine Learning for two-period dynamic treatment effects.

    The target is the path-specific counterfactual mean E[Y(d1, d2)] for the treatment path d=(d1,d2). 
    For S1=(1, X1')' and S2=(1, X1', X2')', define pi1_d(S1)=P(D1=d1|S1), pi2_d(S2)=P(D2=d2|S2,D1=d1), delta_d(W)=E[Y|S2,D1=d1,D2=d2], and nu_d(W)=E[delta_d(W)|S1,D1=d1]. 
    In the package convention, the remaining nuisance weights are alpha_d(W)=1(D1=d1,D2=d2) / {pi1_d(S1) pi2_d(S2)} and eta_d(W)=1(D1=d1) / pi1_d(S1). 
    The main estimator evaluates the doubly robust score

        nu(W) + alpha(W) * {Y - delta(W)} + eta(W) * {delta(W) - nu(W)}.

    This is the paper's generic bilinear score with h1=nu, h2=delta, h3=alpha, and h4=eta.

    The first-period state regression nu can be fitted either by regressing delta(W) on S1 among D1=d1 observations or by using the sequential doubly-robust learner (S-DRL) pseudo-outcome delta(W)+1(D2=d2)/pi2(S2){Y-delta(W)} among D1=d1 observations.

    Parameters
    ----------
    Y : array-like
        Final outcome variable.
    D1 : array-like
        Binary treatment in period 1.
    D2 : array-like
        Binary treatment in period 2.
    X1 : array-like, optional
        Baseline covariates for period 1.
    X2 : array-like, optional
        Intermediate covariates observed after D1 and before D2.
    V : array-like, optional
        Localization covariates. These are period-1 variables and are appended to X1 when include_V is True.
    v_values : array-like, optional
        Values for localization.
    include_V : bool, optional
        Include localization covariates in the nuisance models.
    ci_type : str, optional
        Type of confidence interval ('pointwise', 'uniform').
    loc_kernel : str, optional
        Kernel for localization. Options include 'gau', 'epa', 'uni', 'tri', etc.
    bw_loc : str, optional
        Bandwidth for localization.
    estimator : str, optional
        Estimator type ('MR', 'OR', 'IPW').
    d1 : int, optional
        First-period treatment value in the target path.
    d2 : int, optional
        Second-period treatment value in the target path.
    treatment_path : tuple, optional
        Alternative way to provide (d1, d2). If supplied, overrides d1 and d2.
    nu_score : str, optional
        Method used to fit nu_d(W): 'regression' or 'S-DRL'.
    model1 : estimator /(list), optional
        Model for the outcome stage. Outcome learners must implement the package NPIV-style interface fit(Z, T, Y) and predict(T). 
        Since the dynamic estimator uses a nested sequential regression, pass a list [delta_model, nu_model] when the two stages use distinct models. If a single model is supplied, it is used for both delta_d(W) and nu_d(W).
    nn_1 : bool /(list), optional
        Use neural network for the outcome stage. For sequential fitting, pass [delta_is_nn, nu_is_nn].
    alpha : float, optional
        Significance level for confidence intervals.
    n_folds : int, optional
        Number of folds for estimation.
    n_rep : int, optional
        Number of repetitions for estimation.
    inner_n_jobs : int, optional
        Number of parallel jobs for inner fold processing. If None, defaults to min(n_folds, available_cores).
    random_seed : int, optional
        Seed for random number generator.
    prop_score : estimator, optional
        Classification model with predict_proba for the sequential propensities.
    CHIM : bool, optional
        Use CHIM method for dealing with limited overlap.
    verbose : bool, optional
        Print progress information.
    fitargs1 : dict /(list), optional
        Arguments for fitting the outcome stage. For sequential fitting, pass [delta_fitargs, nu_fitargs].
    opts : dict, optional
        Additional options.
    """

    def __init__(self, Y, D1, D2, X1=None, X2=None,
                 V=None,
                 v_values=None,
                 include_V=True,
                 ci_type='pointwise',
                 loc_kernel='gau',
                 bw_loc='silverman',
                 estimator='MR',
                 d1=1,
                 d2=1,
                 treatment_path=None,
                 nu_score='regression',
                 model1=RKHSIVL2(kernel='rbf', gamma=.1, delta_scale='auto', delta_exp=.4),
                 nn_1=False,
                 alpha=0.05,
                 n_folds=5,
                 n_rep=1,
                 inner_n_jobs=None,
                 random_seed=123,
                 prop_score=LogisticRegression(),
                 CHIM=False,
                 verbose=True,
                 fitargs1=None,
                 opts=None
                 ):
        self.Y = _as_col(Y)
        self.D1 = _as_col(D1)
        self.D2 = _as_col(D2)
        self.X1 = None if X1 is None else _as_2d(X1)
        self.X2 = None if X2 is None else _as_2d(X2)
        self.V = None if V is None else _as_2d(V)
        self.v_values = v_values
        self.include_V = include_V
        self.ci_type = ci_type
        self.loc_kernel = loc_kernel
        self.bw_loc = bw_loc
        self.estimator = estimator
        self.nu_score = self._normalize_nu_score(nu_score)
        self.alpha = alpha
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.inner_n_jobs = self._resolve_inner_n_jobs(inner_n_jobs)
        self.random_seed = random_seed
        self.CHIM = CHIM
        self.verbose = verbose
        self.opts = opts

        if treatment_path is not None:
            if len(treatment_path) != 2:
                raise ValueError("treatment_path must be a tuple/list of length 2.")
            d1, d2 = treatment_path
        self.d1 = int(d1)
        self.d2 = int(d2)
        self.estimand = f"E[Y({self.d1},{self.d2})]"

        if isinstance(model1, list):
            if len(model1) != 2:
                raise ValueError("Sequential outcome model fitting requires model1=[delta_model, nu_model].")
            self.model1 = copy.deepcopy(model1[0])
            self.model2 = copy.deepcopy(model1[1])
            self.sequential_o = True
            if not isinstance(nn_1, list):
                self.nn_1 = nn_1
                self.nn_2 = nn_1
            else:
                if len(nn_1) != 2:
                    raise ValueError("Sequential outcome model fitting requires nn_1=[delta_is_nn, nu_is_nn].")
                self.nn_1 = nn_1[0]
                self.nn_2 = nn_1[1]
            if not isinstance(fitargs1, list):
                if fitargs1 is not None:
                    warnings.warn("Sequential outcome model fitting received one fitargs1 dictionary. Assuming [fitargs1, fitargs1].", UserWarning)
                self.fitargs1 = fitargs1
                self.fitargs2 = fitargs1
            else:
                if len(fitargs1) != 2:
                    raise ValueError("Sequential outcome model fitting requires fitargs1=[delta_fitargs, nu_fitargs].")
                self.fitargs1 = fitargs1[0]
                self.fitargs2 = fitargs1[1]
        else:
            self.model1 = copy.deepcopy(model1)
            self.model2 = copy.deepcopy(model1)
            self.sequential_o = True
            if isinstance(nn_1, list):
                if len(nn_1) != 2:
                    raise ValueError("Sequential outcome model fitting requires nn_1=[delta_is_nn, nu_is_nn].")
                self.nn_1 = nn_1[0]
                self.nn_2 = nn_1[1]
            else:
                self.nn_1 = nn_1
                self.nn_2 = nn_1
            if isinstance(fitargs1, list):
                if len(fitargs1) != 2:
                    raise ValueError("Sequential outcome model fitting requires fitargs1=[delta_fitargs, nu_fitargs].")
                self.fitargs1 = fitargs1[0]
                self.fitargs2 = fitargs1[1]
            else:
                self.fitargs1 = fitargs1
                self.fitargs2 = fitargs1
        self.prop_score = prop_score

        if self.X1 is None:
            if self.V is not None and self.include_V == True:
                self.S1 = self.V
            else:
                self.S1 = np.ones((self.Y.shape[0], 1))
        else:
            if self.V is not None and self.include_V == True:
                self.S1 = np.column_stack([self.X1, self.V])
            else:
                self.S1 = self.X1

        if self.X2 is None:
            self.S2 = self.S1
        else:
            self.S2 = np.column_stack([self.S1, self.X2])

        lengths = [len(self.Y), len(self.D1), len(self.D2), len(self.S1), len(self.S2)]
        if self.V is not None:
            lengths.append(len(self.V))
        if len(set(lengths)) != 1:
            raise ValueError("All input arrays must have the same length.")

        self._validate_treatment_values()

        if self.estimator not in ['MR', 'OR', 'IPW']:
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['MR', 'OR', 'IPW']. Using MR instead.", UserWarning)
            self.estimator = 'MR'

        if self.ci_type not in ['pointwise', 'uniform']:
            warnings.warn(f"Invalid confidence interval type: {ci_type}. Confidence interval type must be one of ['pointwise', 'uniform']. Using pointwise instead.", UserWarning)
            self.ci_type = 'pointwise'
        if self.ci_type == 'uniform' and (self.v_values is None or self.V is None):
            warnings.warn("Uniform confidence intervals require localization values. Using pointwise instead.", UserWarning)
            self.ci_type = 'pointwise'

        if self.loc_kernel not in list(kernel_switch.keys()):
            warnings.warn(f"Invalid kernel: {loc_kernel}. Kernel must be one of {list(kernel_switch.keys())}. Using gau instead.", UserWarning)
            self.loc_kernel = 'gau'

        if isinstance(self.bw_loc, str):
            if self.bw_loc not in ['silverman', 'scott']:
                warnings.warn(f"Invalid bw rule: {bw_loc}. Bandwidth rule must be one of ['silverman', 'scott'] or provided by the user. Using silverman instead.", UserWarning)
                self.bw_loc = 'silverman'

        if self.V is not None:
            if self.v_values is None:
                warnings.warn("v_values is None. Computing localization around mean(V).", UserWarning)
                self.v_values = np.mean(self.V, axis=0)
            self.v_values = self._format_v_values(self.v_values)
            if self.ci_type == 'uniform' and self.v_values.shape[0] == 1:
                warnings.warn("Uniform confidence intervals are not supported for one localization value. Using pointwise instead.", UserWarning)
                self.ci_type = 'pointwise'

    def _format_v_values(self, v_values):
        """
        Format localization values as rows with the same dimension as V.
        """
        v_values = np.asarray(v_values)
        if v_values.ndim == 0:
            v_values = v_values.reshape(1, 1)
        elif v_values.ndim == 1:
            if self.V.shape[1] == 1:
                v_values = v_values.reshape(-1, 1)
            elif v_values.shape[0] == self.V.shape[1]:
                v_values = v_values.reshape(1, -1)
            else:
                raise ValueError("v_values must have one column per localization variable.")

        if v_values.shape[1] != self.V.shape[1]:
            raise ValueError("v_values must have one column per localization variable.")

        return v_values

    def _normalize_nu_score(self, nu_score):
        """
        Normalize aliases for the first-period state regression score.
        """
        aliases = {
            'regression': 'regression',
            's-drl': 'S-DRL',
            's_drl': 'S-DRL',
            'sdrl': 'S-DRL',
            'S-DRL': 'S-DRL',
        }
        if nu_score in aliases:
            return aliases[nu_score]
        nu_score_l = str(nu_score).lower()
        if nu_score_l in aliases:
            return aliases[nu_score_l]
        warnings.warn(f"Invalid nu_score: {nu_score}. nu_score must be one of ['regression', 'S-DRL']. Using regression instead.", UserWarning)
        return 'regression'

    def _validate_treatment_values(self):
        """
        Validate binary treatment inputs and the requested treatment path.
        """
        if self.d1 not in [0, 1] or self.d2 not in [0, 1]:
            raise ValueError("d1 and d2 must be binary values in {0, 1}.")

        for name, D in [('D1', self.D1), ('D2', self.D2)]:
            vals = np.unique(D[~np.isnan(D)])
            if not np.all(np.isin(vals, [0, 1])):
                raise ValueError(f"{name} must contain only binary values in {{0, 1}}.")

    def _resolve_inner_n_jobs(self, inner_n_jobs):
        if inner_n_jobs is None:
            return max(1, min(int(self.n_folds), int(cpu_count())))

        if isinstance(inner_n_jobs, bool):
            raise ValueError(f"inner_n_jobs must be an integer >= 1, got {inner_n_jobs!r}.")

        try:
            value = int(inner_n_jobs)
        except Exception as exc:
            raise ValueError(f"inner_n_jobs must be an integer >= 1, got {inner_n_jobs!r}.") from exc

        if value < 1:
            raise ValueError(f"inner_n_jobs must be an integer >= 1, got {inner_n_jobs!r}.")

        return min(value, int(self.n_folds))

    def _calculate_confidence_interval(self, theta, theta_var, theta_cov):
        """
        Calculate the confidence interval for the given estimates.

        Parameters:
        theta : array-like
            Estimated values.
        theta_var : array-like
            Variance of the estimates.
        theta_cov : array-like
            Covariance matrix of the estimates.

        Returns
        -------
        array-like
            Lower and upper bounds of the confidence intervals.
        """
        n = self.Y.shape[0]

        if self.ci_type == 'pointwise':
            z_alpha_half = norm.ppf(1 - self.alpha / 2)
            margin_of_error = z_alpha_half * np.sqrt(theta_var / n)
        else:
            S = np.diag(np.diag(theta_cov))
            S_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(S)))

            Sigma_hat = S_inv_sqrt @ theta_cov @ S_inv_sqrt

            # Sample Q from N(0, Sigma_hat)
            Q_samples = np.random.multivariate_normal(np.zeros(theta.shape[0]), Sigma_hat, 5000)

            # Compute the (1 - alpha) quantile of the sampled |Q|_infty
            Q_infinity_norms = np.max(np.abs(Q_samples), axis=1)
            c_alpha = np.quantile(Q_infinity_norms, 1 - self.alpha)
            margin_of_error = c_alpha * np.sqrt(np.diag(theta_cov) / n)

        lower_bound = theta - margin_of_error
        upper_bound = theta + margin_of_error
        return np.column_stack((lower_bound, upper_bound))

    def _localization(self, V, v_val, bw):
        """
        Perform localization using kernel density estimation.

        Parameters:
        V : array-like
            Localization covariates.
        v_val : array-like
            Values for localization.
        bw : float
            Bandwidth for localization.

        Returns
        -------
        array-like
            Weights for localization.
        """
        if kernel_switch[self.loc_kernel]().domain is None:
            def K(x):
                return kernel_switch[self.loc_kernel]()(x)
        else:
            def K(x):
                y = kernel_switch[self.loc_kernel]()(x)*((kernel_switch[self.loc_kernel]().domain[0]<=x) & (x<=kernel_switch[self.loc_kernel]().domain[1]))
                return y

        bw = np.where(np.asarray(bw) == 0, 1.0, bw)
        v = (V-v_val)/bw
        KK = np.prod(list(map(K, v)),axis=1)
        omega = np.mean(KK,axis=0)
        ell = KK/omega
        return ell.reshape(-1,1)

    def _fit_regression(self, model, X, Y, fitargs=None, nn=False):
        """
        Fit an NPIV-style regression model after the optional polynomial transform.
        """
        fitargs = {} if fitargs is None else fitargs
        if nn == True:
            X_t, Y_t = map(toT, [X, Y])
            return model.fit(X_t, X_t, Y_t, **fitargs)

        X = _transform_poly(X, self.opts)
        Y = np.asarray(Y).ravel()
        return model.fit(X, X, Y, **fitargs)

    def _predict_regression(self, model, X, nn=False):
        """
        Predict with an NPIV-style regression model after the optional polynomial transform.
        """
        if nn == True:
            X = toT(X)
            pred = model.predict(X.to(DEVICE), model='avg', burn_in=_get(self.opts, 'burnin', 0))
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
        else:
            X = _transform_poly(X, self.opts)
            pred = model.predict(X)
        return np.asarray(pred).reshape(-1, 1)

    def _fit_classifier(self, model, X, D):
        """
        Fit a binary classifier and give a clear error when a fold lacks support.
        """
        vals = np.unique(D.ravel())
        if vals.shape[0] < 2:
            raise ValueError("A training fold has only one treatment class. Increase n, reduce n_folds, or use a splitter with path support.")
        return model.fit(X, D.ravel())

    def _predict_path_proba(self, model, X, d):
        """
        Predict P(D=d | X) from a fitted classifier.
        """
        if not hasattr(model, 'predict_proba'):
            raise AttributeError("prop_score must implement predict_proba.")

        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        ind = np.where(classes == d)[0]
        if ind.shape[0] == 0:
            raise ValueError(f"Treatment value {d} was not observed when fitting a propensity model.")

        p = proba[:, ind[0]].reshape(-1, 1)
        p = np.where(p == 1, 0.99, p)
        p = np.where(p == 0, 0.01, p)
        return p

    def _npivfit_outcome(self, train_Y, train_D1, train_D2, train_S2):
        """
        Fit the path-specific outcome regression delta_d(W).

        Parameters
        ----------
        train_Y : array-like
            Training outcome variable.
        train_D1 : array-like
            Training first-period treatment.
        train_D2 : array-like
            Training second-period treatment.
        train_S2 : array-like
            Training second-period state variables.

        Returns
        -------
        object
            Fitted model for delta_d(W).
        """
        if self.estimator == 'MR' or self.estimator == 'OR':
            model_1 = copy.deepcopy(self.model1)
            ind = np.where((train_D1.ravel() == self.d1) & (train_D2.ravel() == self.d2))[0]
            if len(ind) == 0:
                raise ValueError("No observations for the requested treatment path in a training fold.")
            return self._fit_regression(model_1, train_S2[ind, :], train_Y[ind],
                                        self.fitargs1, nn=self.nn_1)

        return None

    def _fit_propensity_models(self, train_D1, train_D2, train_S1, train_S2):
        """
        Fit the sequential propensity score models pi1_d(S1) and pi2_d(S2).

        Parameters
        ----------
        train_D1 : array-like
            Training first-period treatment.
        train_D2 : array-like
            Training second-period treatment.
        train_S1 : array-like
            Training first-period state variables.
        train_S2 : array-like
            Training second-period state variables.

        Returns
        -------
        tuple
            Fitted propensity models for period 1 and period 2.
        """
        model_pi1 = copy.deepcopy(self.prop_score)
        model_pi2 = copy.deepcopy(self.prop_score)

        model_pi1 = self._fit_classifier(model_pi1, train_S1, train_D1)

        ind_d1 = np.where(train_D1.ravel() == self.d1)[0]
        if len(ind_d1) == 0:
            raise ValueError("No D1=d1 observations in a training fold for fitting pi2.")
        model_pi2 = self._fit_classifier(model_pi2, train_S2[ind_d1, :], train_D2[ind_d1])

        return model_pi1, model_pi2

    def _propensity_score(self, model_pi1, model_pi2, test_S1, test_S2):
        """
        Estimate path-specific propensity scores and the CHIM threshold.

        Parameters
        ----------
        model_pi1 : estimator
            Fitted first-period propensity model.
        model_pi2 : estimator
            Fitted second-period propensity model.
        test_S1 : array-like
            Testing first-period state variables.
        test_S2 : array-like
            Testing second-period state variables.

        Returns
        -------
        tuple
            Estimated pi1, pi2, and threshold alpha.
        """
        pi1_hat = self._predict_path_proba(model_pi1, test_S1, self.d1)
        pi2_hat = self._predict_path_proba(model_pi2, test_S2, self.d2)

        if self.CHIM == True:
            g_values = [1/(pi1_hat*(1-pi1_hat)), 1/(pi2_hat*(1-pi2_hat))]
            optimized_alphas = []

            for g in g_values:
                def _objective_function(alpha):
                    return _fun_threshold_alpha(alpha, g)
                result = minimize_scalar(_objective_function, bounds=(0.001, 0.499))
                optimized_alphas.append(result.x)
            alfa = max(optimized_alphas)
        else:
            alfa = 0.0

        return pi1_hat, pi2_hat, alfa

    def _fit_state_regression(self, train_Y, train_D1, train_D2, train_S1,
                              train_S2, delta_model, model_pi2=None):
        """
        Fit the first-period state regression nu_d(W).

        Parameters
        ----------
        train_Y : array-like
            Training outcome variable.
        train_D1 : array-like
            Training first-period treatment.
        train_D2 : array-like
            Training second-period treatment.
        train_S1 : array-like
            Training first-period state variables.
        train_S2 : array-like
            Training second-period state variables.
        delta_model : estimator
            Fitted outcome regression for delta_d(W).
        model_pi2 : estimator, optional
            Fitted second-period propensity model, required for S-DRL.

        Returns
        -------
        object
            Fitted model for nu_d(W).
        """
        if self.estimator == 'MR' or self.estimator == 'OR':
            model_2 = copy.deepcopy(self.model2)
            ind_d1 = np.where(train_D1.ravel() == self.d1)[0]
            if len(ind_d1) == 0:
                raise ValueError("No D1=d1 observations in a training fold for fitting nu.")

            delta_hat = self._predict_regression(delta_model, train_S2, nn=self.nn_1)
            if self.nu_score == 'S-DRL':
                if model_pi2 is None:
                    raise ValueError("S-DRL nu_score requires a second-period propensity model.")
                pi2_hat = self._predict_path_proba(model_pi2, train_S2, self.d2)
                ind_d2 = (train_D2 == self.d2).astype(float)
                pseudo_y = delta_hat + ind_d2 / pi2_hat * (train_Y - delta_hat)
            else:
                pseudo_y = delta_hat

            return self._fit_regression(model_2, train_S1[ind_d1, :], pseudo_y[ind_d1],
                                        self.fitargs2, nn=self.nn_2)

        return None

    def _process_fold(self, fold_idx, train_data, test_data):
        """
        Process a single fold for cross-validation.

        Parameters:
        fold_idx : int
            Fold index.
        train_data : tuple
            Training data for the fold.
        test_data : tuple
            Testing data for the fold.

        Returns
        -------
        array-like
            Estimated moment functions for the test data.
        """
        train_Y, test_Y = train_data[0], test_data[0]
        train_D1, test_D1 = train_data[1], test_data[1]
        train_D2, test_D2 = train_data[2], test_data[2]
        train_S1, test_S1 = train_data[3], test_data[3]
        train_S2, test_S2 = train_data[4], test_data[4]
        if self.V is not None:
            train_V, test_V = train_data[5], test_data[5]

        if self.estimator == 'MR' or self.estimator == 'OR':
            delta_model = self._npivfit_outcome(train_Y, train_D1, train_D2, train_S2)
        else:
            delta_model = None

        if self.estimator == 'MR' or self.estimator == 'IPW' or self.nu_score == 'S-DRL':
            model_pi1, model_pi2 = self._fit_propensity_models(train_D1, train_D2, train_S1, train_S2)
        else:
            model_pi1, model_pi2 = None, None

        if self.estimator == 'MR' or self.estimator == 'OR':
            nu_model = self._fit_state_regression(train_Y, train_D1, train_D2, train_S1,
                                                  train_S2, delta_model, model_pi2=model_pi2)
            delta_hat = self._predict_regression(delta_model, test_S2, nn=self.nn_1)
            nu_hat = self._predict_regression(nu_model, test_S1, nn=self.nn_2)

        if self.estimator == 'MR' or self.estimator == 'IPW':
            pi1_hat, pi2_hat, alfa = self._propensity_score(model_pi1, model_pi2,
                                                            test_S1, test_S2)
            mask = np.where((pi1_hat >= alfa) & (pi1_hat <= 1 - alfa) &
                            (pi2_hat >= alfa) & (pi2_hat <= 1 - alfa))[0]

        ind_d1 = (test_D1 == self.d1).astype(float)
        ind_d2 = (test_D2 == self.d2).astype(float)
        ind_path = ind_d1 * ind_d2

        # Calculate the score function depending on the estimator.
        if self.estimator == 'MR':
            alpha_hat = ind_path / (pi1_hat * pi2_hat)
            eta_hat = ind_d1 / pi1_hat
            psi_hat = nu_hat + alpha_hat * (test_Y - delta_hat) + eta_hat * (delta_hat - nu_hat)
        if self.estimator == 'OR':
            psi_hat = nu_hat
        if self.estimator == 'IPW':
            alpha_hat = ind_path / (pi1_hat * pi2_hat)
            psi_hat = alpha_hat * test_Y

        # Localization
        if self.V is not None:
            if isinstance(self.bw_loc, str):
                if self.bw_loc == 'silverman':
                    IQR = np.percentile(train_V, 75, axis=0)-np.percentile(train_V, 25, axis=0)
                    A = np.min([np.std(train_V, axis=0), IQR/1.349], axis=0)
                    n = train_V.shape[0]
                    bw = .9 * A * n ** (-0.2)
                elif self.bw_loc == 'scott':
                    A = np.std(train_V, axis=0)
                    n = train_V.shape[0]
                    bw = 1.059 * A * n ** (-0.2)
            else:
                bw_loc = np.asarray(self.bw_loc)
                if bw_loc.ndim == 0:
                    bw = np.ones((train_V.shape[1]))*float(bw_loc)
                elif len(bw_loc)==1:
                    bw = np.ones((train_V.shape[1]))*bw_loc[0]
                else:
                    if len(bw_loc)==train_V.shape[1]:
                        bw = bw_loc
                    else:
                        warnings.warn("bw_loc has incorrect length. Using first element instead.", UserWarning)
                        bw = np.ones((train_V.shape[1]))*bw_loc[0]

            ell = [self._localization(test_V, v, bw) for v in self.v_values]
            ell = np.column_stack(ell)
            psi_hat = ell * psi_hat

        if self.estimator == 'MR' or self.estimator == 'IPW':
            psi_hat = psi_hat[mask]

        if self.verbose == True:
            self.progress_bar.update(1)

        return psi_hat

    def _split_and_estimate(self):
        """
        Split the data and estimate the model for each fold.

        Returns
        -------
        tuple
            Estimated values, variances, and confidence intervals.
        """
        theta = []
        theta_var = []
        theta_cov = []

        for rep in range(self.n_rep):

            if self.verbose == True:
                print(f"Rep: {rep+1}")
                self.progress_bar = tqdm(total=self.n_folds, position=0)

            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed+rep)
            if self.V is None:
                fold_results = Parallel(n_jobs=self.inner_n_jobs, backend='threading')(
                    delayed(self._process_fold)(
                        fold_idx,
                        (self.Y[train_index], self.D1[train_index], self.D2[train_index],
                         self.S1[train_index], self.S2[train_index]),
                        (self.Y[test_index], self.D1[test_index], self.D2[test_index],
                         self.S1[test_index], self.S2[test_index]))
                    for fold_idx, (train_index, test_index) in enumerate(kf.split(self.Y))
                )
            else:
                fold_results = Parallel(n_jobs=self.inner_n_jobs, backend='threading')(
                    delayed(self._process_fold)(
                        fold_idx,
                        (self.Y[train_index], self.D1[train_index], self.D2[train_index],
                         self.S1[train_index], self.S2[train_index], self.V[train_index]),
                        (self.Y[test_index], self.D1[test_index], self.D2[test_index],
                         self.S1[test_index], self.S2[test_index], self.V[test_index]))
                    for fold_idx, (train_index, test_index) in enumerate(kf.split(self.Y))
                )
            if self.verbose == True:
                self.progress_bar.close()

            psi_hat_array = np.concatenate(fold_results, axis=0)
            theta_rep = np.mean(psi_hat_array, axis=0)
            theta_var_rep = np.var(psi_hat_array, axis=0, ddof=1)
            theta_cov_rep = np.cov(psi_hat_array, rowvar=False)

            theta.append(theta_rep)
            theta_var.append(theta_var_rep)
            theta_cov.append(theta_cov_rep)

        theta_hat = np.mean(np.stack(theta, axis=0), axis=0)
        theta_var_hat = np.mean(np.stack(theta_var, axis=0), axis=0)
        theta_cov_hat = np.mean(np.stack(theta_cov, axis=0), axis=0)

        confidence_interval = self._calculate_confidence_interval(theta_hat, theta_var_hat, theta_cov_hat)

        return theta_hat, theta_var_hat, confidence_interval, theta_cov_hat

    def dml(self):
        """
        Perform Debiased Machine Learning for dynamic treatment effects.

        Returns
        -------
        tuple
            Estimated values, variances, and confidence intervals.
        """
        theta, theta_var, confidence_interval, theta_cov_hat = self._split_and_estimate()
        if self.V is None:
            return theta[0], theta_var[0], confidence_interval[0]
        else:
            return theta, theta_cov_hat, confidence_interval
