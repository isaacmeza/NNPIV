"""
This module implements Debiased Machine Learning for Nonparametric Instrumental Variables (DML-npiv).
It provides tools for estimating causal effects using a combination of machine learning models and
instrumental variables techniques. The module supports cross-validation, kernel density estimation
for localization, and confidence interval computation with pointwise or uniform guarantees.

Classes:
    DML_npiv: Main class for performing DML-npiv with various configuration options.

DML_npiv Methods:
    __init__: Initialize the DML_npiv instance with data and model configurations.

    _calculate_confidence_interval: Calculate confidence intervals for the estimates.

    _localization: Perform localization using kernel density estimation.

    _npivfit_outcome: Fit the outcome model using nonparametric instrumental variables.

    _propensity_score: Estimate the propensity score.

    _npivfit_action: Fit the action model using nonparametric instrumental variables.

    _process_fold: Process a single fold for cross-validation.

    _split_and_estimate: Split the data and estimate the model using cross-validation.

    dml: Perform Debiased Machine Learning for Nonparametric Instrumental Variables.
"""

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.kde import kernel_switch
import warnings
from tqdm import tqdm
import copy
import torch
from nnpiv.rkhs import ApproxRKHSIVCV
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from ._utils import (
    as_2d,
    as_column,
    canonicalize_localization_inputs,
    localization_loadings,
    prepare_localization,
    summarize_ratio_scores,
)

device = torch.cuda.current_device() if torch.cuda.is_available() else None

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
    Auxiliary function for computation of optimal alpha for improvement in overlap: CHIM (Dealing with limited overlap in estimation of average treatment effects).

    Richard K. Crump, V. Joseph Hotz, Guido W. Imbens, Oscar A. Mitnik
    Biometrika, Volume 96, Issue 1, March 2009.

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

class DML_npiv:
    """
    Debiased Machine Learning for Nonparametric Instrumental Variables (DML-npiv) class.

    Parameters:
        Y (array-like): Outcome variable.
        D (array-like): Treatment variable.
        Z (array-like): Instrumental variable.
        W (array-like): Negative control outcome.
        X1 (array-like or None): Additional covariates.
        V (array-like or None): Localization covariates. Supplying ``V`` changes
            the estimand to a finite-bandwidth kernel-ratio target.
        v_values (array-like or None): Evaluation points for localization.
        include_V (bool): Whether to include localization covariates in the model.
        ci_type (str): Type of confidence interval ('pointwise', 'uniform').
        loc_kernel (str): Kernel for localization. Options include 'gau', 'epa', 'uni', and 'tri'.
        bw_loc (str): Bandwidth for localization.
        estimator (str): Estimator type ('MR', 'OR', 'IPW').
        model1 (estimator): Model for the first stage.
        nn_1 (bool): Whether to use a neural network for the first stage.
        modelq1 (estimator): Model for the second stage.
        nn_q1 (bool): Whether to use a neural network for the second stage.
        alpha (float): Significance level for confidence intervals.
        n_folds (int): Number of folds for estimation.
        n_rep (int): Number of repetitions for estimation.
        random_seed (int): Seed for the random number generator.
        prop_score (estimator): Model for the propensity score.
        CHIM (bool): Whether to drop observations with extreme propensity scores using CHIM (2009).
        verbose (bool): Whether to print progress information.
        fitargs1 (dict or None): Arguments for fitting the first-stage model.
        fitargsq1 (dict or None): Arguments for fitting the second-stage model.
        opts (dict or None): Additional options.
    """

    def __init__(self, Y, D, Z, W, X1=None,
                 V=None,
                 v_values=None,
                 include_V=True,
                 ci_type='pointwise',
                 loc_kernel='gau',
                 bw_loc='silverman',
                 estimator='MR',
                 model1=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5),
                 nn_1=False,
                 modelq1=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5),
                 nn_q1=False,
                 alpha=0.05,
                 n_folds=5,
                 n_rep=1,
                 random_seed=123,
                 prop_score=LogisticRegression(),
                 CHIM=False,
                 verbose=True,
                 fitargs1=None,
                 fitargsq1=None,
                 opts=None
                 ):
        self.Y = as_column(Y, "Y")
        self.D = as_column(D, "D")
        self.Z = as_2d(Z, "Z")
        self.W = as_2d(W, "W")
        self.X1 = None if X1 is None else as_2d(X1, "X1")
        self.V = V
        self.v_values = v_values
        self.include_V = include_V
        self.ci_type = ci_type
        self.loc_kernel = loc_kernel
        self.bw_loc = bw_loc
        self.estimator = estimator
        self.model1 = copy.deepcopy(model1)
        self.modelq1 = copy.deepcopy(modelq1)
        self.nn_1 = nn_1
        self.nn_q1 = nn_q1
        self.prop_score = prop_score
        self.CHIM = CHIM
        self.alpha = alpha
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_seed = random_seed
        self.verbose = verbose
        self.fitargs1 = fitargs1
        self.fitargsq1 = fitargsq1
        self.opts = opts

        self.bw_loc_ = None
        self.loc_normalizers_ = None
        if self.V is not None:
            if self.v_values is None:
                warnings.warn(
                    "v_values is None. Computing localization around mean(V).",
                    UserWarning,
                )
            self.V, self.v_values = canonicalize_localization_inputs(
                self.V, self.v_values
            )

        if self.X1 is None:
            if self.V is not None and self.include_V == True:
                self.X = self.V
            else:
                self.X = np.ones((self.Y.shape[0], 1))
        else:
            if self.V is not None and self.include_V == True:
                self.X = np.column_stack([self.X1, self.V])
            else:
                self.X = self.X1

        lengths = [
            len(self.Y), len(self.D), len(self.Z), len(self.W), len(self.X)
        ]
        if self.V is not None:
            lengths.append(len(self.V))
        if len(set(lengths)) != 1:
            raise ValueError("All input vectors must have the same length.")


        if self.estimator not in ['MR', 'OR', 'IPW']:
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['MR', 'OR', 'IPW']. Using MR instead.", UserWarning)
            self.estimator = 'MR'

        if self.ci_type not in ['pointwise', 'uniform']:
            warnings.warn(f"Invalid confidence interval type: {ci_type}. Confidence interval type must be one of ['pointwise', 'uniform']. Using pointwise instead.", UserWarning)
            self.ci_type = 'pointwise'
        if self.ci_type == 'uniform' and (self.V is None or self.v_values.shape[0] == 1):
            warnings.warn("Uniform confidence intervals require at least two localization values. Using pointwise instead.", UserWarning)
            self.ci_type = 'pointwise'

        if self.loc_kernel not in list(kernel_switch.keys()):
            warnings.warn(f"Invalid kernel: {loc_kernel}. Kernel must be one of {list(kernel_switch.keys())}. Using gau instead.", UserWarning)
            self.loc_kernel = 'gau'

        if isinstance(self.bw_loc, str):
            if self.bw_loc not in ['silverman', 'scott']:
                warnings.warn(f"Invalid bw rule: {bw_loc}. Bandwidth rule must be one of ['silverman', 'scott'] or provided by the user. Using silverman instead.", UserWarning)
                self.bw_loc = 'silverman'

        if self.V is not None:
            self.V, self.v_values, self.bw_loc_, self.loc_normalizers_ = (
                prepare_localization(
                    self.V,
                    self.v_values,
                    self.bw_loc,
                    kernel_switch[self.loc_kernel](),
                )
            )

    def _calculate_confidence_interval(self, theta, theta_var, theta_cov):
        """
        Calculate the confidence interval for the given estimates.

        Parameters:
        theta : array-like
            Estimated values.
        theta_var : array-like
            Estimated variance of the centered influence values.
        theta_cov : array-like
            Estimated covariance of the centered influence values across
            evaluation points.

        Returns
        -------
        array-like
            Lower and upper bounds of the confidence intervals.
        """
        n = self.Y.shape[0]
        theta = np.asarray(theta, dtype=float).reshape(-1)
        theta_var = np.asarray(theta_var, dtype=float).reshape(-1)
        theta_cov = np.atleast_2d(np.asarray(theta_cov, dtype=float))

        if self.ci_type == 'pointwise':
            z_alpha_half = norm.ppf(1 - self.alpha / 2)
            margin_of_error = z_alpha_half * np.sqrt(
                np.maximum(theta_var, 0.0) / n
            )
        else:
            if theta_cov.shape != (theta.size, theta.size):
                raise ValueError(
                    "theta_cov must have one row and column per target."
                )
            if not np.all(np.isfinite(theta_cov)):
                raise ValueError("theta_cov must contain only finite values.")

            variances = np.maximum(np.diag(theta_cov), 0.0)
            standard_deviations = np.sqrt(variances)
            tolerance = np.finfo(float).eps * max(
                1.0, float(np.max(standard_deviations, initial=0.0))
            )
            active = standard_deviations > tolerance
            margin_of_error = np.zeros(theta.size, dtype=float)

            if np.count_nonzero(active) == 1:
                c_alpha = norm.ppf(1 - self.alpha / 2)
                margin_of_error[active] = (
                    c_alpha * standard_deviations[active] / np.sqrt(n)
                )
            elif np.any(active):
                active_covariance = theta_cov[np.ix_(active, active)]
                active_sd = standard_deviations[active]
                correlation = active_covariance / np.outer(active_sd, active_sd)
                correlation = (correlation + correlation.T) / 2
                np.fill_diagonal(correlation, 1.0)

                # Remove negligible negative eigenvalues caused by numerical
                # covariance error, then renormalize to a correlation matrix.
                eigenvalues, eigenvectors = np.linalg.eigh(correlation)
                correlation = (
                    eigenvectors
                    @ np.diag(np.maximum(eigenvalues, 0.0))
                    @ eigenvectors.T
                )
                projected_sd = np.sqrt(np.maximum(np.diag(correlation), 0.0))
                correlation = correlation / np.outer(projected_sd, projected_sd)
                correlation = (correlation + correlation.T) / 2
                np.fill_diagonal(correlation, 1.0)

                rng = np.random.default_rng(self.random_seed)
                samples = rng.multivariate_normal(
                    np.zeros(active_sd.size),
                    correlation,
                    size=5000,
                    check_valid="ignore",
                )
                c_alpha = np.quantile(
                    np.max(np.abs(samples), axis=1), 1 - self.alpha
                )
                margin_of_error[active] = c_alpha * active_sd / np.sqrt(n)

        lower_bound = theta - margin_of_error
        upper_bound = theta + margin_of_error
        return np.column_stack((lower_bound, upper_bound))

    def _localization(self, V, v_val=None, bw=None):
        """
        Compute localization loadings using the run-level specification.

        Parameters:
        V : array-like
            Localization covariates.
        v_val, bw : array-like, optional
            A legacy one-off evaluation value and bandwidth. When omitted,
            use the fixed run-level grid, bandwidth, and normalizers.

        Returns
        -------
        array-like
            Kernel loadings with shape ``(n_samples, n_targets)`` using the
            bandwidth and normalizers fixed when the estimator was created.
        """
        kernel = kernel_switch[self.loc_kernel]()
        if v_val is None and bw is None:
            return localization_loadings(
                V, self.v_values, self.bw_loc_, kernel, self.loc_normalizers_
            )
        if v_val is None or bw is None:
            raise ValueError("v_val and bw must be supplied together.")
        V, values, bandwidth, normalizers = prepare_localization(
            V, v_val, bw, kernel
        )
        return localization_loadings(
            V, values, bandwidth, kernel, normalizers
        )

    def _npivfit_outcome(self, Y, D, X, Z):
        """
        Fit the outcome model using nonparametric instrumental variables.

        Parameters:
        Y : array-like
            Outcome variable.
        D : array-like
            Treatment variable.
        X : array-like
            Covariates.
        Z : array-like
            Instrumental variable.

        Returns
        -------
        tuple
            Fitted models for treatment and control groups.
        """
        bridge_ = [None]*2

        if self.estimator == 'MR' or self.estimator == 'OR':
            model_1 = copy.deepcopy(self.model1)

            # First stage
            if self.nn_1==True:
                Y, X, Z = tuple(map(lambda x: torch.Tensor(x), [Y, X, Z]))

            if self.nn_1==False:
                X = _transform_poly(X,self.opts)
                Z = _transform_poly(Z,self.opts)

            for d in [0,1]:
                ind = np.where(D==d)[0]
                Y1 = Y[ind]
                X1 = X[ind,:]
                Z1 = Z[ind]

                if self.fitargs1 is not None:
                    bridge_[d] = copy.deepcopy(model_1).fit(Z1, X1, Y1, **self.fitargs1)
                else:
                    bridge_[d] = copy.deepcopy(model_1).fit(Z1, X1, Y1)

        return bridge_[1], bridge_[0]


    def _propensity_score(self, X, W, D):
        """
        Estimate the propensity score.

        Parameters:
        X : array-like
            Covariates.
        W : array-like
            Control variable.
        D : array-like
            Treatment variable.

        Returns
        -------
        tuple
            Estimated propensity scores and threshold alpha.
        """
        model_ps = copy.deepcopy(self.prop_score)
        X1 = np.column_stack((X,W))

        # First stage
        model_ps.fit(X1, D.flatten())
        ps_hat_1 = model_ps.predict_proba(X1)[:,1]

        # Overlap assumption
        ps_hat_1 = np.where(ps_hat_1 == 1, 0.99, ps_hat_1)
        ps_hat_1 = np.where(ps_hat_1 == 0, 0.01, ps_hat_1)

        if self.CHIM==True:
            # Dropping observations with extreme values of the propensity score - CHIM (2009)
            # One finds the smallest value of \alpha\in [0,0.5] s.t.
            # $\lambda:=\frac{1}{\alpha(1-\alpha)}$
            # $2\frac{\sum 1(g(X)\leq\lambda)*g(X)}{\sum 1(g(X)\leq\lambda)}-\lambda\geq 0$
            #
            # Equivalently the first value of alpha (in increasing order) such that the constraint is achieved by equality
            # (as the constraint is a monotone increasing function in alpha)

            g_values = [1/(ps_hat_1*(1-ps_hat_1))]
            optimized_alphas = []

            for g in g_values:
                def _objective_function(alpha):
                    return _fun_threshold_alpha(alpha, g)
                result = minimize_scalar(_objective_function, bounds=(0.001, 0.499))
                optimized_alphas.append(result.x)
            alfa = max(optimized_alphas)
        else:
            alfa = 0.0

        return ps_hat_1.reshape(-1,1), alfa


    def _npivfit_action(self, ps_hat_1, W, X, Z, alfa=0.0):
        """
        Fit the action model using nonparametric instrumental variables.

        Parameters:
        ps_hat_1 : array-like
            Estimated propensity scores.
        W : array-like
            Control variable.
        X : array-like
            Covariates.
        Z : array-like
            Instrumental variable.
        alfa : float, optional
            Threshold alpha for propensity scores.

        Returns
        -------
        tuple
            Fitted models for treated and control groups.
        """
        bridge_ = [None]*2

        if self.estimator == 'MR' or self.estimator == 'IPW':
            mask = np.where((ps_hat_1 >= alfa) & (ps_hat_1 <= 1 - alfa))[0]
            ps_hat_1 = ps_hat_1[mask]
            ps_hat_0 = 1 - ps_hat_1

            W = W[mask]
            X = X[mask,:]
            Z = Z[mask]

            model_q1 = copy.deepcopy(self.modelq1)

            # First stage
            if self.nn_q1==True:
                ps_hat_1, ps_hat_0, W, X, Z = tuple(map(lambda x: torch.Tensor(x), [ps_hat_1, ps_hat_0, W, X, Z]))

            if self.nn_q1==True:
                A2 = torch.cat((X,W),1)
                A1 = torch.cat((X,Z),1)
            else:
                A2 = _transform_poly(np.column_stack((X,W)),self.opts)
                A1 = _transform_poly(np.column_stack((X,Z)),self.opts)

            if self.fitargsq1 is not None:
                bridge_[0] = copy.deepcopy(model_q1).fit(A2, A1, 1/ps_hat_0, **self.fitargsq1)
                bridge_[1] = copy.deepcopy(model_q1).fit(A2, A1, 1/ps_hat_1, **self.fitargsq1)
            else:
                bridge_[0] = copy.deepcopy(model_q1).fit(A2, A1, 1/ps_hat_0)
                bridge_[1] = copy.deepcopy(model_q1).fit(A2, A1, 1/ps_hat_1)

        return bridge_[1], bridge_[0]


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
        tuple
            Uncentered moment functions and the corresponding target loading
            for the test data.
        """
        train_Y, test_Y = train_data[0], test_data[0]
        train_D, test_D = train_data[1], test_data[1]
        train_W, test_W = train_data[2], test_data[2]
        train_X, test_X = train_data[3], test_data[3]
        train_Z, test_Z = train_data[4], test_data[4]

        if self.V is not None:
            train_V, test_V = train_data[5], test_data[5]

        if self.estimator == 'MR' or self.estimator == 'OR':
            gamma_1, gamma_0 = self._npivfit_outcome(train_Y, train_D, train_X, train_Z)

        if self.estimator == 'MR'  or self.estimator == 'IPW':
            ps_hat_1, alfa = self._propensity_score(train_X, train_W, train_D)
            q_1, q_0 = self._npivfit_action(ps_hat_1, train_W, train_X, train_Z, alfa=alfa)

        # Evaluate the estimated moment functions using test_data
        if self.estimator == 'MR' or self.estimator == 'OR':
            if self.nn_1 == True:
                test_X = torch.Tensor(test_X)
                gamma_1_hat = gamma_1.predict(test_X.to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                gamma_0_hat = gamma_0.predict(test_X.to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                gamma_1_hat = gamma_1.predict(_transform_poly(test_X, opts=self.opts)).reshape(-1, 1)
                gamma_0_hat = gamma_0.predict(_transform_poly(test_X, opts=self.opts)).reshape(-1, 1)

        if self.estimator == 'MR' or self.estimator == 'IPW':
            if self.nn_q1 == True:
                test_X, test_Z = tuple(map(lambda x: torch.Tensor(x), [test_X, test_Z]))
                q_1_hat = q_1.predict(torch.cat((test_X, test_Z), 1).to(device),
                                    model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                q_0_hat = q_0.predict(torch.cat((test_X, test_Z), 1).to(device),
                                    model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                q_1_hat = q_1.predict(_transform_poly(np.column_stack((test_X, test_Z)), opts=self.opts)).reshape(-1, 1)
                q_0_hat = q_0.predict(_transform_poly(np.column_stack((test_X, test_Z)), opts=self.opts)).reshape(-1, 1)

        # Calculate the score function depending on the estimator
        if self.estimator == 'MR':
            psi_hat = (gamma_1_hat-gamma_0_hat +
                    test_D * q_1_hat * (test_Y - gamma_1_hat) - (1-test_D) * q_0_hat * (test_Y - gamma_0_hat))
        if self.estimator == 'OR':
            psi_hat = gamma_1_hat-gamma_0_hat
        if self.estimator == 'IPW':
            psi_hat = test_D * q_1_hat * test_Y - (1 - test_D) * q_0_hat * test_Y

        # The loading multiplying theta is one for an average target and ell
        # for a data-normalized kernel-ratio target.
        theta_loading = np.ones((psi_hat.shape[0], 1), dtype=float)

        # Localization
        if self.V is not None:
            ell = self._localization(test_V)
            psi_hat = ell * psi_hat
            theta_loading = ell * theta_loading

        # Print progress bar using tqdm
        if self.verbose==True:
            self.progress_bar.update(1)

        return psi_hat, theta_loading


    def _split_and_estimate(self):
        """
        Split the data and estimate the model for each fold.

        Returns
        -------
        tuple
            Without ``V``, returns the estimate, influence-value variance, and
            confidence interval. With ``V``, returns estimates, the
            influence-value covariance matrix across evaluation points, and
            confidence intervals. Variance and covariance are not divided by
            the sample size.
        """
        theta = []
        theta_var = []
        theta_cov = []

        for rep in range(self.n_rep):

            if self.verbose==True:
                print(f"Rep: {rep+1}")
                self.progress_bar = tqdm(total=self.n_folds, position=0)

            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed+rep)
            if self.V is None:
                fold_results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self._process_fold)(
                        fold_idx,
                        (self.Y[train_index], self.D[train_index], self.W[train_index],
                        self.X[train_index], self.Z[train_index]),
                        (self.Y[test_index], self.D[test_index], self.W[test_index],
                        self.X[test_index], self.Z[test_index]))
                        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.Y))
                )
            else:
                fold_results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self._process_fold)(
                        fold_idx,
                        (self.Y[train_index], self.D[train_index], self.W[train_index],
                        self.X[train_index], self.Z[train_index], self.V[train_index]),
                        (self.Y[test_index], self.D[test_index], self.W[test_index],
                        self.X[test_index], self.Z[test_index], self.V[test_index]))
                        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.Y))
                )
            if self.verbose==True:
                self.progress_bar.close()

            # Calculate the average of psi_hat_array for each rep
            psi_hat_array = np.concatenate([result[0] for result in fold_results], axis=0)
            theta_loading_array = np.concatenate([result[1] for result in fold_results], axis=0)
            theta_rep, theta_var_rep, theta_cov_rep = summarize_ratio_scores(
                psi_hat_array, theta_loading_array
            )

            # Store results for each rep
            theta.append(theta_rep)
            theta_var.append(theta_var_rep)
            theta_cov.append(theta_cov_rep)

        # Calculate the overall average of theta and theta_var
        theta_hat = np.mean(np.stack(theta, axis=0), axis=0)
        theta_var_hat = np.mean(np.stack(theta_var, axis=0), axis=0)
        theta_cov_hat = np.mean(np.stack(theta_cov, axis=0), axis=0)

        # Calculate the confidence interval
        confidence_interval = self._calculate_confidence_interval(theta_hat, theta_var_hat, theta_cov_hat)

        return theta_hat, theta_var_hat, confidence_interval, theta_cov_hat


    def dml(self):
        """
        Perform Debiased Machine Learning for Nonparametric Instrumental Variables.

        Returns
        -------
        tuple
            Without ``V``, returns the estimate, influence-value variance, and
            confidence interval. With ``V``, returns estimates, the
            influence-value covariance matrix across evaluation points, and
            confidence intervals. Variance and covariance are not divided by
            the sample size.
        """
        theta, theta_var, confidence_interval, theta_cov_hat = self._split_and_estimate()
        if self.V is None:
            return theta[0], theta_var[0], confidence_interval[0]
        else:
            return theta, theta_cov_hat, confidence_interval
