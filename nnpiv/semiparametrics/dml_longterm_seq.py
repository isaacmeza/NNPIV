"""
This module implements the Debiased Machine Learning for long-term causal analysis (DML-longterm) class.
The estimand can be either for a model with a surrogacy assumption (Athey et al., 2020b. [Estimating treatment effects using multiple surrogates: the role of the surrogate score and the surrogate index](https://arxiv.org/abs/1603.09326)) or with a latent unconfounded model (Athey et al., 2020a. [Combining experimental and observational data to estimate treatment effects on long-term outcomes](https://arxiv.org/abs/2006.09676)). 
The semiparametric efficiency is derived in Chen and Ritzwoller (2023. [Semiparametric estimation of long-term treatment effects](https://doi.org/10.1016/j.jeconom.2023.105545)).
The module supports different types of longterm models, cross-validation, kernel density estimation 
for localization, and confidence interval computation with pointwise or uniform guarantees.

Classes:
    DML_longterm: Main class for performing DML for long-term causal analysis with sequential model fitting.

DML_longterm_seq Methods:
    __init__: Initialize the DML_longterm instance with data and model configurations.
    
    _calculate_confidence_interval: Calculate confidence intervals for the estimates.
    
    _localization: Perform localization using kernel density estimation.
    
    _nnpivfit_outcome_latent: Fit the outcome model using nonparametric instrumental variables for the latent unconfounded model sequentially.

    _nnpivfit_outcome_surrogacy: Fit the outcome model using nonparametric instrumental variables for the surrogacy model sequentially.
    
    _propensity_score_latent: Estimate the propensity score for the latent unconfounded model.

    _propensity_score_surrogacy: Estimate the propensity score for the surrogacy model.
    
    _process_fold: Process a single fold for cross-validation.
    
    _split_and_estimate: Split the data and estimate the model for each fold.
    
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

device = torch.cuda.current_device() if torch.cuda.is_available() else None

def _get(opts, key, default):
    """
    Retrieve the value associated with 'key' in 'opts', or return 'default' if not present.

    Parameters
    ----------
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

    Parameters
    ----------
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

    Parameters
    ----------
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

class DML_longterm_seq:
    """
    This class is deprecated and will be removed in a future version.
    
    .. deprecated:: 1.0
        Use `DML_mediated` instead.

    Debiased Machine Learning for long-term causal analysis (DML-longterm) class.

    The estimand can be either for a model with a surrogacy assumption (Athey, S., Chetty, R., Imbens, G., Kang, H., 2020b. Estimating treatment effects using multiple surrogates: the role of the surrogate score and the surrogate index. arXiv preprint arXiv:1603.09326) or with a latent unconfounded model (Athey, S.; Chetty, R.; Imbens, G., Combining experimental and observational data to estimate treatment effects on long-term outcomes. arXiv preprint arXiv:2006.09676 (2020)). 
    The semiparametric efficiency is derived in Jiafeng Chen, David M. Ritzwoller, Semiparametric estimation of long-term treatment effects, Journal of Econometrics, Volume 237, Issue 2, Part A, 2023.

    Parameters
    ----------
    Y : array-like
        Long-term outcome variable.
    D : array-like
        Treatment variable.
    S : array-like
        Surrogate outcome variable.
    G : array-like
        Group indicator (0 for experimental, 1 for observational).
    X1 : array-like, optional
        Additional covariates.
    V : array-like, optional
        Localization covariates.
    v_values : array-like, optional
        Values for localization.
    ci_type : str, optional
        Type of confidence interval ('pointwise', 'uniform').
    loc_kernel : str, optional
        Kernel for localization. Options are ['gau', 'epa', 'uni'].
    bw_loc : str, optional
        Bandwidth for localization.
    estimator : str, optional
        Estimator type ('MR', 'OR', 'hybrid', 'IPW').
    longterm_model : str, optional
        Long-term model type ('latent_unconfounded', 'surrogacy').
    model1 : estimator, optional
        Model for the first stage.
    nn_1 : bool, optional
        Use neural network for the first stage.
    model2 : estimator, optional
        Model for the second stage.
    nn_2 : bool, optional
        Use neural network for the second stage.
    alpha : float, optional
        Significance level for confidence intervals.
    n_folds : int, optional
        Number of folds for estimation.
    n_rep : int, optional
        Number of repetitions for estimation.
    random_seed : int, optional
        Seed for random number generator.
    prop_score : estimator, optional
        Model for propensity score.
    CHIM : bool, optional
        Use CHIM method.
    verbose : bool, optional
        Print progress information.
    fitargs1 : dict, optional
        Arguments for fitting the first stage model.
    fitargs2 : dict, optional
        Arguments for fitting the second stage model.
    opts : dict, optional
        Additional options.
    """
    def __init__(self, Y, D, S, G, X1=None, 
                 V=None, 
                 v_values=None,
                 ci_type='pointwise',
                 loc_kernel='gau',
                 bw_loc='silverman',
                 estimator='MR',
                 longterm_model='surrogacy',
                 model1=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5), 
                 nn_1=False,
                 model2=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5), 
                 nn_2=False,
                 alpha=0.05,
                 n_folds=5,
                 n_rep=1,
                 random_seed=123,
                 prop_score=LogisticRegression(),
                 CHIM=False,
                 verbose=True,
                 fitargs1=None,
                 fitargs2=None,
                 opts=None):        
        warnings.warn(
            "DML_longterm_seq is deprecated and will be removed in a future version. "
            "Use DML_longterm instead.",
            DeprecationWarning,
            stacklevel=2
        )
                
        self.Y = Y
        self.D = D
        self.S = S
        self.G = G
        self.X1 = X1
        self.V = V
        self.v_values = v_values
        self.ci_type = ci_type
        self.loc_kernel = loc_kernel
        self.bw_loc = bw_loc
        self.estimator = estimator
        self.longterm_model = longterm_model
        self.model1 = copy.deepcopy(model1)
        self.model2 = copy.deepcopy(model2)
        self.nn_1 = nn_1
        self.nn_2 = nn_2
        self.prop_score = prop_score
        self.CHIM = CHIM
        self.alpha = alpha
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_seed = random_seed
        self.verbose = verbose
        self.fitargs1 = fitargs1
        self.fitargs2 = fitargs2
        self.opts = opts

        if self.X1 is None:
            if self.V is None:
                self.X = np.ones((self.Y.shape[0], 1))
            else:
                self.X = self.V
        else:
            if self.V is None:
                self.X = self.X1
            else:
                self.X = np.column_stack([self.X1, self.V])

        lengths = [len(Y), len(D), len(S), len(G), len(self.X)]
        if len(set(lengths)) != 1:
            raise ValueError("All input vectors must have the same length.")
        
        if self.estimator not in ['MR', 'OR', 'hybrid', 'IPW']:
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['MR', 'OR', 'hybrid', 'IPW']. Using MR instead.", UserWarning)
            self.estimator = 'MR'

        if longterm_model not in ['latent_unconfounded', 'surrogacy']:
            warnings.warn(f"Invalid long-term model: {longterm_model}. Long-term model must be one of ['latent_unconfounded', 'surrogacy']. Using surrogacy instead.", UserWarning)
            self.longterm_model = 'surrogacy'   

        if longterm_model == 'latent_unconfounded':
            ind = np.where(self.G==1)[0]
            nnan = np.isnan(self.D[ind]).sum()
            if nnan>0:
                warnings.warn(f"{nnan} missing values in treatment variable in the observational sample. Using surrogacy instead.", UserWarning)
                self.longterm_model = 'surrogacy'   

        if self.ci_type not in ['pointwise', 'uniform']:
            warnings.warn(f"Invalid confidence interval type: {ci_type}. Confidence interval type must be one of ['pointwise', 'uniform']. Using pointwise instead.", UserWarning)
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
                warnings.warn(f"v_values is None. Computing localization around mean(V).", UserWarning)
                self.v_values = np.mean(self.V, axis=0)    

    def _calculate_confidence_interval(self, theta, theta_var, theta_cov):
        """
        Calculate the confidence interval for the given estimates.

        Parameters
        ----------
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

        Parameters
        ----------
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

        v = (V-v_val)/bw    
        KK = np.prod(list(map(K, v)),axis=1) 
        omega = np.mean(KK,axis=0)   
        ell = KK/omega
        return ell.reshape(-1,1)

    def _nnpivfit_outcome_latent(self, Y, D, S, X, G):
        """
        Fit the outcome model using the latent unconfounded framework.

        This method is based on the model proposed in Athey, S.; Chetty, R.; Imbens, G., Combining experimental and observational data to estimate treatment effects on long-term outcomes. arXiv preprint arXiv:2006.09676 (2020).

        Parameters
        ----------
        Y : array-like
            Outcome variable.
        D : array-like
            Treatment variable.
        S : array-like
            Surrogate variable.
        X : array-like
            Covariates.
        G : array-like
            Group indicator.

        Returns
        -------
        tuple
            Fitted models for treatment and control groups.
        """
        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':
            model_1_d1 = copy.deepcopy(self.model1)
            model_1_d0 = copy.deepcopy(self.model1)
            model_2_d1 = copy.deepcopy(self.model2)
            model_2_d0 = copy.deepcopy(self.model2)

            # First stage in observational data
            if self.nn_1 == True:
                Y, D, S, X, G = map(lambda x: torch.Tensor(x), [Y, D, S, X, G]) 

            ind = np.where(np.logical_and(G == 1, D == 1))[0]
            S1_1 = S[ind]
            X1_1 = X[ind, :]
            Y1_1 = Y[ind]

            ind = np.where(np.logical_and(G == 1, D == 0))[0]
            S1_0 = S[ind]
            X1_0 = X[ind, :]
            Y1_0 = Y[ind]

            if self.nn_1 == True:
                A1_1 = torch.cat((S1_1, X1_1), 1)
                A1_0 = torch.cat((S1_0, X1_0), 1)
            else:
                A1_1 = _transform_poly(np.column_stack((S1_1, X1_1)), self.opts)
                A1_0 = _transform_poly(np.column_stack((S1_0, X1_0)), self.opts)

            if self.fitargs1 is not None:
                bridge_1_d1 = model_1_d1.fit(A1_1, A1_1, Y1_1, **self.fitargs1)
                bridge_1_d0 = model_1_d0.fit(A1_0, A1_0, Y1_0, **self.fitargs1)
            else:
                bridge_1_d1 = model_1_d1.fit(A1_1, A1_1, Y1_1)
                bridge_1_d0 = model_1_d0.fit(A1_0, A1_0, Y1_0)

            if self.nn_1 == True:
                A1 = torch.cat((S, X), 1)
                bridge_1_d1_hat = torch.Tensor(bridge_1_d1.predict(A1.to(device),
                            model='avg', burn_in=_get(self.opts, 'burnin', 0)))
                bridge_1_d0_hat = torch.Tensor(bridge_1_d0.predict(A1.to(device),
                            model='avg', burn_in=_get(self.opts, 'burnin', 0)))
            else:
                A1 = _transform_poly(np.column_stack((S, X)), self.opts)
                bridge_1_d1_hat = bridge_1_d1.predict(A1)
                bridge_1_d1_hat = bridge_1_d1_hat.reshape(A1.shape[:1] + Y.shape[1:])
                bridge_1_d0_hat = bridge_1_d0.predict(A1)
                bridge_1_d0_hat = bridge_1_d0_hat.reshape(A1.shape[:1] + Y.shape[1:])
        else:
            bridge_1_d1 = None
            bridge_1_d0 = None

        if self.estimator == 'MR' or self.estimator == 'OR':
            # Second stage in experimental data
            if self.nn_1 != self.nn_2:
                if self.nn_2 == False:
                    D, X, G, bridge_1_d1_hat, bridge_1_d0_hat = map(lambda x: x.numpy(), [D, X, G, bridge_1_d1_hat, bridge_1_d0_hat])
                else:
                    D, X, G, bridge_1_d1_hat, bridge_1_d0_hat = map(lambda x: torch.Tensor(x), [D, X, G, bridge_1_d1_hat, bridge_1_d0_hat])

            ind_1 = np.where(np.logical_and(G == 0, D == 1))[0]
            ind_0 = np.where(np.logical_and(G == 0, D == 0))[0]
            X0_1 = X[ind_1, :]
            bridge_1_d1_hat = bridge_1_d1_hat[ind_1]
            X0_0 = X[ind_0, :]
            bridge_1_d0_hat = bridge_1_d0_hat[ind_0]

            if self.nn_2 == True:
                B1_1 = X0_1
                B1_0 = X0_0
            else:            
                B1_1 = _transform_poly(X0_1, self.opts)
                B1_0 = _transform_poly(X0_0, self.opts)

            if self.fitargs2 is not None:
                bridge_2_d1 = model_2_d1.fit(B1_1, B1_1, bridge_1_d1_hat, **self.fitargs2)
                bridge_2_d0 = model_2_d0.fit(B1_0, B1_0, bridge_1_d0_hat, **self.fitargs2)
            else:
                bridge_2_d1 = model_2_d1.fit(B1_1, B1_1, bridge_1_d1_hat)
                bridge_2_d0 = model_2_d0.fit(B1_0, B1_0, bridge_1_d0_hat)

        else:
            bridge_2_d1 = None
            bridge_2_d0 = None
        
        return bridge_1_d1, bridge_1_d0, bridge_2_d1, bridge_2_d0

    def _nnpivfit_outcome_surrogacy(self, Y, D, S, X, G):
        """
        Fit the outcome model using the surrogacy framework.

        This method is based on the model proposed in Athey, S., Chetty, R., Imbens, G., Kang, H., 2020b. Estimating treatment effects using multiple surrogates: the role of the surrogate score and the surrogate index. arXiv preprint arXiv:1603.09326.

        Parameters
        ----------
        Y : array-like
            Outcome variable.
        D : array-like
            Treatment variable.
        S : array-like
            Surrogate variable.
        X : array-like
            Covariates.
        G : array-like
            Group indicator.

        Returns
        -------
        tuple
            Fitted models for the outcome.
        """
        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':
            model_1 = copy.deepcopy(self.model1)
            model_2_d1 = copy.deepcopy(self.model2)
            model_2_d0 = copy.deepcopy(self.model2)

            # First stage in observational data
            if self.nn_1 == True:
                Y, D, S, X, G = map(lambda x: torch.Tensor(x), [Y, D, S, X, G]) 

            ind = np.where(G == 1)[0]
            S1 = S[ind]
            X1 = X[ind, :]
            Y1 = Y[ind]

            if self.nn_1 == True:
                A1 = torch.cat((S1, X1), 1)
            else:
                A1 = _transform_poly(np.column_stack((S1, X1)), self.opts)

            if self.fitargs1 is not None:
                bridge_1 = model_1.fit(A1, A1, Y1, **self.fitargs1)
            else:
                bridge_1 = model_1.fit(A1, A1, Y1)

            if self.nn_1 == True:
                A1 = torch.cat((S, X), 1)
                bridge_1_hat = torch.Tensor(bridge_1.predict(A1.to(device),
                            model='avg', burn_in=_get(self.opts, 'burnin', 0)))
            else:
                A1 = _transform_poly(np.column_stack((S, X)), self.opts)
                bridge_1_hat = bridge_1.predict(A1)
                bridge_1_hat = bridge_1_hat.reshape(A1.shape[:1] + Y.shape[1:])
        else:
            bridge_1 = None

        if self.estimator == 'MR' or self.estimator == 'OR':
            # Second stage in experimental data
            if self.nn_1 != self.nn_2:
                if self.nn_2 == False:
                    D, X, G, bridge_1_hat = map(lambda x: x.numpy(), [D, X, G, bridge_1_hat])
                else:
                    D, X, G, bridge_1_hat = map(lambda x: torch.Tensor(x), [D, X, G, bridge_1_hat])

            ind_1 = np.where(np.logical_and(G == 0, D == 1))[0]
            ind_0 = np.where(np.logical_and(G == 0, D == 0))[0]
            X0_1 = X[ind_1, :]
            bridge_1_hat_1 = bridge_1_hat[ind_1]
            X0_0 = X[ind_0, :]
            bridge_1_hat_0 = bridge_1_hat[ind_0]

            if self.nn_2 == True:
                B1_1 = X0_1
                B1_0 = X0_0
            else:            
                B1_1 = _transform_poly(X0_1, self.opts)
                B1_0 = _transform_poly(X0_0, self.opts)

            if self.fitargs2 is not None:
                bridge_2_d1 = model_2_d1.fit(B1_1, B1_1, bridge_1_hat_1, **self.fitargs2)
                bridge_2_d0 = model_2_d0.fit(B1_0, B1_0, bridge_1_hat_0, **self.fitargs2)
            else:
                bridge_2_d1 = model_2_d1.fit(B1_1, B1_1, bridge_1_hat_1)
                bridge_2_d0 = model_2_d0.fit(B1_0, B1_0, bridge_1_hat_0)

        else:
            bridge_2_d1 = None
            bridge_2_d0 = None
        
        return bridge_1, bridge_2_d1, bridge_2_d0

    def _propensity_score_latent(self, S_train, X_train, D_train, G_train,
                                 S_test, X_test):
        """
        Estimate the propensity scores using the latent unconfounded framework.

        This method is based on the model proposed in Athey, S.; Chetty, R.; Imbens, G., Combining experimental and observational data to estimate treatment effects on long-term outcomes. arXiv preprint arXiv:2006.09676 (2020).

        Parameters
        ----------
        S_train : array-like
            Training surrogate variable.
        X_train : array-like
            Training covariates.
        D_train : array-like
            Training treatment variable.
        G_train : array-like
            Training group indicator.
        S_test : array-like
            Testing surrogate variable.
        X_test : array-like
            Testing covariates.

        Returns
        -------
        tuple
            Estimated propensity scores and threshold alpha.
        """
        model_ps = copy.deepcopy(self.prop_score)
        ind = np.where(G_train == 0)[0]
        X_g0_train = X_train[ind, :]
        D_g0_train = D_train[ind]
        ind = np.where(G_train == 1)[0]
        ind = np.where(D_train == 1)[0]
        S_d1_train = S_train[ind]
        X_d1_train = X_train[ind, :]
        G_d1_train = G_train[ind]
        ind = np.where(D_train == 0)[0]
        S_d0_train = S_train[ind]
        X_d0_train = X_train[ind, :]
        G_d0_train = G_train[ind]

        # Treatment propensity score
        model_ps.fit(X_g0_train, D_g0_train.flatten())
        pr_d1_g0_x = model_ps.predict_proba(X_test)[:, 1]
        
        # Selection propensity score
        model_ps.fit(X_train, G_train.flatten())
        pr_g1_x = model_ps.predict_proba(X_test)[:, 1]

        model_ps.fit(np.column_stack((S_d1_train, X_d1_train)), G_d1_train.flatten())
        pr_g1_d1_sx = model_ps.predict_proba(np.column_stack((S_test, X_test)))[:, 1]

        model_ps.fit(np.column_stack((S_d0_train, X_d0_train)), G_d0_train.flatten())
        pr_g1_d0_sx = model_ps.predict_proba(np.column_stack((S_test, X_test)))[:, 1]

        # Overlap assumption
        pr_d1_g0_x = np.where(pr_d1_g0_x == 1, 0.99, pr_d1_g0_x)
        pr_d1_g0_x = np.where(pr_d1_g0_x == 0, 0.01, pr_d1_g0_x)
        pr_g1_d1_sx = np.where(pr_g1_d1_sx == 1, 0.99, pr_g1_d1_sx)
        pr_g1_d1_sx = np.where(pr_g1_d1_sx == 0, 0.01, pr_g1_d1_sx)
        pr_g1_d0_sx = np.where(pr_g1_d0_sx == 1, 0.99, pr_g1_d0_sx)
        pr_g1_d0_sx = np.where(pr_g1_d0_sx == 0, 0.01, pr_g1_d0_sx)
        pr_g1_x = np.where(pr_g1_x == 1, 0.99, pr_g1_x)
        pr_g1_x = np.where(pr_g1_x == 0, 0.01, pr_g1_x)

        if self.CHIM == True:
            # Dropping observations with extreme values of the propensity score - CHIM (2009)
            # One finds the smallest value of \alpha\in [0,0.5] s.t.
            # $\lambda:=\frac{1}{\alpha(1-\alpha)}$
            # $2\frac{\sum 1(g(X)\leq\lambda)*g(X)}{\sum 1(g(X)\leq\lambda)}-\lambda\geq 0$
            # 
            # Equivalently the first value of alpha (in increasing order) such that the constraint is achieved by equality
            # (as the constraint is a monotone increasing function in alpha)

            g_values = [1 / (pr_d1_g0_x * (1 - pr_d1_g0_x)), 1 / (pr_g1_d1_sx * (1 - pr_g1_d1_sx)), 1 / (pr_g1_d0_sx * (1 - pr_g1_d0_sx)), 1 / (pr_g1_x * (1 - pr_g1_x))]
            optimized_alphas = []

            for g in g_values:
                def _objective_function(alpha):
                    return _fun_threshold_alpha(alpha, g)
                result = minimize_scalar(_objective_function, bounds=(0.001, 0.499))
                optimized_alphas.append(result.x)
            alfa = max(optimized_alphas)
        else:
            alfa = 0.0

        return pr_d1_g0_x.reshape(-1, 1), pr_g1_d1_sx.reshape(-1, 1), pr_g1_d0_sx.reshape(-1, 1), pr_g1_x.reshape(-1, 1), alfa

    def _propensity_score_surrogacy(self, S_train, X_train, D_train, G_train,
                                    S_test, X_test):
        """
        Estimate the propensity scores using the surrogacy framework.

        This method is based on the model proposed in Athey, S., Chetty, R., Imbens, G., Kang, H., 2020b. Estimating treatment effects using multiple surrogates: the role of the surrogate score and the surrogate index. arXiv preprint arXiv:1603.09326.

        Parameters
        ----------
        S_train : array-like
            Training surrogate variable.
        X_train : array-like
            Training covariates.
        D_train : array-like
            Training treatment variable.
        G_train : array-like
            Training group indicator.
        S_test : array-like
            Testing surrogate variable.
        X_test : array-like
            Testing covariates.

        Returns
        -------
        tuple
            Estimated propensity scores and threshold alpha.
        """
        model_ps = copy.deepcopy(self.prop_score)
        SX_train = np.column_stack((S_train, X_train))
        ind = np.where(G_train == 0)[0]
        X0_train = X_train[ind, :]
        D0_train = D_train[ind]
        SX0_train = SX_train[ind, :]

        SX_test = np.column_stack((S_test, X_test))

        # Surrogate score
        model_ps.fit(SX0_train, D0_train.flatten())
        pr_d1_g0_sx = model_ps.predict_proba(SX_test)[:, 1]
        model_ps.fit(X0_train, D0_train.flatten())
        pr_d1_g0_x = model_ps.predict_proba(X_test)[:, 1]

        # Sampling score
        model_ps.fit(SX_train, G_train.flatten())
        pr_g1_sx = model_ps.predict_proba(SX_test)[:, 1]
        model_ps.fit(X_train, G_train.flatten())
        pr_g1_x = model_ps.predict_proba(X_test)[:, 1]

        # Overlap assumption
        pr_d1_g0_sx = np.where(pr_d1_g0_sx == 1, 0.99, pr_d1_g0_sx)
        pr_d1_g0_sx = np.where(pr_d1_g0_sx == 0, 0.01, pr_d1_g0_sx)
        pr_d1_g0_x = np.where(pr_d1_g0_x == 1, 0.99, pr_d1_g0_x)
        pr_d1_g0_x = np.where(pr_d1_g0_x == 0, 0.01, pr_d1_g0_x)
        pr_g1_sx = np.where(pr_g1_sx == 1, 0.99, pr_g1_sx)
        pr_g1_sx = np.where(pr_g1_sx == 0, 0.01, pr_g1_sx)
        pr_g1_x = np.where(pr_g1_x == 1, 0.99, pr_g1_x)
        pr_g1_x = np.where(pr_g1_x == 0, 0.01, pr_g1_x)

        if self.CHIM == True:
            # Dropping observations with extreme values of the propensity score - CHIM (2009)
            # One finds the smallest value of \alpha\in [0,0.5] s.t.
            # $\lambda:=\frac{1}{\alpha(1-\alpha)}$
            # $2\frac{\sum 1(g(X)\leq\lambda)*g(X)}{\sum 1(g(X)\leq\lambda)}-\lambda\geq 0$
            # 
            # Equivalently the first value of alpha (in increasing order) such that the constraint is achieved by equality
            # (as the constraint is a monotone increasing function in alpha)

            g_values = [1 / (pr_d1_g0_sx * (1 - pr_d1_g0_sx)), 1 / (pr_d1_g0_x * (1 - pr_d1_g0_x)), 1 / (pr_g1_sx * (1 - pr_g1_sx)), 1 / (pr_g1_x * (1 - pr_g1_x))]
            optimized_alphas = []

            for g in g_values:
                def _objective_function(alpha):
                    return _fun_threshold_alpha(alpha, g)
                result = minimize_scalar(_objective_function, bounds=(0.001, 0.499))
                optimized_alphas.append(result.x)
            alfa = max(optimized_alphas)
        else:
            alfa = 0.0

        return pr_d1_g0_sx.reshape(-1, 1), pr_d1_g0_x.reshape(-1, 1), pr_g1_sx.reshape(-1, 1), pr_g1_x.reshape(-1, 1), alfa

    def _process_fold(self, fold_idx, train_data, test_data):
        """
        Process each fold in the K-fold cross-validation.

        Parameters
        ----------
        fold_idx : int
            Fold index.
        train_data : tuple
            Training data for the current fold.
        test_data : tuple
            Testing data for the current fold.

        Returns
        -------
        array-like
            Estimated score function for the current fold.
        """
        train_Y, test_Y = train_data[0], test_data[0]
        train_D, test_D = train_data[1], test_data[1]
        train_S, test_S = train_data[2], test_data[2]
        train_X, test_X = train_data[3], test_data[3]
        train_G, test_G = train_data[4], test_data[4]
        if self.V is not None:
            train_V, test_V = train_data[5], test_data[5]

        if self.longterm_model == 'surrogacy':
            delta_0, nu_1, nu_0 = self._nnpivfit_outcome_surrogacy(train_Y, train_D, train_S, train_X, train_G)
            
            # Evaluate the estimated moment functions using test_data
            if self.estimator == 'MR' or self.estimator == 'hybrid':
                if self.nn_1 == True:
                    test_S, test_X = tuple(map(lambda x: torch.Tensor(x), [test_S, test_X]))
                    delta_d0_hat = delta_0.predict(torch.cat((test_S, test_X), 1).to(device),
                                                model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                    delta_d1_hat = delta_d0_hat
                else:
                    delta_d0_hat = delta_0.predict(_transform_poly(np.column_stack((test_S, test_X)), self.opts)).reshape(-1, 1)
                    delta_d1_hat = delta_d0_hat
        else:
            delta_d1, delta_d0, nu_1, nu_0 = self._nnpivfit_outcome_latent(train_Y, train_D, train_S, train_X, train_G)

            # Evaluate the estimated moment functions using test_data
            if self.estimator == 'MR' or self.estimator == 'hybrid':
                if self.nn_1 == True:
                    test_S, test_X = tuple(map(lambda x: torch.Tensor(x), [test_S, test_X]))
                    delta_d1_hat = delta_d1.predict(torch.cat((test_S, test_X), 1).to(device),
                                                model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                    delta_d0_hat = delta_d0.predict(torch.cat((test_S, test_X), 1).to(device),
                                                model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                else:
                    delta_d1_hat = delta_d1.predict(_transform_poly(np.column_stack((test_S, test_X)), self.opts)).reshape(-1, 1)
                    delta_d0_hat = delta_d0.predict(_transform_poly(np.column_stack((test_S, test_X)), self.opts)).reshape(-1, 1)
         
        if self.estimator == 'MR' or self.estimator == 'OR':
            if self.nn_2 == True:
                test_X = torch.Tensor(test_X)
                nu_1_hat = nu_1.predict(test_X.to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                nu_0_hat = nu_0.predict(test_X.to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                nu_1_hat = nu_1.predict(_transform_poly(test_X, self.opts)).reshape(-1, 1)
                nu_0_hat = nu_0.predict(_transform_poly(test_X, self.opts)).reshape(-1, 1)

        if self.estimator == 'MR' or self.estimator == 'hybrid' or self.estimator == 'IPW':
            # Obtain propensity score for action bridges
            if self.longterm_model == 'surrogacy':
                pr_d1_g0_sx, pr_d1_g0_x, pr_g1_sx, pr_g1_x, alfa = self._propensity_score_surrogacy(train_S, train_X, train_D, train_G, 
                                                                  test_S, test_X)
                mask = np.where((pr_d1_g0_sx >= alfa) & (pr_d1_g0_sx <= 1 - alfa) &
                                (pr_d1_g0_x >= alfa) & (pr_d1_g0_x <= 1 - alfa) &
                                (pr_g1_sx >= alfa) & (pr_g1_sx <= 1 - alfa) &
                                (pr_g1_x >= alfa) & (pr_g1_x <= 1 - alfa))[0]
                                
                # IPW to residuals of approximation of first outcome bridge 
                alfa_1_hat = (test_G * pr_d1_g0_sx * (1 - pr_g1_sx)) / (pr_g1_sx * pr_d1_g0_x * (1 - pr_g1_x))
                alfa_0_hat = (test_G * (1 - pr_d1_g0_sx) * (1 - pr_g1_sx)) / (pr_g1_sx * (1 - pr_d1_g0_x) * (1 - pr_g1_x))

                # IPW to residuals of approximation of second outcome bridge
                eta_1_hat = ((1 - test_G) * test_D ) / (pr_d1_g0_x * (1 - pr_g1_x))
                eta_0_hat = ((1 - test_G) * (1 - test_D) ) / ((1 - pr_d1_g0_x) * (1 - pr_g1_x))
            else:
                pr_d1_g0_x, pr_g1_d1_sx, pr_g1_d0_sx, pr_g1_x, alfa = self._propensity_score_latent(train_S, train_X, train_D, train_G,
                                                                    test_S, test_X)
                mask = np.where((pr_d1_g0_x >= alfa) & (pr_d1_g0_x <= 1 - alfa) &
                                (pr_g1_d1_sx >= alfa) & (pr_g1_d1_sx <= 1 - alfa) &
                                (pr_g1_d0_sx >= alfa) & (pr_g1_d0_sx <= 1 - alfa) &
                                (pr_g1_x >= alfa) & (pr_g1_x <= 1 - alfa))[0]

                # IPW to residuals of approximation of first outcome bridge
                alfa_1_hat = (test_G * test_D * (1 - pr_g1_d1_sx)) / (pr_g1_d1_sx * pr_d1_g0_x * (1 - pr_g1_x))
                alfa_0_hat = (test_G * (1 - test_D) * (1 - pr_g1_d0_sx)) / (pr_g1_d0_sx * (1 - pr_d1_g0_x) * (1 - pr_g1_x))

                # IPW to residuals of approximation of second outcome bridge
                eta_1_hat = ((1 - test_G) * test_D ) / (pr_d1_g0_x * (1 - pr_g1_x))
                eta_0_hat = ((1 - test_G) * (1 - test_D) ) / ((1 - pr_d1_g0_x) * (1 - pr_g1_x))
        
        # Calculate the score function depending on the estimator
        if self.estimator == 'MR':
            y1_hat = nu_1_hat + alfa_1_hat * (test_Y - delta_d1_hat) + eta_1_hat * (delta_d1_hat - nu_1_hat)
            y0_hat = nu_0_hat + alfa_0_hat * (test_Y - delta_d0_hat) + eta_0_hat * (delta_d0_hat - nu_0_hat)
            psi_hat = y1_hat - y0_hat
        if self.estimator == 'OR':
            psi_hat = nu_1_hat - nu_0_hat 
        if self.estimator == 'hybrid':
            psi_hat = eta_1_hat * delta_d1_hat - eta_0_hat * delta_d0_hat
        if self.estimator == 'IPW':
            psi_hat = (alfa_1_hat - alfa_0_hat) * test_Y 

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
                if len(self.bw_loc)==1:
                    bw = np.ones((train_V.shape[1]))*self.bw_loc[0]
                else:
                    if len(self.bw_loc)==train_V.shape[1]:
                        bw = self.bw_loc
                    else:
                        warnings.warn(f"bw_loc has incorrect length. Using first element instead.", UserWarning)
                        bw = np.ones((train_V.shape[1]))*self.bw_loc[0]

            ell = [self._localization(test_V, v, bw) for v in self.v_values]
            ell = np.column_stack(ell)

            psi_hat = ell * psi_hat

        if self.estimator == 'MR' or self.estimator == 'hybrid' or self.estimator == 'IPW':
            psi_hat = psi_hat[mask]
            
        # Print progress bar using tqdm
        if self.verbose == True:
            self.progress_bar.update(1)

        return psi_hat

    def _split_and_estimate(self):
        """
        Split the data into K folds and estimate the model.

        Returns
        -------
        tuple
            Estimated treatment effect, variance, and confidence interval.
        """
        theta = []
        theta_var = []
        theta_cov = []

        for rep in range(self.n_rep):
            
            if self.verbose == True:
                print(f"Rep: {rep + 1}")
                self.progress_bar = tqdm(total=self.n_folds, position=0)
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed + rep)
            if self.V is None:
                fold_results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self._process_fold)(
                        fold_idx, 
                        (self.Y[train_index], self.D[train_index], self.S[train_index], self.X[train_index], self.G[train_index]),
                        (self.Y[test_index], self.D[test_index], self.S[test_index], self.X[test_index], self.G[test_index])) 
                        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.Y))
                )
            else:
                fold_results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self._process_fold)(
                        fold_idx, 
                        (self.Y[train_index], self.D[train_index], self.S[train_index], self.X[train_index], self.G[train_index], self.V[train_index]),
                        (self.Y[test_index], self.D[test_index], self.S[test_index], self.X[test_index], self.G[test_index], self.V[test_index])) 
                        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.Y))
                )
            if self.verbose == True:       
                self.progress_bar.close()

            # Calculate the average of psi_hat_array for each rep
            psi_hat_array = np.concatenate(fold_results, axis=0)
            theta_rep = np.mean(psi_hat_array, axis=0)
            theta_var_rep = np.var(psi_hat_array, axis=0, ddof=1)
            theta_cov_rep = np.cov(psi_hat_array, rowvar=False)

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
        Execute the debiased machine learning procedure.

        Returns
        -------
        tuple
            Estimated treatment effect, variance, and confidence interval.
        """
        theta, theta_var, confidence_interval, theta_cov_hat = self._split_and_estimate()
        if self.V is None:
            return theta[0], theta_var[0], confidence_interval[0]
        else:
            return theta, theta_cov_hat, confidence_interval
