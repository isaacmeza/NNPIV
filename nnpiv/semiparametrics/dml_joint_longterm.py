"""
Debiased Machine Learning for long-term causal analysis with a joint estimator (DML-joint-longterm) class.
The estimand can be either for a model with a surrogacy assumption (Athey et al., 2020b. [Estimating treatment effects using multiple surrogates: the role of the surrogate score and the surrogate index](https://arxiv.org/abs/1603.09326)) or with a latent unconfounded model (Athey et al., 2020a. [Combining experimental and observational data to estimate treatment effects on long-term outcomes](https://arxiv.org/abs/2006.09676)). 
The semiparametric efficiency is derived in Chen and Ritzwoller (2023. [Semiparametric estimation of long-term treatment effects](https://doi.org/10.1016/j.jeconom.2023.105545)).
"""

import numpy as np
from scipy.stats import norm 
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.kde import kernel_switch
import warnings

from tqdm import tqdm  # Import tqdm
import copy
import torch
from nnpiv.rkhs import RKHS2IVCV
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


class DML_joint_longterm:
    """
    Debiased Machine Learning for long-term causal analysis (DML-longterm) class with joint model fitting.

    Parameters
    ----------
    Y : array-like
        Outcome variable.
    D : array-like
        Treatment variable.
    S : array-like
        Surrogate variable.
    G : array-like
        Group variable.
    X1 : array-like, optional
        Additional covariates.
    V : array-like, optional
        Localization covariates.
    v_values : array-like, optional
        Values for localization.
    loc_kernel : str, optional
        Kernel for localization. Options are ['gau', 'epa', 'uni'].
    bw_loc : str, optional
        Bandwidth for localization.
    estimator : str, optional
        Estimator type ('MR', 'OR', 'hybrid', 'IPW').
    longterm_model : str, optional
        Model type for long-term analysis ('surrogacy', 'latent_unconfounded').
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
        Use CHIM method for dealing with limited overlap.
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
                 loc_kernel='gau',
                 bw_loc='silverman',
                 estimator='MR',
                 longterm_model='surrogacy',
                 model1=RKHS2IVCV(kernel='rbf', gamma=.1, delta_scale='auto', 
                                  delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5), 
                 nn_1=False,
                 model2=RKHS2IVCV(kernel='rbf', gamma=.1, delta_scale='auto', 
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
                 opts=None
                 ):
        self.Y = Y
        self.D = D
        self.S = S
        self.G = G
        self.X1 = X1
        self.V = V
        self.v_values = v_values
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

    def _calculate_confidence_interval(self, theta, theta_var):
        """
        Calculate the confidence interval for the given estimates.

        Parameters
        ----------
        theta : array-like
            Estimated values.
        theta_var : array-like
            Variance of the estimates.

        Returns
        -------
        array-like
            Lower and upper bounds of the confidence intervals.
        """
        z_alpha_half = norm.ppf(1 - self.alpha / 2)
        n = self.Y.shape[0]
        margin_of_error = z_alpha_half * np.sqrt(theta_var) * np.sqrt(1 / n)
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

    def _nnpivfit_outcome_latent(self, train_Y, train_D, train_S, train_X, train_G,
                                 test_X, test_S):
        """
        Fit the outcome model using nonparametric instrumental variables for the latent unconfounded model.

        Parameters
        ----------
        train_Y : array-like
            Training outcome variable.
        train_D : array-like
            Training treatment variable.
        train_S : array-like
            Training surrogate variable.
        train_X : array-like
            Training covariates.
        train_G : array-like
            Training group variable.
        test_X : array-like
            Testing covariates.
        test_S : array-like
            Testing surrogate variable.

        Returns
        -------
        tuple
            Estimated values for delta_d1_hat, delta_d0_hat, nu_1_hat, nu_0_hat.
        """
        model_1_d1 = copy.deepcopy(self.model1)
        model_1_d0 = copy.deepcopy(self.model1)

        delta_d0_hat = None
        delta_d1_hat = None
        nu_1_hat = None
        nu_0_hat = None

        # Outcome model
        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':
            A_train = np.column_stack((train_S, train_X))
            E_train = np.column_stack((train_S, train_X))
            B_train = train_X
            C_train = train_X
            B_test = test_X
            A_test = np.column_stack((test_S, test_X))

            if self.nn_1==True:
                A_train, E_train, B_train, C_train, B_test, A_test = map(lambda x: torch.Tensor(x), [A_train, E_train, B_train, C_train, B_test, A_test])

            ind = np.where(train_D==1)[0]
            A1_train = A_train[ind,:]
            E1_train = E_train[ind,:]
            B1_train = B_train[ind,:]
            C1_train = C_train[ind,:]
            Y1_train = train_Y[ind]

            ind = np.where(train_D==0)[0]
            A0_train = A_train[ind,:]
            E0_train = E_train[ind,:]
            B0_train = B_train[ind,:]
            C0_train = C_train[ind,:]
            Y0_train = train_Y[ind]

            if self.nn_1==False:
                A1_train = _transform_poly(A1_train,self.opts)
                E1_train = _transform_poly(E1_train,self.opts)
                A0_train = _transform_poly(A0_train,self.opts)
                E0_train = _transform_poly(E0_train,self.opts)
                B1_train = _transform_poly(B1_train,self.opts)
                C1_train = _transform_poly(C1_train,self.opts)
                B0_train = _transform_poly(B0_train,self.opts)
                C0_train = _transform_poly(C0_train,self.opts)
                B_test = _transform_poly(B_test,self.opts)
                A_test = _transform_poly(A_test,self.opts)

            if self.fitargs1 is not None:
                model_1_d1.fit(A1_train, B1_train, C1_train, E1_train, Y1_train, subsetted=True, subset_ind1=train_G, **self.fitargs1)
                model_1_d0.fit(A0_train, B0_train, C0_train, E0_train, Y0_train, subsetted=True, subset_ind1=train_G, **self.fitargs1)
            else:
                model_1_d1.fit(A1_train, B1_train, C1_train, E1_train, Y1_train, subsetted=True, subset_ind1=train_G)
                model_1_d0.fit(A0_train, B0_train, C0_train, E0_train, Y0_train, subsetted=True, subset_ind1=train_G)
                
            if self.nn_1==True:
                nu_1_hat, delta_d1_hat = model_1_d1.predict(B_test.to(device), A_test.to(device), model='avg', burn_in=_get(self.opts, 'burnin', 0))
                nu_1_hat = nu_1_hat.reshape(-1, 1)
                delta_d1_hat = delta_d1_hat.reshape(-1, 1)
                nu_0_hat, delta_d0_hat = model_1_d0.predict(B_test.to(device), A_test.to(device), model='avg', burn_in=_get(self.opts, 'burnin', 0))
                nu_0_hat = nu_0_hat.reshape(-1, 1)
                delta_d0_hat = delta_d0_hat.reshape(-1, 1)
            else:
                nu_1_hat, delta_d1_hat = model_1_d1.predict(B_test, A_test)
                nu_1_hat = nu_1_hat.reshape(-1, 1)
                delta_d1_hat = delta_d1_hat.reshape(-1, 1)
                nu_0_hat, delta_d0_hat = model_1_d0.predict(B_test, A_test)
                nu_0_hat = nu_0_hat.reshape(-1, 1)
                delta_d0_hat = delta_d0_hat.reshape(-1, 1)

        return delta_d1_hat, delta_d0_hat, nu_1_hat, nu_0_hat
    

    def _nnpivfit_outcome_surrogacy(self, train_Y, train_D, train_S, train_X, train_G,
                                    test_X, test_S):
        """
        Fit the outcome model using nonparametric instrumental variables for the surrogacy model.

        Parameters
        ----------
        train_Y : array-like
            Training outcome variable.
        train_D : array-like
            Training treatment variable.
        train_S : array-like
            Training surrogate variable.
        train_X : array-like
            Training covariates.
        train_G : array-like
            Training group variable.
        test_X : array-like
            Testing covariates.
        test_S : array-like
            Testing surrogate variable.

        Returns
        -------
        tuple
            Estimated values for delta_d1_hat, delta_d0_hat, nu_1_hat, nu_0_hat.
        """
        model_1_d1 = copy.deepcopy(self.model1)
        model_1_d0 = copy.deepcopy(self.model1)

        delta_d0_hat = None
        delta_d1_hat = None
        nu_1_hat = None
        nu_0_hat = None

        # Outcome model
        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':
            A_train = np.column_stack((train_S, train_X))
            E_train = np.column_stack((train_S, train_X))
            B_train = train_X
            C_train = train_X
            B_test = test_X
            A_test = np.column_stack((test_S, test_X))

            if self.nn_1==True:
                A_train, E_train, B_train, C_train, B_test, A_test = map(lambda x: torch.Tensor(x), [A_train, E_train, B_train, C_train, B_test, A_test])

            if self.nn_1==False:
                A_train = _transform_poly(A_train,self.opts)
                E_train = _transform_poly(E_train,self.opts)
                B_train = _transform_poly(B_train,self.opts)
                C_train = _transform_poly(C_train,self.opts)
                B_test = _transform_poly(B_test,self.opts)
                A_test = _transform_poly(A_test,self.opts)

            G0_D1 = (1-train_G)*(train_D)
            G0_D0 = (1-train_G)*(1-train_D)
            if self.fitargs1 is not None:
                model_1_d1.fit(A_train, B_train, C_train, E_train, train_Y, subsetted=True, subset_ind1=train_G, subset_ind2=G0_D1, **self.fitargs1)
                model_1_d0.fit(A_train, B_train, C_train, E_train, train_Y, subsetted=True, subset_ind1=train_G, subset_ind2=G0_D0, **self.fitargs1)
            else:
                model_1_d1.fit(A_train, B_train, C_train, E_train, train_Y, subsetted=True, subset_ind1=train_G, subset_ind2=G0_D1)
                model_1_d0.fit(A_train, B_train, C_train, E_train, train_Y, subsetted=True, subset_ind1=train_G, subset_ind2=G0_D0)

            if self.nn_1==True:
                nu_1_hat, delta_d1_hat = model_1_d1.predict(B_test.to(device), A_test.to(device), model='avg', burn_in=_get(self.opts, 'burnin', 0))
                nu_1_hat = nu_1_hat.reshape(-1, 1)
                delta_d1_hat = delta_d1_hat.reshape(-1, 1)
                nu_0_hat, delta_d0_hat = model_1_d0.predict(B_test.to(device), A_test.to(device), model='avg', burn_in=_get(self.opts, 'burnin', 0))
                nu_0_hat = nu_0_hat.reshape(-1, 1)
                delta_d0_hat = delta_d0_hat.reshape(-1, 1)
            else:
                nu_1_hat, delta_d1_hat = model_1_d1.predict(B_test, A_test)
                nu_1_hat = nu_1_hat.reshape(-1, 1)
                delta_d1_hat = delta_d1_hat.reshape(-1, 1)
                nu_0_hat, delta_d0_hat = model_1_d0.predict(B_test, A_test)
                nu_0_hat = nu_0_hat.reshape(-1, 1)
                delta_d0_hat = delta_d0_hat.reshape(-1, 1)

        return delta_d1_hat, delta_d0_hat, nu_1_hat, nu_0_hat


    def _propensity_score_latent(self, S_train, X_train, D_train, G_train,
                           S_test, X_test):
        """
        Estimate the propensity score for the latent unconfounded model.

        Parameters
        ----------
        S_train : array-like
            Training surrogate variable.
        X_train : array-like
            Training covariates.
        D_train : array-like
            Training treatment variable.
        G_train : array-like
            Training group variable.
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
        ind = np.where(G_train==0)[0]
        X_g0_train = X_train[ind,:]
        D_g0_train = D_train[ind]
        ind = np.where(G_train==1)[0]
        ind = np.where(D_train==1)[0]
        S_d1_train = S_train[ind]
        X_d1_train = X_train[ind,:]
        G_d1_train = G_train[ind]
        ind = np.where(D_train==0)[0]
        S_d0_train = S_train[ind]
        X_d0_train = X_train[ind,:]
        G_d0_train = G_train[ind]

        #Treatment propensity score
        model_ps.fit(X_g0_train, D_g0_train.flatten())
        pr_d1_g0_x = model_ps.predict_proba(X_test)[:,1]
        
        #Selection propensity score
        model_ps.fit(X_train, G_train.flatten())
        pr_g1_x = model_ps.predict_proba(X_test)[:,1]

        model_ps.fit(np.column_stack((S_d1_train,X_d1_train)), G_d1_train.flatten())
        pr_g1_d1_sx = model_ps.predict_proba(np.column_stack((S_test,X_test)))[:,1]

        model_ps.fit(np.column_stack((S_d0_train,X_d0_train)), G_d0_train.flatten())
        pr_g1_d0_sx = model_ps.predict_proba(np.column_stack((S_test,X_test)))[:,1]

        # Overlap assumption
        pr_d1_g0_x = np.where(pr_d1_g0_x == 1, 0.99, pr_d1_g0_x)
        pr_d1_g0_x = np.where(pr_d1_g0_x == 0, 0.01, pr_d1_g0_x)
        pr_g1_d1_sx = np.where(pr_g1_d1_sx == 1, 0.99, pr_g1_d1_sx)
        pr_g1_d1_sx = np.where(pr_g1_d1_sx == 0, 0.01, pr_g1_d1_sx)
        pr_g1_d0_sx = np.where(pr_g1_d0_sx == 1, 0.99, pr_g1_d0_sx)
        pr_g1_d0_sx = np.where(pr_g1_d0_sx == 0, 0.01, pr_g1_d0_sx)
        pr_g1_x = np.where(pr_g1_x == 1, 0.99, pr_g1_x)
        pr_g1_x = np.where(pr_g1_x == 0, 0.01, pr_g1_x)

        if self.CHIM==True:
            g_values = [1/(pr_d1_g0_x*(1-pr_d1_g0_x)), 1/(pr_g1_d1_sx*(1-pr_g1_d1_sx)), 1/(pr_g1_d0_sx*(1-pr_g1_d0_sx)), 1/(pr_g1_x*(1-pr_g1_x))]
            optimized_alphas = []

            for g in g_values:
                def _objective_function(alpha):
                    return _fun_threshold_alpha(alpha, g)
                result = minimize_scalar(_objective_function, bounds=(0.001, 0.499))
                optimized_alphas.append(result.x)
            alfa = max(optimized_alphas)
        else:
            alfa = 0.0

        return pr_d1_g0_x.reshape(-1,1), pr_g1_d1_sx.reshape(-1,1), pr_g1_d0_sx.reshape(-1,1), pr_g1_x.reshape(-1,1), alfa


    def _propensity_score_surrogacy(self, S_train, X_train, D_train, G_train,
                           S_test, X_test):
        """
        Estimate the propensity score for the surrogacy model.

        Parameters
        ----------
        S_train : array-like
            Training surrogate variable.
        X_train : array-like
            Training covariates.
        D_train : array-like
            Training treatment variable.
        G_train : array-like
            Training group variable.
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
        SX_train = np.column_stack((S_train,X_train))
        ind = np.where(G_train==0)[0]
        X0_train = X_train[ind,:]
        D0_train = D_train[ind]
        SX0_train = SX_train[ind,:]

        SX_test = np.column_stack((S_test,X_test))

        #Surrogate score
        model_ps.fit(SX0_train, D0_train.flatten())
        pr_d1_g0_sx = model_ps.predict_proba(SX_test)[:,1]
        model_ps.fit(X0_train, D0_train.flatten())
        pr_d1_g0_x = model_ps.predict_proba(X_test)[:,1]

        #Sampling score
        model_ps.fit(SX_train, G_train.flatten())
        pr_g1_sx = model_ps.predict_proba(SX_test)[:,1]
        model_ps.fit(X_train, G_train.flatten())
        pr_g1_x = model_ps.predict_proba(X_test)[:,1]

        # Overlap assumption
        pr_d1_g0_sx = np.where(pr_d1_g0_sx == 1, 0.99, pr_d1_g0_sx)
        pr_d1_g0_sx = np.where(pr_d1_g0_sx == 0, 0.01, pr_d1_g0_sx)
        pr_d1_g0_x = np.where(pr_d1_g0_x == 1, 0.99, pr_d1_g0_x)
        pr_d1_g0_x = np.where(pr_d1_g0_x == 0, 0.01, pr_d1_g0_x)
        pr_g1_sx = np.where(pr_g1_sx == 1, 0.99, pr_g1_sx)
        pr_g1_sx = np.where(pr_g1_sx == 0, 0.01, pr_g1_sx)
        pr_g1_x = np.where(pr_g1_x == 1, 0.99, pr_g1_x)
        pr_g1_x = np.where(pr_g1_x == 0, 0.01, pr_g1_x)

        if self.CHIM==True:
            g_values = [1/(pr_d1_g0_sx*(1-pr_d1_g0_sx)), 1/(pr_d1_g0_x*(1-pr_d1_g0_x)), 1/(pr_g1_sx*(1-pr_g1_sx)), 1/(pr_g1_x*(1-pr_g1_x))]
            optimized_alphas = []

            for g in g_values:
                def _objective_function(alpha):
                    return _fun_threshold_alpha(alpha, g)
                result = minimize_scalar(_objective_function, bounds=(0.001, 0.499))
                optimized_alphas.append(result.x)
            alfa = max(optimized_alphas)
        else:
            alfa = 0.0

        return pr_d1_g0_sx.reshape(-1,1), pr_d1_g0_x.reshape(-1,1), pr_g1_sx.reshape(-1,1), pr_g1_x.reshape(-1,1), alfa


    def _process_fold(self, fold_idx, train_data, test_data):
        """
        Process a single fold for cross-validation.

        Parameters
        ----------
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
        train_D, test_D = train_data[1], test_data[1]
        train_S, test_S = train_data[2], test_data[2]
        train_X, test_X = train_data[3], test_data[3]
        train_G, test_G = train_data[4], test_data[4]
        if self.V is not None:
            train_V, test_V = train_data[5], test_data[5]

        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':
            if self.longterm_model == 'surrogacy':
                delta_d1_hat, delta_d0_hat, nu_1_hat, nu_0_hat = self._nnpivfit_outcome_surrogacy(train_Y, train_D, train_S, train_X, train_G,
                                                                                                test_X, test_S)
            else:
                delta_d1_hat, delta_d0_hat, nu_1_hat, nu_0_hat = self._nnpivfit_outcome_latent(train_Y, train_D, train_S, train_X, train_G,
                                                                            test_X, test_S)

        if self.estimator == 'MR' or self.estimator == 'hybrid' or self.estimator == 'IPW':
            if self.longterm_model == 'surrogacy':
                pr_d1_g0_sx, pr_d1_g0_x, pr_g1_sx, pr_g1_x, alfa = self._propensity_score_surrogacy(train_S, train_X, train_D, train_G, 
                                                                  test_S, test_X)
                mask = np.where((pr_d1_g0_sx >= alfa) & (pr_d1_g0_sx <= 1 - alfa) &
                                (pr_d1_g0_x >= alfa) & (pr_d1_g0_x <= 1 - alfa) &
                                (pr_g1_sx >= alfa) & (pr_g1_sx <= 1 - alfa) &
                                (pr_g1_x >= alfa) & (pr_g1_x <= 1 - alfa))[0]
                                
                alfa_1_hat = (test_G * pr_d1_g0_sx * (1-pr_g1_sx)) / (pr_g1_sx * pr_d1_g0_x * (1-pr_g1_x))
                alfa_0_hat = (test_G * (1-pr_d1_g0_sx) * (1-pr_g1_sx)) / (pr_g1_sx * (1-pr_d1_g0_x) * (1-pr_g1_x))

                eta_1_hat = ((1-test_G) * test_D ) / (pr_d1_g0_x * (1-pr_g1_x))
                eta_0_hat = ((1-test_G) * (1-test_D) ) / ((1-pr_d1_g0_x) * (1-pr_g1_x))
            else:
                pr_d1_g0_x, pr_g1_d1_sx, pr_g1_d0_sx, pr_g1_x, alfa = self._propensity_score_latent(train_S, train_X, train_D, train_G,
                                                                    test_S, test_X)
                mask = np.where((pr_d1_g0_x >= alfa) & (pr_d1_g0_x <= 1 - alfa) &
                                (pr_g1_d1_sx >= alfa) & (pr_g1_d1_sx <= 1 - alfa) &
                                (pr_g1_d0_sx >= alfa) & (pr_g1_d0_sx <= 1 - alfa) &
                                (pr_g1_x >= alfa) & (pr_g1_x <= 1 - alfa))[0]

                alfa_1_hat = (test_G * test_D * (1-pr_g1_d1_sx)) / (pr_g1_d1_sx * pr_d1_g0_x * (1-pr_g1_x))
                alfa_0_hat = (test_G * (1-test_D) * (1-pr_g1_d0_sx)) / (pr_g1_d0_sx * (1-pr_d1_g0_x) * (1-pr_g1_x))

                eta_1_hat = ((1-test_G) * test_D ) / (pr_d1_g0_x * (1-pr_g1_x))
                eta_0_hat = ((1-test_G) * (1-test_D) ) / ((1-pr_d1_g0_x) * (1-pr_g1_x))
        
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

        if self.V is not None:
            if isinstance(self.bw_loc, str):
                if self.bw_loc == 'silverman':
                    IQR = np.percentile(train_V, 75, axis=0)-np.percentile(train_V, 25, axis=0)
                    A = np.min([np.std(train_V, axis=0), IQR/1.349], axis=0)
                    n = train_V.shape[0]
                    bw = .9 * A * n ** (-0.2)
                elif self.bw_loc == 'scott':
                    IQR = np.percentile(train_V, 75, axis=0)-np.percentile(train_V, 25, axis=0)
                    A = np.min([np.std(train_V, axis=0), IQR/1.349], axis=0)
                    n = train_V.shape[0]
                    bw = 1.059 * A * n ** (-0.2)
            else:
                if len(self.bw_loc)==1:
                    bw = [train_V.shape[1]]*self.bw_loc

            ell = [self._localization(test_V, v, bw) for v in self.v_values]
            ell = np.column_stack(ell)

            psi_hat = ell * psi_hat

        if self.estimator == 'MR' or self.estimator == 'hybrid' or self.estimator == 'IPW':
            psi_hat = psi_hat[mask]
            
        if self.verbose==True:
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

        for rep in range(self.n_rep):
            
            if self.verbose==True:
                print(f"Rep: {rep+1}")
                self.progress_bar = tqdm(total=self.n_folds, position=0)
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed+rep)
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
            if self.verbose==True:       
                self.progress_bar.close()

            psi_hat_array = np.concatenate(fold_results, axis=0)
            theta_rep = np.mean(psi_hat_array, axis=0)
            theta_var_rep = np.var(psi_hat_array, axis=0)

            theta.append(theta_rep)
            theta_var.append(theta_var_rep)

        theta_hat = np.mean(np.stack(theta, axis=0), axis=0)
        theta_var_hat = np.mean(np.stack(theta_var, axis=0), axis=0)
        
        confidence_interval = self._calculate_confidence_interval(theta_hat, theta_var_hat)

        return theta_hat, theta_var_hat, confidence_interval
    

    def dml(self):
        """
        Perform Debiased Machine Learning for Nonparametric Instrumental Variables.

        Returns
        -------
        tuple
            Estimated values, variances, and confidence intervals.
        """
        theta, theta_var, confidence_interval = self._split_and_estimate()
        if self.V is None:
            return theta[0], theta_var[0], confidence_interval[0]
        else:
            return theta, theta_var, confidence_interval
