"""
This module performs Debiased Machine Learning for mediation analysis, using joint or sequential estimation for longitudinal
nonparametric parameters (in the Nested NPIV framework). It provides tools for estimating causal effects with 
mediation using a combination of machine learning models and instrumental variables techniques. The module supports different types of mediated estimands, cross-validation, kernel density estimation 
for localization, and confidence interval computation with pointwise or uniform guarantees.

Classes:
    DML_mediated: Main class for performing DML for mediation analysis with joint/sequential model fitting.

DML_mediated Methods:
    __init__: Initialize the DML_mediated instance with data and model configurations.
    
    _calculate_confidence_interval: Calculate confidence intervals for the estimates.
    
    _localization: Perform localization using kernel density estimation.
    
    _npivfit_outcome: Fit the outcome model using nonparametric instrumental variables.

    _nnpivfit_outcome_m: Fit the mediated outcome model sequentially using nonparametric instrumental variables.

    _propensity_score: Estimate the propensity score.
    
    _npivfit_action: Fit the action model using nonparametric instrumental variables.

    _nnpivfit_action_m: Fit the mediated action model sequentially using nonparametric instrumental variables.

    _scores_mediated: Calculate the scores for the mediated effects.
    
    _scores_Y1: Calculate the scores for the Y1 estimand.
    
    _process_fold: Process a single fold for cross-validation.
    
    _split_and_estimate: Split the data and estimate the model for each fold.
    
    dml: Perform Debiased Machine Learning for Nonparametric Instrumental Variables.
"""

import numpy as np
from scipy.stats import norm 
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.kde import kernel_switch
import warnings
from tqdm import tqdm  # Import tqdm
import copy
import torch
from nnpiv.rkhs import RKHS2IVCV, ApproxRKHSIVCV, RKHS2IVL2
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
toT = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
def _to_np(a):
    """
    Convert a tensor to a numpy array if it is a tensor.
    """
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a


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


class DML_mediated:
    """
    Debiased Machine Learning for mediation analysis (DML-mediation) class with joint/sequential model fitting.

    Parameters
    ----------
    Y : array-like
        Outcome variable.
    D : array-like
        Treatment variable.
    M : array-like
        Mediator variable.
    W : array-like
        Negative control outcome.
    Z : array-like
        Instrumental variable.
    X1 : array-like, optional
        Additional covariates.
    V : array-like, optional
        Localization covariates.
    v_values : array-like, optional
        Values for localization.
    include_V : bool, optional
        Include localization covariates in the model.        
    ci_type : str, optional
        Type of confidence interval ('pointwise', 'uniform').
    loc_kernel : str, optional
        Kernel for localization. Options include 'gau', 'epa', 'uni', 'tri', etc.
    bw_loc : str, optional
        Bandwidth for localization.
    estimator : str, optional
        Estimator type ('MR', 'OR', 'hybrid', 'IPW').
    estimand : str, optional
        Type of estimand ('ATE', 'Indirect', 'Direct', 'E[Y1]', 'E[Y0]', 'E[Y(1,M(0))]').
    model1 : estimator /(list), optional
        Model for the outcome stage - Can be a joint or sequential estimator; if the latter a list must be given
    nn_1 : bool /(list), optional
        Use neural network for the outcome stage.
    modelq1 : estimator /(list), optional
        Model for the q1 stage - Can be a joint or sequential estimator; if the latter a list must be given
    nn_q1 : bool /(list), optional
        Use neural network for the q1 stage.
    model_y : estimator, optional
        Model for the outcome - for use with 'E[Y1]', 'E[Y0]', 'Direct', 'Indirect', and 'ATE' estimands.
    nn_y : bool, optional
        Use neural network for the outcome model.
    model_a : estimator, optional
        Model for the action - for use with 'E[Y1]', 'E[Y0]', 'Direct', 'Indirect', and 'ATE' estimands.
    nn_a : bool, optional
        Use neural network for the action model.          
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
        Use CHIM method:
        Dropping observations with extreme values of the propensity score - CHIM (2009)
    verbose : bool, optional
        Print progress information.
    fitargs1 : dict, optional
        Arguments for fitting the outcome stage model.
    fitargsq1 : dict, optional
        Arguments for fitting the q1 stage model.
    fitargsy : dict, optional
        Arguments for fitting the one stage outcome model.
    fitargsa : dict, optional
        Arguments for fitting the one stage action model.        
    opts : dict, optional
        Additional options.
    """
    def __init__(self, Y, D, M, W, Z, X1=None,
                 V=None, 
                 v_values=None,
                 include_V=True,
                 ci_type='pointwise',
                 loc_kernel='gau',
                 bw_loc='silverman',
                 estimator='MR',
                 estimand='ATE',
                 model1=RKHS2IVL2(kernel='rbf', gamma=.0013, delta_scale='auto', delta_exp=10),
                 nn_1=False,
                 modelq1=RKHS2IVL2(kernel='rbf', gamma=.0013, delta_scale='auto', delta_exp=10),
                 nn_q1=False,
                 model_y=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                            kernel='rbf', gamma=.1, delta_scale='auto',
                            delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5),
                 nn_y=False,
                 model_a=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                            kernel='rbf', gamma=.1, delta_scale='auto',
                            delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5),
                 nn_a=False,
                 alpha=0.05,
                 n_folds=5,
                 n_rep=1,
                 random_seed=123,
                 prop_score=LogisticRegression(),
                 CHIM=False,
                 verbose=True,
                 fitargs1=None,
                 fitargsq1=None,
                 fitargsy=None,
                 fitargsa=None,
                 opts=None
                 ):
        self.Y = Y
        self.D = D
        self.M = M
        self.W = W
        self.Z = Z
        self.X1 = X1
        self.V = V
        self.v_values = v_values
        self.include_V = include_V
        self.ci_type = ci_type
        self.loc_kernel = loc_kernel
        self.bw_loc = bw_loc
        self.estimator = estimator
        self.estimand = estimand
        self.model_y = copy.deepcopy(model_y)
        self.model_a = copy.deepcopy(model_a)
        self.nn_y = nn_y
        self.nn_a = nn_a
        self.prop_score = prop_score
        self.CHIM = CHIM
        self.alpha = alpha
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_seed = random_seed
        self.verbose = verbose
        self.fitargsy = fitargsy
        self.fitargsa = fitargsa
        self.opts = opts

        if isinstance(model1, list):
            self.model1 = copy.deepcopy(model1[0])
            self.model2 = copy.deepcopy(model1[1])
            self.sequential_o = True
            if not isinstance(nn_1, list):
                warnings.warn("Sequential outcome model fitting requires nn_1 to be a list. Assuming [nn_1, nn_1]", UserWarning)
                self.nn_1 = nn_1
                self.nn_2 = nn_1
            else:
                self.nn_1 = nn_1[0]
                self.nn_2 = nn_1[1]
            if not isinstance(fitargs1, list):
                warnings.warn("Sequential outcome model fitting requires fitargs1 to be a list. Assuming [fitargs1, fitargs1]", UserWarning)
                self.fitargs1 = fitargs1
                self.fitargs2 = fitargs1
            else:   
                self.fitargs1 = fitargs1[0]
                self.fitargs2 = fitargs1[1]
        else:
            self.model1 = copy.deepcopy(model1)
            self.nn_1 = nn_1
            self.fitargs1 = fitargs1
            self.sequential_o = False

        if isinstance(modelq1, list):
            self.modelq1 = copy.deepcopy(modelq1[0])
            self.modelq2 = copy.deepcopy(modelq1[1])
            self.sequential_a = True
            if not isinstance(nn_q1, list):
                warnings.warn("Sequential outcome model fitting requires nn_q1 to be a list. Assuming [nn_q1, nn_q1]", UserWarning)
                self.nn_q1 = nn_q1
                self.nn_q2 = nn_q1
            else:
                self.nn_q1 = nn_q1[0]
                self.nn_q2 = nn_q1[1]
            if not isinstance(fitargsq1, list):
                warnings.warn("Sequential outcome model fitting requires fitargsq1 to be a list. Assuming [fitargsq1, fitargsq1]", UserWarning)
                self.fitargsq1 = fitargsq1
                self.fitargsq2 = fitargsq1
            else:   
                self.fitargsq1 = fitargsq1[0]
                self.fitargsq2 = fitargsq1[1]                
        else:
            self.modelq1 = copy.deepcopy(modelq1) 
            self.nn_q1 = nn_q1
            self.fitargsq1 = fitargsq1 
            self.sequential_a = False          
        
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

        lengths = [len(Y), len(D), len(M), len(W), len(Z), len(self.X)]
        if len(set(lengths)) != 1:
            raise ValueError("All input vectors must have the same length.")

        if self.estimator not in ['MR', 'OR', 'hybrid', 'IPW']:
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['MR', 'OR', 'hybrid', 'IPW']. Using MR instead.", UserWarning)
            self.estimator = 'MR'

        if self.estimand not in ['ATE', 'Indirect', 'Direct', 'E[Y1]', 'E[Y0]', 'E[Y(1,M(0))]']:
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['ATE', 'Indirect', 'Direct', 'E[Y1]', 'E[Y0]', 'E[Y(1,M(0))]']. Using ATE instead.", UserWarning)
            self.estimand = 'ATE'

        if self.estimand in ['ATE', 'E[Y1]', 'E[Y0]'] and self.estimator=='hybrid':
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['MR', 'OR', 'IPW'] when estimand is {estimand}. Using MR instead.", UserWarning)
            self.estimator = 'MR'                

        if self.ci_type not in ['pointwise', 'uniform']:
            warnings.warn(f"Invalid confidence interval type: {ci_type}. Confidence interval type must be one of ['pointwise', 'uniform']. Using pointwise instead.", UserWarning)
            self.ci_type = 'pointwise'
        if self.ci_type == 'uniform' and (self.v_values is None or self.v_values.shape[0] == 1 or self.V is None):
            warnings.warn(f"Uniform confidence intervals are not supported for less than one localization value. Using pointwise instead.", UserWarning)
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


    def _nnpivfit_outcome_m(self, Y, D, M, W, X, Z):
        """
        Fit the mediated outcome model using nonparametric instrumental variables.

        Parameters
        ----------
        Y : array-like
            Outcome variable.
        D : array-like
            Treatment variable.
        M : array-like
            Mediator variable.
        W : array-like
            Negative control outcome.
        X : array-like
            Covariates.
        Z : array-like
            Instrumental variable.

        Returns
        -------
        tuple
            Fitted models for treatment and control groups.
        """
        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':
            model_1 = copy.deepcopy(self.model1)
            model_2 = copy.deepcopy(self.model2)

            #First stage
            if self.nn_1==True:
                Y, D, M, W, X, Z = map(toT, [Y, D, M, W, X, Z]) 

            ind = (torch.nonzero(D.reshape(-1) == 1).squeeze(1) if self.nn_1 else np.where(D==1)[0])
            M1 = M[ind]
            W1 = W[ind]
            X1 = X[ind,:]
            Z1 = Z[ind]
            Y1 = Y[ind]

            if self.nn_1==True:
                A2 = torch.cat((M1,X1,Z1),1)
                A1 = torch.cat((M1,X1,W1),1)
            else:
                A2 = _transform_poly(np.column_stack((M1,X1,Z1)),self.opts)
                A1 = _transform_poly(np.column_stack((M1,X1,W1)),self.opts)

            if self.fitargs1 is not None:
                bridge_1 = model_1.fit(A2, A1, Y1, **self.fitargs1)
            else:
                bridge_1 = model_1.fit(A2, A1, Y1)

            if self.nn_1==True:
                A1 = torch.cat((M,X,W),1)
                _pred = bridge_1.predict(A1.to(DEVICE), model='avg', burn_in=_get(self.opts, 'burnin', 0))
                bridge_1_hat = _pred if isinstance(_pred, torch.Tensor) else toT(_pred)

            else:
                A1 = _transform_poly(np.column_stack((M,X,W)),self.opts)
                bridge_1_hat = bridge_1.predict(A1)
                bridge_1_hat = bridge_1_hat.reshape(A1.shape[:1] + Y.shape[1:])
        else:
            bridge_1 = None

        if self.estimator == 'MR' or self.estimator == 'OR':
            #Second stage 
            if self.nn_1!=self.nn_2:
                if self.nn_2==False:
                    D, W, X, Z, bridge_1_hat = map(_to_np, [D, W, X, Z, bridge_1_hat])
                else:
                    D, W, X, Z, bridge_1_hat = map(toT, [D, W, X, Z, bridge_1_hat])

            ind = (torch.nonzero(D.reshape(-1) == 0).squeeze(1) if self.nn_2 else np.where(D==0)[0])
            W0 = W[ind]
            X0 = X[ind,:]
            Z0 = Z[ind]
            bridge_1_hat = bridge_1_hat[ind]

            if self.nn_2==True:
                B2 = torch.cat((X0,Z0),1)
                B1 = torch.cat((X0,W0),1)
            else:            
                B2 = _transform_poly(np.column_stack((X0,Z0)),self.opts)
                B1 = _transform_poly(np.column_stack((X0,W0)),self.opts)

            if self.fitargs2 is not None:
                bridge_2 = model_2.fit(B2, B1, bridge_1_hat, **self.fitargs2)
            else:
                bridge_2 = model_2.fit(B2, B1, bridge_1_hat)
        else:
            bridge_2 = None
        
        Y, D, M, W, X, Z = map(_to_np, [Y, D, M, W, X, Z])
        return bridge_1, bridge_2
    

    def _npivfit_outcome(self, Y, D, X, Z):
        """
        Fit the outcome model using nonparametric instrumental variables.

        Parameters
        ----------
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
        object
            Fitted model.
        """
        model_y1 = copy.deepcopy(self.model_y)

        # First stage
        if self.nn_y==True:
            Y, D, X, Z = map(toT, [Y, D, X, Z])
        else:
            Y, D, X, Z = map(_to_np, [Y, D, X, Z])
            X = _transform_poly(X, self.opts)
            Z = _transform_poly(Z, self.opts)

        ind = (torch.nonzero(D.reshape(-1) == 1).squeeze(1) if self.nn_y else np.where(D==1)[0])
        Y1 = Y[ind]
        X1 = X[ind, :]
        Z1 = Z[ind]

        if self.fitargsy is not None:
            bridge_1 = model_y1.fit(Z1, X1, Y1, **self.fitargsy)
        else:
            bridge_1 = model_y1.fit(Z1, X1, Y1)
        
        Y, D, X, Z = map(_to_np, [Y, D, X, Z])
        return bridge_1

    def _propensity_score(self, M, X, W, D):
        """
        Estimate the propensity score.

        Parameters
        ----------
        M : array-like
            Mediator variable.
        X : array-like
            Covariates.
        W : array-like
            Negative control outcome.
        D : array-like
            Treatment variable.

        Returns
        -------
        tuple
            Estimated propensity scores and threshold alpha.
        """

        M, X, W, D = map(_to_np, (M, X, W, D))

        model_ps = copy.deepcopy(self.prop_score)
        X1 = np.column_stack((X,W))
        X0 = np.column_stack((M,X,W))
            
        #First stage
        model_ps.fit(X1, D.flatten())
        ps_hat_0 = model_ps.predict_proba(X1)[:,0]

        if self.estimand in ['Indirect', 'Direct', 'E[Y(1,M(0))]']:
            #Second stage
            model_ps.fit(X0, D.flatten())
            ps_hat_00 = model_ps.predict_proba(X0)[:,0] 
        else:
            ps_hat_00 = ps_hat_0

        # Overlap assumption
        ps_hat_0 = np.where(ps_hat_0 == 1, 0.99, ps_hat_0)
        ps_hat_0 = np.where(ps_hat_0 == 0, 0.01, ps_hat_0)
        ps_hat_00 = np.where(ps_hat_00 == 1, 0.99, ps_hat_00)
        ps_hat_00 = np.where(ps_hat_00 == 0, 0.01, ps_hat_00)

        if self.CHIM==True:
            # Dropping observations with extreme values of the propensity score - CHIM (2009)
            # One finds the smallest value of \alpha\in [0,0.5] s.t.
            # $\lambda:=\frac{1}{\alpha(1-\alpha)}$
            # $2\frac{\sum 1(g(X)\leq\lambda)*g(X)}{\sum 1(g(X)\leq\lambda)}-\lambda\geq 0$
            # 
            # Equivalently the first value of alpha (in increasing order) such that the constraint is achieved by equality
            # (as the constraint is a monotone increasing function in alpha)

            g_values = [1/(ps_hat_0*(1-ps_hat_0)), 1/(ps_hat_00*(1-ps_hat_00))]  
            optimized_alphas = []

            for g in g_values:
                def _objective_function(alpha):
                    return _fun_threshold_alpha(alpha, g)
                result = minimize_scalar(_objective_function, bounds=(0.001, 0.499))
                optimized_alphas.append(result.x)
            alfa = max(optimized_alphas)
        else:
            alfa = 0.0

        M, X, W, D = map(toT, (M, X, W, D))
        return ps_hat_0.reshape(-1,1), ps_hat_00.reshape(-1,1), alfa


    def _nnpivfit_action_m(self, ps_hat_0, ps_hat_00, D, M, W, X, Z, alfa=0.0):
        """
        Fit the mediated action model using nonparametric instrumental variables.

        Parameters
        ----------
        ps_hat_0 : array-like
            Estimated propensity scores for control group.
        ps_hat_00 : array-like
            Estimated propensity scores for mediated control group.
        D : array-like
            Treatment variable.
        M : array-like
            Mediator variable.
        W : array-like
            Negative control outcome.
        X : array-like
            Covariates.
        Z : array-like
            Instrumental variable.
        alfa : float, optional
            Threshold alpha for propensity scores.

        Returns
        -------
        tuple
            Fitted models for mediated action.
        """
        if self.estimator == 'MR' or self.estimator == 'IPW' or self.estimator == 'hybrid':
            mask = np.where((ps_hat_0 >= alfa) & (ps_hat_0 <= 1 - alfa) &
                            (ps_hat_00 >= alfa) & (ps_hat_00 <= 1 - alfa))[0]
            ps_hat_0 = ps_hat_0[mask]
            ps_hat_00 = ps_hat_00[mask]
            ps_hat_01 = 1 - ps_hat_00

            D = D[mask]
            M = M[mask]
            W = W[mask]
            X = X[mask,:]
            Z = Z[mask]

            model_q1 = copy.deepcopy(self.modelq1)
            model_q2 = copy.deepcopy(self.modelq2)

            #First stage
            if self.nn_q1==True:
                ps_hat_0, ps_hat_00, ps_hat_01, D, M, W, X, Z = map(toT, [ps_hat_0, ps_hat_00, ps_hat_01, D, M, W, X, Z])
            else:
                ps_hat_0, ps_hat_00, ps_hat_01, D, M, W, X, Z = map(_to_np, [ps_hat_0, ps_hat_00, ps_hat_01, D, M, W, X, Z])

            ind = (torch.nonzero(D.reshape(-1) == 0).squeeze(1) if self.nn_q1 else np.where(D==0)[0])
            ps_hat_0 = ps_hat_0[ind]
            W1 = W[ind]
            X1 = X[ind,:]
            Z1 = Z[ind]

            if self.nn_q1==True:
                A2 = torch.cat((X1,W1),1)
                A1 = torch.cat((X1,Z1),1)
            else:
                A2 = _transform_poly(np.column_stack((X1,W1)),self.opts)
                A1 = _transform_poly(np.column_stack((X1,Z1)),self.opts)

            if self.fitargsq1 is not None:
                bridge_1 = model_q1.fit(A2, A1, 1/ps_hat_0, **self.fitargsq1)
            else:
                bridge_1 = model_q1.fit(A2, A1, 1/ps_hat_0)

            if self.nn_q1==True:
                A1 = torch.cat((X,Z),1)
                _pred = bridge_1.predict(A1.to(DEVICE), model='avg', burn_in=_get(self.opts, 'burnin', 0))
                bridge_1_hat = _pred if isinstance(_pred, torch.Tensor) else toT(_pred)
            else:    
                A1 = _transform_poly(np.column_stack((X,Z)),self.opts)    
                bridge_1_hat = bridge_1.predict(A1)
                bridge_1_hat = bridge_1_hat.reshape(A1.shape[:1] + ps_hat_0.shape[1:])
        else:
            bridge_1 = None
           

        if self.estimator == 'MR' or self.estimator == 'IPW':
            #Second stage
            if self.nn_q1!=self.nn_q2:
                if self.nn_q2==False:
                    D, M, W, X, Z, bridge_1_hat, ps_hat_00, ps_hat_01 = map(_to_np, [D, M, W, X, Z, bridge_1_hat, ps_hat_00, ps_hat_01])
                else:
                    D, M, W, X, Z, bridge_1_hat, ps_hat_00, ps_hat_01 = map(toT, [D, M, W, X, Z, bridge_1_hat, ps_hat_00, ps_hat_01])

            bridge_1_hat = bridge_1_hat*(ps_hat_00/ps_hat_01)
            ind = (torch.nonzero(D.reshape(-1) == 1).squeeze(1) if self.nn_q2 else np.where(D==1)[0])
            M0 = M[ind]
            W0 = W[ind]
            X0 = X[ind,:]
            Z0 = Z[ind]
            bridge_1_hat = bridge_1_hat[ind]

            if self.nn_q2==True:
                B2 = torch.cat((M0,X0,W0),1)
                B1 = torch.cat((M0,X0,Z0),1)
            else:     
                B2 = _transform_poly(np.column_stack((M0,X0,W0)),self.opts)
                B1 = _transform_poly(np.column_stack((M0,X0,Z0)),self.opts)

            if self.fitargsq2 is not None:
                bridge_2 = model_q2.fit(B2, B1, bridge_1_hat, **self.fitargsq2)
            else:
                bridge_2 = model_q2.fit(B2, B1, bridge_1_hat)
        else:
            bridge_2 = None

        ps_hat_0, ps_hat_00, D, M, W, X, Z = map(_to_np, [ps_hat_0, ps_hat_00, D, M, W, X, Z])
        return bridge_1, bridge_2
    

    def _npivfit_action(self, ps_hat_1, W, X, Z, alfa=0.0):
        """
        Fit the action model using nonparametric instrumental variables.

        Parameters
        ----------
        ps_hat_1 : array-like
            Estimated propensity scores.
        W : array-like
            Negative control outcome.
        X : array-like
            Covariates.
        Z : array-like
            Instrumental variable.
        alfa : float, optional
            Threshold alpha for propensity scores.

        Returns
        -------
        object
            Fitted model for the action.
        """
        mask = np.where((ps_hat_1 >= alfa) & (ps_hat_1 <= 1 - alfa))[0]
        ps_hat_1 = ps_hat_1[mask]
        W = W[mask]
        X = X[mask, :]
        Z = Z[mask]

        model_a1 = copy.deepcopy(self.model_a)

        # First stage
        if self.nn_a==True:
            ps_hat_1, W, X, Z = map(toT, [ps_hat_1, W, X, Z])
            A2 = torch.cat((X, W), 1)
            A1 = torch.cat((X, Z), 1)
        else:
            ps_hat_1, W, X, Z = map(_to_np, [ps_hat_1, W, X, Z])
            A2 = _transform_poly(np.column_stack((X, W)), self.opts)
            A1 = _transform_poly(np.column_stack((X, Z)), self.opts)

        if self.fitargsa is not None:
            bridge_1 = model_a1.fit(A2, A1, 1 / ps_hat_1, **self.fitargsa)
        else:
            bridge_1 = model_a1.fit(A2, A1, 1 / ps_hat_1)

        ps_hat_1, W, X, Z = map(_to_np, [ps_hat_1, W, X, Z])
        return bridge_1

    def _scores_mediated(self, train_Y, train_D, train_M, train_W, train_X, train_Z, 
                         test_Y, test_D, test_M, test_W, test_X, test_Z):
        """
        Calculate the scores for the mediated effects.

        Parameters
        ----------
        train_Y : array-like
            Training outcome variable.
        train_D : array-like
            Training treatment variable.
        train_M : array-like
            Training mediator variable.
        train_W : array-like
            Training negative control outcome.
        train_X : array-like
            Training covariates.
        train_Z : array-like
            Training instrumental variable.
        test_Y : array-like
            Testing outcome variable.
        test_D : array-like
            Testing treatment variable.
        test_M : array-like
            Testing mediator variable.
        test_W : array-like
            Testing negative control outcome.
        test_X : array-like
            Testing covariates.
        test_Z : array-like
            Testing instrumental variable.

        Returns
        -------
        array-like
            Estimated moment functions for the test data.
        """
        model_1 = copy.deepcopy(self.model1)
        model_q1 = copy.deepcopy(self.modelq1)

        # Outcome model
        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':

            if self.sequential_o==True:
                gamma_1, gamma_0 = self._nnpivfit_outcome_m(train_Y, train_D, train_M, train_W, train_X, train_Z)
                # Evaluate the estimated moment functions using test_data
                if self.estimator == 'MR' or self.estimator == 'hybrid':
                    if self.nn_1 == True:
                        test_M, test_X, test_W = map(toT, [test_M, test_X, test_W])
                        gamma_1_hat = gamma_1.predict(torch.cat((test_M, test_X, test_W), 1).to(DEVICE),
                                                    model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                    else:
                        test_M, test_X, test_W = map(_to_np, [test_M, test_X, test_W])
                        gamma_1_hat = gamma_1.predict(_transform_poly(np.column_stack((test_M, test_X, test_W)), opts=self.opts)).reshape(-1, 1)

                if self.estimator == 'MR' or self.estimator == 'OR':
                    if self.nn_2 == True:
                        test_X, test_W = map(toT, [test_X, test_W])
                        gamma_0_hat = gamma_0.predict(torch.cat((test_X, test_W), 1).to(DEVICE),
                                                    model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                    else:
                        test_X, test_W = map(_to_np, [test_X, test_W])
                        gamma_0_hat = gamma_0.predict(_transform_poly(np.column_stack((test_X, test_W)), opts=self.opts)).reshape(-1, 1)

            else:
                A_train = np.column_stack((train_M, train_X, train_W))
                E_train = np.column_stack((train_M, train_X, train_Z))
                B_train = np.column_stack((train_X, train_W))
                C_train = np.column_stack((train_X, train_Z))
                B_test = np.column_stack((test_X, test_W))
                A_test = np.column_stack((test_M, test_X, test_W))

                if self.nn_1 == True:
                    A_train, E_train, B_train, C_train, B_test, A_test, train_Y, train_D = map(toT, 
                                                            [A_train, E_train, B_train, C_train, B_test, A_test, train_Y, train_D])

                else:
                    A_train = _transform_poly(A_train, self.opts)
                    E_train = _transform_poly(E_train, self.opts)
                    B_train = _transform_poly(B_train, self.opts)
                    C_train = _transform_poly(C_train, self.opts)
                    B_test = _transform_poly(B_test, self.opts)
                    A_test = _transform_poly(A_test, self.opts)

                if self.fitargs1 is not None:
                    model_1.fit(A_train, B_train, C_train, E_train, train_Y, subsetted=True, subset_ind1=train_D, **self.fitargs1)
                else:
                    model_1.fit(A_train, B_train, C_train, E_train, train_Y, subsetted=True, subset_ind1=train_D)

                if self.nn_1 == True:
                    gamma_0_hat, gamma_1_hat = model_1.predict(B_test.to(DEVICE), A_test.to(DEVICE), model='avg', burn_in=_get(self.opts, 'burnin', 0))
                    gamma_0_hat = gamma_0_hat.reshape(-1, 1)
                    gamma_1_hat = gamma_1_hat.reshape(-1, 1)
                else:
                    gamma_0_hat, gamma_1_hat = model_1.predict(B_test, A_test)
                    gamma_0_hat = gamma_0_hat.reshape(-1, 1)
                    gamma_1_hat = gamma_1_hat.reshape(-1, 1)

        # Action model
        if self.estimator == 'MR' or self.estimator == 'IPW' or self.estimator == 'hybrid':

            if self.sequential_a==True:
                ps_hat_0, ps_hat_00, alfa = self._propensity_score(train_M, train_X, train_W, train_D)
                q_0, q_1 = self._nnpivfit_action_m(ps_hat_0, ps_hat_00, train_D, train_M, train_W, train_X, train_Z, alfa=alfa)
                # Evaluate the estimated moment functions using test_data
                if self.estimator == 'MR' or self.estimator == 'hybrid':
                    if self.nn_q1 == True:
                        test_X, test_Z = map(toT, [test_X, test_Z])
                        q_0_hat = q_0.predict(torch.cat((test_X, test_Z), 1).to(DEVICE),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                    else:
                        test_X, test_Z = map(_to_np, [test_X, test_Z])
                        q_0_hat = q_0.predict(_transform_poly(np.column_stack((test_X, test_Z)), opts=self.opts)).reshape(-1, 1)

                if self.estimator == 'MR' or self.estimator == 'IPW':
                    if self.nn_q2 == True:
                        test_M, test_X, test_Z = map(toT, [test_M, test_X, test_Z])
                        q_1_hat = q_1.predict(torch.cat((test_M, test_X, test_Z), 1).to(DEVICE),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
                    else:
                        test_M, test_X, test_Z = map(_to_np, [test_M, test_X, test_Z])
                        q_1_hat = q_1.predict(_transform_poly(np.column_stack((test_M, test_X, test_Z)), opts=self.opts)).reshape(-1, 1)

            else:
                A_train = np.column_stack((train_X, train_Z))
                E_train = np.column_stack((train_X, train_W))
                B_train = np.column_stack((train_M, train_X, train_Z))
                C_train = np.column_stack((train_M, train_X, train_W))
                B_test = np.column_stack((test_M, test_X, test_Z))
                A_test = np.column_stack((test_X, test_Z))

                ps_hat_0, ps_hat_00, alfa = self._propensity_score(train_M, train_X, train_W, train_D)
                mask = np.where((ps_hat_0 >= alfa) & (ps_hat_0 <= 1 - alfa) &
                                (ps_hat_00 >= alfa) & (ps_hat_00 <= 1 - alfa))[0]
                ps_hat_0 = ps_hat_0[mask]
                ps_hat_00 = ps_hat_00[mask]
                ps_hat_01 = 1 - ps_hat_00

                A_train, E_train, B_train, C_train, train_D = map(lambda x: x[mask], 
                                                    [A_train, E_train, B_train, C_train, train_D])
                
                if self.nn_1 == True:
                    A_train, E_train, B_train, C_train, train_D, ps_hat_0, ps_hat_00, ps_hat_01, B_test, A_test = map(
                        toT, [A_train, E_train, B_train, C_train, train_D, ps_hat_0, ps_hat_00, ps_hat_01, B_test, A_test]
                    )
                    
                else:
                    A_train = _transform_poly(A_train, self.opts)
                    E_train = _transform_poly(E_train, self.opts)
                    B_train = _transform_poly(B_train, self.opts)
                    C_train = _transform_poly(C_train, self.opts)
                    B_test = _transform_poly(B_test, self.opts)
                    A_test = _transform_poly(A_test, self.opts)

                if self.fitargs1 is not None:
                    #Using weights in the action stage
                    model_q1.fit(A_train, B_train, C_train, E_train, 1/ps_hat_0, W=(ps_hat_00/ps_hat_01), subsetted=True, subset_ind1=1-train_D, **self.fitargs1)
                else:
                    model_q1.fit(A_train, B_train, C_train, E_train, 1/ps_hat_0, W=(ps_hat_00/ps_hat_01), subsetted=True, subset_ind1=1-train_D)

                if self.nn_1 == True:
                    q_1_hat, q_0_hat = model_q1.predict(B_test.to(DEVICE), A_test.to(DEVICE), model='avg', burn_in=_get(self.opts, 'burnin', 0))
                    q_0_hat = q_0_hat.reshape(-1, 1)
                    q_1_hat = q_1_hat.reshape(-1, 1)
                else:
                    q_1_hat, q_0_hat = model_q1.predict(B_test, A_test)
                    q_0_hat = q_0_hat.reshape(-1, 1)
                    q_1_hat = q_1_hat.reshape(-1, 1)

        # Calculate the score function depending on the estimator
        if self.estimator == 'MR':
            psi_hat = (gamma_0_hat +
                    test_D * q_1_hat * (test_Y - gamma_1_hat) +
                    (1 - test_D) * q_0_hat * (gamma_1_hat - gamma_0_hat))
        if self.estimator == 'OR':
            psi_hat = gamma_0_hat 
        if self.estimator == 'hybrid':
            psi_hat = (1 - test_D) * q_0_hat * gamma_1_hat
        if self.estimator == 'IPW':
            psi_hat = test_D * q_1_hat * test_Y
        return psi_hat

    def _scores_Y1(self, train_Y, train_D, train_M, train_W, train_X, train_Z, 
                         test_Y, test_D, test_X, test_Z):
        """
        Calculate the scores for the Y1 estimand.

        Parameters
        ----------
        train_Y : array-like
            Training outcome variable.
        train_D : array-like
            Training treatment variable.
        train_M : array-like
            Training mediator variable.
        train_W : array-like
            Training negative control outcome.
        train_X : array-like
            Training covariates.
        train_Z : array-like
            Training instrumental variable.
        test_Y : array-like
            Testing outcome variable.
        test_D : array-like
            Testing treatment variable.
        test_X : array-like
            Testing covariates.
        test_Z : array-like
            Testing instrumental variable.

        Returns
        -------
        array-like
            Estimated moment functions for the test data.
        """
        if self.estimator == 'MR' or self.estimator == 'OR':
            gamma_1 = self._npivfit_outcome(train_Y, train_D, train_X, train_Z)

        if self.estimator == 'MR' or self.estimator == 'IPW' or self.estimator == 'hybrid':
            ps_hat_0, _, alfa = self._propensity_score(train_M, train_X, train_W, train_D)
            q_1 = self._npivfit_action(1-ps_hat_0, train_W, train_X, train_Z, alfa=alfa)

        # Evaluate the estimated moment functions using test_data
        if self.estimator == 'MR' or self.estimator == 'OR':
            if self.nn_y == True:
                test_X = toT(test_X)
                gamma_1_hat = gamma_1.predict(test_X.to(DEVICE),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                test_X = _to_np(test_X)
                gamma_1_hat = gamma_1.predict(_transform_poly(test_X, opts=self.opts)).reshape(-1, 1)

        if self.estimator == 'MR' or self.estimator == 'IPW' or self.estimator == 'hybrid':
            if self.nn_a == True:
                test_X, test_Z = map(toT, [test_X, test_Z])
                q_1_hat = q_1.predict(torch.cat((test_X, test_Z), 1).to(DEVICE),
                                    model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                test_X, test_Z = map(_to_np, [test_X, test_Z])
                q_1_hat = q_1.predict(_transform_poly(np.column_stack((test_X, test_Z)), opts=self.opts)).reshape(-1, 1)

        # Calculate the score function depending on the estimator
        if self.estimator == 'MR':
            psi_hat = gamma_1_hat + test_D * q_1_hat * (test_Y - gamma_1_hat) 
        if self.estimator == 'OR':
            psi_hat = gamma_1_hat
        if self.estimator == 'IPW' or self.estimator == 'hybrid':
            psi_hat = test_D * q_1_hat * test_Y 
        return psi_hat
    
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
        train_M, test_M = train_data[2], test_data[2]
        train_W, test_W = train_data[3], test_data[3]
        train_X, test_X = train_data[4], test_data[4]
        train_Z, test_Z = train_data[5], test_data[5]
        if self.V is not None:
            train_V, test_V = train_data[6], test_data[6]

        if self.estimand == 'ATE':
            psi_hat_1 = self._scores_Y1(train_Y, train_D, train_M, train_W, train_X, train_Z,
                                        test_Y, test_D, test_X, test_Z)
            psi_hat_0 = self._scores_Y1(train_Y, 1-train_D, train_M, train_W, train_X, train_Z,
                                        test_Y, 1-test_D, test_X, test_Z)
            psi_hat = psi_hat_1 - psi_hat_0
        if self.estimand == 'Indirect':
            psi_hat_mediated = self._scores_mediated(train_Y, train_D, train_M, train_W, train_X, train_Z, 
                                            test_Y, test_D, test_M, test_W, test_X, test_Z)
            psi_hat_1 = self._scores_Y1(train_Y, train_D, train_M, train_W, train_X, train_Z,
                                        test_Y, test_D, test_X, test_Z)
            psi_hat = psi_hat_1 - psi_hat_mediated 
        if self.estimand == 'Direct':
            psi_hat_mediated = self._scores_mediated(train_Y, train_D, train_M, train_W, train_X, train_Z, 
                                            test_Y, test_D, test_M, test_W, test_X, test_Z)
            psi_hat_0 = self._scores_Y1(train_Y, 1-train_D, train_M, train_W, train_X, train_Z,
                                        test_Y, 1-test_D, test_X, test_Z)
            psi_hat = psi_hat_mediated - psi_hat_0
        if self.estimand == 'E[Y1]':
            psi_hat = self._scores_Y1(train_Y, train_D, train_M, train_W, train_X, train_Z,
                                        test_Y, test_D, test_X, test_Z)
        if self.estimand == 'E[Y0]':
            psi_hat = self._scores_Y1(train_Y, 1-train_D, train_M, train_W, train_X, train_Z,
                                        test_Y, 1-test_D, test_X, test_Z)
        if self.estimand == 'E[Y(1,M(0))]':
            psi_hat = self._scores_mediated(train_Y, train_D, train_M, train_W, train_X, train_Z, 
                                            test_Y, test_D, test_M, test_W, test_X, test_Z)            
        

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

        # Print progress bar using tqdm
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
                        (self.Y[train_index], self.D[train_index], self.M[train_index], self.W[train_index],
                        self.X[train_index], self.Z[train_index]),
                        (self.Y[test_index], self.D[test_index], self.M[test_index], self.W[test_index],
                        self.X[test_index], self.Z[test_index])) 
                        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.Y))
                )
            else:   
                fold_results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(self._process_fold)(
                        fold_idx, 
                        (self.Y[train_index], self.D[train_index], self.M[train_index], self.W[train_index],
                        self.X[train_index], self.Z[train_index], self.V[train_index]),
                        (self.Y[test_index], self.D[test_index], self.M[test_index], self.W[test_index],
                        self.X[test_index], self.Z[test_index], self.V[test_index])) 
                        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.Y))
                )                
            if self.verbose==True:       
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
        Perform Debiased Machine Learning for Nonparametric Instrumental Variables.

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
