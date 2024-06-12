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
from mliv.rkhs import ApproxRKHSIVCV
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

device = torch.cuda.current_device() if torch.cuda.is_available() else None

def _get(opts, key, default):
    return opts[key] if (opts is not None and key in opts) else default

def _transform_poly(X, opts):
    degree = _get(opts, 'lin_degree', 1)
    if degree == 1:
        return X
    else:
        trans = PolynomialFeatures(degree=degree, include_bias=False)
        return trans.fit_transform(X)

def _fun_threshold_alpha(alpha, g):
    lambda_val = 1 / (alpha * (1 - alpha))
    ind = (g <= lambda_val)
    den = sum(ind)
    num = ind * g
    result = (2 * sum(num) / den - lambda_val) ** 2
    return result


class DML_mediated:

    def __init__(self, Y, D, M, W, Z, X1=None,
                 V=None, 
                 v_values=None,
                 loc_kernel='gau',
                 bw_loc='silverman',
                 estimator='MR',
                 estimand='ATE',
                 model1=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5), 
                 nn_1=False,
                 model2=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5), 
                 nn_2=False,
                 modelq1=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5), 
                 nn_q1=False,
                 modelq2=ApproxRKHSIVCV(kernel_approx='nystrom', n_components=100,
                           kernel='rbf', gamma=.1, delta_scale='auto',
                           delta_exp=.4, alpha_scales=np.geomspace(1, 10000, 10), cv=5), 
                 nn_q2=False,
                 alpha=0.05,
                 n_folds=5,
                 n_rep=1,
                 random_seed=123,
                 prop_score=LogisticRegression(),
                 CHIM=False,
                 verbose=True,
                 fitargs1=None,
                 fitargs2=None,
                 fitargsq1=None,
                 fitargsq2=None,
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
        self.loc_kernel = loc_kernel
        self.bw_loc = bw_loc
        self.estimator = estimator
        self.estimand = estimand
        self.model1 = copy.deepcopy(model1)
        self.model2 = copy.deepcopy(model2)
        self.modelq1 = copy.deepcopy(modelq1)
        self.modelq2 = copy.deepcopy(modelq2)
        self.nn_1 = nn_1
        self.nn_2 = nn_2
        self.nn_q1 = nn_q1
        self.nn_q2 = nn_q2
        self.prop_score = prop_score
        self.CHIM = CHIM
        self.alpha = alpha
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_seed = random_seed
        self.verbose = verbose
        self.fitargs1 = fitargs1
        self.fitargs2 = fitargs2
        self.fitargsq1 = fitargsq1
        self.fitargsq2 = fitargsq2
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

        lengths = [len(Y), len(D), len(M), len(W), len(Z), len(self.X)]
        if len(set(lengths)) != 1:
            raise ValueError("All input vectors must have the same length.")

        if self.estimator not in ['MR', 'OR', 'hybrid', 'IPW']:
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['MR', 'OR', 'hybrid', 'IPW']. Using MR instead.", UserWarning)
            self.estimator = 'MR'

        if self.estimand not in ['ATE', 'Indirect', 'Direct', 'E[Y1]', 'E[Y0]', 'E[Y(1,M(0))]']:
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['ATE', 'Indirect', 'Direct', 'E[Y1]', 'E[Y0]', 'E[Y(1,M(0))]']. Using ATE instead.", UserWarning)
            self.estimand = 'ATE'

        if self.estimand in ['ATE', 'E[Y1]', 'E[Y0]'] & self.estimator=='hybrid':
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['MR', 'OR', 'IPW'] when estimand is {estimand}. Using MR instead.", UserWarning)
            self.estimator = 'MR'                

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
        z_alpha_half = norm.ppf(1 - self.alpha / 2)
        n = self.Y.shape[0]
        margin_of_error = z_alpha_half * np.sqrt(theta_var) * np.sqrt(1 / n)
        lower_bound = theta - margin_of_error
        upper_bound = theta + margin_of_error
        return np.column_stack((lower_bound, upper_bound))

    def _localization(self, V, v_val, bw):
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

        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':
            model_1 = copy.deepcopy(self.model1)
            model_2 = copy.deepcopy(self.model2)

            #First stage
            if self.nn_1==True:
                Y, D, M, W, X, Z = map(lambda x: torch.Tensor(x), [Y, D, M, W, X, Z]) 

            ind = np.where(D==1)[0]
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
                bridge_1_hat = torch.Tensor(bridge_1.predict(A1.to(device),
                            model='avg', burn_in=_get(self.opts, 'burnin', 0)))
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
                    D, W, X, Z, bridge_1_hat = map(lambda x: x.numpy(), [D, W, X, Z, bridge_1_hat])
                else:
                    D, W, X, Z, bridge_1_hat = map(lambda x: torch.Tensor(x), [D, W, X, Z, bridge_1_hat])

            ind = np.where(D==0)[0]
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
        
        return bridge_1, bridge_2


    def _npivfit_outcome(self, Y, D, X, Z):
        model_1 = copy.deepcopy(self.model1)

        # First stage
        if self.nn_1==True:
            Y, X, Z = tuple(map(lambda x: torch.Tensor(x), [Y, X, Z]))
        else:
            X = _transform_poly(X, self.opts)
            Z = _transform_poly(Z, self.opts)

        ind = np.where(D==1)[0]
        Y1 = Y[ind]
        X1 = X[ind, :]
        Z1 = Z[ind]

        if self.fitargs1 is not None:
            bridge_1 = model_1.fit(Z1, X1, Y1, **self.fitargs1)
        else:
            bridge_1 = model_1.fit(Z1, X1, Y1)
        
        return bridge_1
    

    def _propensity_score(self, M, X, W, D):
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

        return ps_hat_0.reshape(-1,1), ps_hat_00.reshape(-1,1), alfa


    def _nnpivfit_action_m(self, ps_hat_0, ps_hat_00, D, M, W, X, Z, alfa=0.0):

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
                ps_hat_0, ps_hat_00, ps_hat_01, D, M, W, X, Z = map(lambda x: torch.Tensor(x), [ps_hat_0, ps_hat_00, ps_hat_01, D, M, W, X, Z]) 

            ind = np.where(D==0)[0]
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
                bridge_1_hat = torch.Tensor(bridge_1.predict(A1.to(device),
                            model='avg', burn_in=_get(self.opts, 'burnin', 0)))
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
                    D, M, W, X, Z, bridge_1_hat, ps_hat_00, ps_hat_01 = map(lambda x: x.numpy(), [D, M, W, X, Z, bridge_1_hat, ps_hat_00, ps_hat_01])
                else:
                    D, M, W, X, Z, bridge_1_hat, ps_hat_00, ps_hat_01 = map(lambda x: torch.Tensor(x), [D, M, W, X, Z, bridge_1_hat, ps_hat_00, ps_hat_01])

            bridge_1_hat = bridge_1_hat*(ps_hat_00/ps_hat_01)
            ind = np.where(D==1)[0]
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

        return bridge_1, bridge_2
    

    def _npivfit_action(self, ps_hat_1, W, X, Z, alfa=0.0):
        mask = np.where((ps_hat_1 >= alfa) & (ps_hat_1 <= 1 - alfa))[0]
        ps_hat_1 = ps_hat_1[mask]
        W = W[mask]
        X = X[mask, :]
        Z = Z[mask]

        model_q1 = copy.deepcopy(self.modelq1)

        # First stage
        if self.nn_q1==True:
            ps_hat_1, W, X, Z = tuple(map(lambda x: torch.Tensor(x), [ps_hat_1, W, X, Z]))
            A2 = torch.cat((X, W), 1)
            A1 = torch.cat((X, Z), 1)
        else:
            A2 = _transform_poly(np.column_stack((X, W)), self.opts)
            A1 = _transform_poly(np.column_stack((X, Z)), self.opts)

        if self.fitargsq1 is not None:
            bridge_1 = model_q1.fit(A2, A1, 1 / ps_hat_1, **self.fitargsq1)
        else:
            bridge_1 = model_q1.fit(A2, A1, 1 / ps_hat_1)

        return bridge_1

    def _scores_mediated(self, train_Y, train_D, train_M, train_W, train_X, train_Z, 
                         test_Y, test_D, test_M, test_W, test_X, test_Z):
        
        if self.estimator == 'MR' or self.estimator == 'OR' or self.estimator == 'hybrid':
            gamma_1, gamma_0 = self._nnpivfit_outcome_m(train_Y, train_D, train_M, train_W, train_X, train_Z)
        if self.estimator == 'MR' or self.estimator == 'hybrid' or self.estimator == 'IPW':
            ps_hat_0, ps_hat_00, alfa = self._propensity_score(train_M, train_X, train_W, train_D)
            q_0, q_1 = self._nnpivfit_action_m(ps_hat_0, ps_hat_00, train_D, train_M, train_W, train_X, train_Z, alfa=alfa)

        # Evaluate the estimated moment functions using test_data
        if self.estimator == 'MR' or self.estimator == 'hybrid':
            if self.nn_1 == True:
                test_M, test_X, test_W = tuple(map(lambda x: torch.Tensor(x), [test_M, test_X, test_W]))
                gamma_1_hat = gamma_1.predict(torch.cat((test_M, test_X, test_W), 1).to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                gamma_1_hat = gamma_1.predict(_transform_poly(np.column_stack((test_M, test_X, test_W)), opts=self.opts)).reshape(-1, 1)

        if self.estimator == 'MR' or self.estimator == 'OR':
            if self.nn_2 == True:
                test_X, test_W = tuple(map(lambda x: torch.Tensor(x), [test_X, test_W]))
                gamma_0_hat = gamma_0.predict(torch.cat((test_X, test_W), 1).to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                gamma_0_hat = gamma_0.predict(_transform_poly(np.column_stack((test_X, test_W)), opts=self.opts)).reshape(-1, 1)

        if self.estimator == 'MR' or self.estimator == 'hybrid':
            if self.nn_q1 == True:
                test_X, test_Z = tuple(map(lambda x: torch.Tensor(x), [test_X, test_Z]))
                q_0_hat = q_0.predict(torch.cat((test_X, test_Z), 1).to(device),
                                    model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                q_0_hat = q_0.predict(_transform_poly(np.column_stack((test_X, test_Z)), opts=self.opts)).reshape(-1, 1)

        if self.estimator == 'MR' or self.estimator == 'IPW':
            if self.nn_q2 == True:
                test_M, test_X, test_Z = tuple(map(lambda x: torch.Tensor(x), [test_M, test_X, test_Z]))
                q_1_hat = q_1.predict(torch.cat((test_M, test_X, test_Z), 1).to(device),
                                    model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                q_1_hat = q_1.predict(_transform_poly(np.column_stack((test_M, test_X, test_Z)), opts=self.opts)).reshape(-1, 1)

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
        
        if self.estimator == 'MR' or self.estimator == 'OR':
            gamma_1 = self._npivfit_outcome(train_Y, train_D, train_X, train_Z)

        if self.estimator == 'MR' or self.estimator == 'IPW' or self.estimator == 'hybrid':
            ps_hat_0, _, alfa = self._propensity_score(train_M, train_X, train_W, train_D)
            q_1 = self._npivfit_action(1-ps_hat_0, train_W, train_X, train_Z, alfa=alfa)

        # Evaluate the estimated moment functions using test_data
        if self.estimator == 'MR' or self.estimator == 'OR':
            if self.nn_1 == True:
                test_X = torch.Tensor(test_X)
                gamma_1_hat = gamma_1.predict(test_X.to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
                gamma_1_hat = gamma_1.predict(_transform_poly(test_X, opts=self.opts)).reshape(-1, 1)

        if self.estimator == 'MR' or self.estimator == 'IPW' or self.estimator == 'hybrid':
            if self.nn_q1 == True:
                test_X, test_Z = tuple(map(lambda x: torch.Tensor(x), [test_X, test_Z]))
                q_1_hat = q_1.predict(torch.cat((test_X, test_Z), 1).to(device),
                                    model='avg', burn_in=_get(self.opts, 'burnin', 0)).reshape(-1, 1)
            else:
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

        # Print progress bar using tqdm
        if self.verbose==True:
            self.progress_bar.update(1)
   
        return psi_hat


    def _split_and_estimate(self):
        
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
            theta_var_rep = np.var(psi_hat_array, axis=0)

            # Store results for each rep
            theta.append(theta_rep)
            theta_var.append(theta_var_rep)

        # Calculate the overall average of theta and theta_var
        theta_hat = np.mean(np.stack(theta, axis=0), axis=0)
        theta_var_hat = np.mean(np.stack(theta_var, axis=0), axis=0)

        # Calculate the confidence interval
        confidence_interval = self._calculate_confidence_interval(theta_hat, theta_var_hat)

        return theta_hat, theta_var_hat, confidence_interval
    

    def dml(self):
        theta, theta_var, confidence_interval = self._split_and_estimate()
        if self.V is None:
            return theta[0], theta_var[0], confidence_interval[0]
        else:
            return theta, theta_var, confidence_interval
    
