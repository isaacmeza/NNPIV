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


class DML_npiv:

    def __init__(self, Y, D, Z, W, X1=None,
                 V=None, 
                 v_values=None,
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
        self.Y = Y
        self.D = D
        self.Z = Z
        self.W = W
        self.X1 = X1
        self.V = V
        self.v_values = v_values
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

        lengths = [len(Y), len(D), len(Z), len(W), len(self.X)]
        if len(set(lengths)) != 1:
            raise ValueError("All input vectors must have the same length.")
        

        if self.estimator not in ['MR', 'OR', 'IPW']:
            warnings.warn(f"Invalid estimator: {estimator}. Estimator must be one of ['MR', 'OR', 'IPW']. Using MR instead.", UserWarning)
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
    
    def _npivfit_outcome(self, Y, D, X, Z):

        bridge_ = [None]*2

        if self.estimator == 'MR' or self.estimator == 'OR':
            model_1 = copy.deepcopy(self.model1)

            #First stage
            if self.nn_1==True:
                Y, X, Z = map(lambda x: torch.Tensor(x), [Y, X, Z]) 

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
        
        return copy.deepcopy(bridge_[1]), copy.deepcopy(bridge_[0])


    def _propensity_score(self, X, W, D):
        
        model_ps = copy.deepcopy(self.prop_score)
        X1 = np.column_stack((X,W))
            
        #First stage
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

        bridge_ = [None]*2

        if self.estimator == 'MR' or self.estimator == 'IPW':
            mask = np.where((ps_hat_1 >= alfa) & (ps_hat_1 <= 1 - alfa))[0]
            ps_hat_1 = ps_hat_1[mask]
            ps_hat_0 = 1 - ps_hat_1

            W = W[mask]
            X = X[mask,:]
            Z = Z[mask]

            model_q1 = copy.deepcopy(self.modelq1)

            #First stage
            if self.nn_q1==True:
                ps_hat_1, ps_hat_0, W, X, Z = map(lambda x: torch.Tensor(x), [ps_hat_1, ps_hat_0, W, X, Z]) 

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
           
        return copy.deepcopy(bridge_[1]), copy.deepcopy(bridge_[0])


    def _process_fold(self, fold_idx, train_data, test_data):
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
                test_X = map(lambda x: torch.Tensor(x), [test_X])
                gamma_1_hat = gamma_1.predict(torch.cat((test_X), 1).to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0))
                gamma_0_hat = gamma_0.predict(torch.cat((test_X), 1).to(device),
                                            model='avg', burn_in=_get(self.opts, 'burnin', 0))
        
            else:
                gamma_1_hat = gamma_1.predict(_transform_poly(test_X, opts=self.opts))
                gamma_0_hat = gamma_0.predict(_transform_poly(test_X, opts=self.opts))

        if self.estimator == 'MR' or self.estimator == 'IPW':
            if self.nn_q1 == True:
                test_X, test_Z = map(lambda x: torch.Tensor(x), [test_X, test_Z])
                q_1_hat = q_1.predict(torch.cat((test_X, test_Z), 1).to(device),
                                    model='avg', burn_in=_get(self.opts, 'burnin', 0))
                q_0_hat = q_0.predict(torch.cat((test_X, test_Z), 1).to(device),
                                    model='avg', burn_in=_get(self.opts, 'burnin', 0))
            else:
                q_1_hat = q_1.predict(_transform_poly(np.column_stack((test_X, test_Z)), opts=self.opts))
                q_0_hat = q_0.predict(_transform_poly(np.column_stack((test_X, test_Z)), opts=self.opts))

        # Calculate the score function depending on the estimator
        if self.estimator == 'MR':
            psi_hat = (gamma_1_hat-gamma_0_hat +
                    test_D * q_1_hat * (test_Y - gamma_1_hat) - (1-test_D) * q_0_hat * (test_Y - gamma_0_hat))
        if self.estimator == 'OR':
            psi_hat = gamma_1_hat-gamma_0_hat 
        if self.estimator == 'IPW':
            psi_hat = test_D * q_1_hat * test_Y - (1 - test_D) * q_0_hat * test_Y

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
    
