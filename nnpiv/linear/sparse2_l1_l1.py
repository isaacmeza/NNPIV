r"""
This module provides implementations of sparse linear NPIV estimators with L1 norm regularization for nested NPIV.

Classes:
-------
_SparseLinear2AdversarialGMM
    Base class for sparse linear adversarial GMM for nested NPIV.
sparse2_l1vsl1
    Sparse Linear NPIV estimator using :math:`\ell_1-\ell_1` optimization for nested NPIV.
sparse2_ridge_l1vsl1
    Sparse Ridge NPIV estimator using :math:`\ell_1-\ell_1` optimization for nested NPIV.
"""

# Licensed under the MIT License.

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from nnpiv.linear.utilities import cross_product


class _SparseLinear2AdversarialGMM:
    """
    Base class for sparse linear adversarial GMM for nested NPIV.

    This class implements common functionality for sparse linear models using adversarial GMM in a nested NPIV setting.

    Parameters:
        mu (float): Regularization parameter.
        V1 (int): Budget parameter for the first stage.
        V2 (int): Budget parameter for the second stage.
        eta_alpha (str or float): Learning rate for alpha.
        eta_w1 (str or float): Learning rate for w1.
        eta_beta (str or float): Learning rate for beta.
        eta_w2 (str or float): Learning rate for w2.
        n_iter (int): Number of iterations.
        tol (float): Tolerance for duality gap.
        sparsity (int or None): Sparsity level for the model.
        fit_intercept (bool): Whether to fit an intercept.
    """

    def __init__(self, mu=0.01, V1=100, V2=100,
                 eta_alpha='auto', eta_w1='auto', eta_beta='auto', eta_w2='auto',
                 n_iter=2000, tol=1e-2, sparsity=None, fit_intercept=True):
        """
        Initialize the _SparseLinear2AdversarialGMM.

        Parameters:
            mu (float): Regularization parameter.
            V1 (int): Budget parameter for the first stage.
            V2 (int): Budget parameter for the second stage.
            eta_alpha (str or float): Learning rate for alpha.
            eta_w1 (str or float): Learning rate for w1.
            eta_beta (str or float): Learning rate for beta.
            eta_w2 (str or float): Learning rate for w2.
            n_iter (int): Number of iterations.
            tol (float): Tolerance for duality gap.
            sparsity (int or None): Sparsity level for the model.
            fit_intercept (bool): Whether to fit an intercept.
        """
        self.V1 = V1
        self.V2 = V2
        self.mu = mu
        self.eta_alpha = eta_alpha
        self.eta_w1 = eta_w1
        self.eta_beta = eta_beta
        self.eta_w2 = eta_w2
        self.n_iter = n_iter
        self.tol = tol
        self.sparsity = sparsity
        self.fit_intercept = fit_intercept

    def weighted_mean(self, arr, weights, axis=0):
        """
        Compute the weighted mean of an array.

        Parameters:
            arr (array-like): Input array.
            weights (array-like): Weights for computing the mean.
            axis (int, optional): Axis along which the mean is computed.

        Returns:
            array: Weighted mean.
        """
        weights = np.array(weights)
        if arr.ndim == 1 or axis is None:
            return np.sum(arr * weights) / np.sum(weights)
        else:
            return np.sum(arr * weights[:, np.newaxis], axis=axis) / np.sum(weights)

    def _check_input(self, A, B, C, D, Y, W):
        """
        Check and preprocess input arrays.

        Parameters:
            A (array-like): Covariates for the first stage.
            B (array-like): Covariates for the second stage.
            C (array-like): Instrumental variables for the second stage.
            D (array-like): Instrumental variables for the first stage.
            Y (array-like): Outcomes.
            W (array-like): Weights.

        Returns:
            tuple: Processed A, B, C, D, Y, W.
        """
        if self.fit_intercept:
            A = np.hstack([np.ones((A.shape[0], 1)), A])
            B = np.hstack([np.ones((B.shape[0], 1)), B])
            C = np.hstack([np.ones((C.shape[0], 1)), C])
            D = np.hstack([np.ones((D.shape[0], 1)), D])
        return A, B, C, D, Y.flatten(), W.reshape(-1, 1)

    def predict(self, B, *args):
        """
        Predict using the fitted model.

        Parameters:
            B (array-like): Covariates for the second stage.
            args (array-like): Optional covariates for the first stage.

        Returns:
            array: Predicted values for the second stage.
            If args are provided, also returns predicted values for the first stage.
        """
        if len(args) == 0:
            if self.fit_intercept:
                B = np.hstack([np.ones((B.shape[0], 1)), B])
            return np.dot(B, self.beta_)
        elif len(args) == 1:
            A = args[0]
            if self.fit_intercept:
                B = np.hstack([np.ones((B.shape[0], 1)), B])
                A = np.hstack([np.ones((A.shape[0], 1)), A])
            return np.dot(B, self.beta_), np.dot(A, self.alpha_)
        else:
            raise ValueError("predict expects at most two parameters, B_test and optionally A_test")

    @property
    def coef(self):
        return self.beta_[1:] if self.fit_intercept else self.beta_

    @property
    def intercept(self):
        return self.beta_[0] if self.fit_intercept else 0


class sparse2_l1vsl1(_SparseLinear2AdversarialGMM):
    r"""
    Sparse Linear NPIV estimator using :math:`\ell_1-\ell_1` optimization for nested NPIV.

    This class solves the high-dimensional sparse linear problem using :math:`\ell_1` relaxations for the minimax optimization problem in a nested NPIV setting.

    Parameters:
        Same as `_SparseLinear2AdversarialGMM`.
    """

    def _check_duality_gap(self, A, B, C, D, Y, W):
        r"""
        Calculate the duality gap to certify convergence of the algorithm.

        The ensembles can be thought of as primal and dual solutions, and the duality gap can be used as a certificate for convergence of the algorithm.

        Parameters:
            A (array-like): Covariates for the first stage.
            B (array-like): Covariates for the second stage.
            C (array-like): Instrumental variables for the second stage.
            D (array-like): Instrumental variables for the first stage.
            Y (array-like): Outcomes.
            W (array-like): Weights.

        Returns:
            bool: True if the duality gap is below the tolerance level, indicating convergence.
        """
        self.max_response_loss_ = np.linalg.norm(self.weighted_mean(D * (Y - np.dot(A, self.alpha_)).reshape(-1, 1), self.weights1, axis=0), ord=np.inf)\
            + np.linalg.norm(self.weighted_mean(C * (np.dot(W * A, self.alpha_) - np.dot(B, self.beta_)).reshape(-1, 1), self.weights2, axis=0), ord=np.inf)\
            + self.mu * np.linalg.norm(self.alpha_, ord=1) + self.mu * np.linalg.norm(self.beta_, ord=1)
              
        self.min_response_loss_ = self.weighted_mean(Y * np.dot(D, self.w1_), self.weights1)\
            + self.V1 * np.clip(self.mu - 2 * np.linalg.norm(self.weighted_mean(A * np.dot(D, self.w1_).reshape(-1, 1),
                                                                            self.weights1, axis=0),
                                                                    ord=np.inf)
                                        + 2 * np.linalg.norm(self.weighted_mean(W * A * np.dot(C, self.w2_).reshape(-1, 1),
                                                                self.weights2, axis=0),
                                                            ord=np.inf),
                                                   -np.inf, 0)\
            + self.V2 * np.clip(self.mu - 2 * np.linalg.norm(self.weighted_mean(B * np.dot(C, self.w2_).reshape(-1, 1),
                                                                self.weights2, axis=0),
                                                            ord=np.inf),
                                                   -np.inf, 0)
        
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, A, B, C, D, Y, W):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (A.shape[0])**(2 / 3))
            filt = (np.abs(self.alpha_) < thresh)
            self.alpha_[filt] = 0
            thresh = 1 / (self.sparsity * (B.shape[0])**(2 / 3))
            filt = (np.abs(self.beta_) < thresh)
            self.beta_[filt] = 0
        self._check_duality_gap(A, B, C, D, Y, W)

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        """
        Fit the model.

        Parameters:
            A (array-like): Covariates for the first stage.
            B (array-like): Covariates for the second stage.
            C (array-like): Instrumental variables for the second stage.
            D (array-like): Instrumental variables for the first stage.
            Y (array-like): Outcomes.
            W (array-like, optional): Weights. Defaults to None.
            subsetted (bool, optional): Whether to use subsets. Defaults to False.
            subset_ind1 (array-like, optional): Subset indices for the first stage. Required if subsetted is True.
            subset_ind2 (array-like, optional): Subset indices for the second stage. Defaults to None.

        Returns:
            self: Fitted estimator.
        """
        W = np.ones(Y.shape[0]) if W is None else W 
        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W) 
        self.weights1 = np.ones(Y.shape[0])
        self.weights2 = np.ones(Y.shape[0])
        if subsetted:
            if subset_ind1 is None:
                raise ValueError("subset_ind1 must be provided when subsetted is True")
            if len(subset_ind1) != len(Y):
                raise ValueError("subset_ind1 must have the same length as Y")
            ind1 = np.where(subset_ind1==0)[0] 
            ind2 = np.where(subset_ind2==0)[0] if subset_ind2 is not None else np.where(subset_ind1==1)[0]  
            self.weights1[ind1] = 0
            self.weights2[ind2] = 0
         
        T = self.n_iter
        d_a = A.shape[1]
        d_b = B.shape[1]
        d_c = C.shape[1]
        d_d = D.shape[1]
        n = A.shape[0]
        V1 = self.V1
        V2 = self.V2
        eta_alpha = .5 if self.eta_alpha == 'auto' else self.eta_alpha
        eta_beta = .5 if self.eta_beta == 'auto' else self.eta_beta
        eta_w1 = .5 if self.eta_w1 == 'auto' else self.eta_w1
        eta_w2 = .5 if self.eta_w2 == 'auto' else self.eta_w2
        mu = self.mu

        yd = self.weighted_mean(Y.reshape(-1, 1) * D, self.weights1, axis=0)
        if d_a * d_d < n**2:
            ad = self.weighted_mean(cross_product(A, D), self.weights1,
                         axis=0).reshape(d_d, d_a).T
        if d_a * d_c < n**2:
            ac = self.weighted_mean(cross_product(W*A, C), self.weights2,
                         axis=0).reshape(d_c, d_a).T
        if d_b * d_c < n**2:
            bc = self.weighted_mean(cross_product(B, C), self.weights2,
                         axis=0).reshape(d_c, d_b).T
        
        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                alpha = np.ones(2 * d_a) * V1 / (2 * d_a)
                beta = np.ones(2 * d_b) * V1 / (2 * d_b)
                w1 = np.ones(2 * d_d) / (2 * d_d)
                w2 = np.ones(2 * d_c) / (2 * d_c)
                alpha_acc = np.ones(2 * d_a) * V1 / (2 * d_a)
                beta_acc = np.ones(2 * d_b) * V1 / (2 * d_b)
                w1_acc = np.ones(2 * d_d) / (2 * d_d)
                w2_acc = np.ones(2 * d_c) / (2 * d_c)
                res1 = np.zeros(2 * d_d)
                res2 = np.zeros(2 * d_c)
                res1_pre = np.zeros(2 * d_d)
                res2_pre = np.zeros(2 * d_c)
                cors1 = 0
                cors2 = 0

            # quantities for updating alpha
            if d_a * d_d < n**2:
                cors1_t = - ad @ (w1[:d_d] - w1[d_d:])
            else:
                test_fn = np.dot(D, w1[:d_d] - w1[d_d:]).reshape(-1, 1)
                cors1_t = - self.weighted_mean(test_fn * A, self.weights1, axis=0)
            if d_a * d_c < n**2:
                cors1_t += ac @ (w2[:d_c] - w2[d_c:])
            else:
                test_fn = np.dot(C, w2[:d_c] - w2[d_c:]).reshape(-1, 1)
                cors1_t += self.weighted_mean(test_fn * A * W, self.weights2, axis=0)
            cors1 += cors1_t

            # quantities for updating beta
            if d_b * d_c < n**2:
                cors2_t = - bc @ (w2[:d_c] - w2[d_c:])
            else:
                test_fn = np.dot(C, w2[:d_c] - w2[d_c:]).reshape(-1, 1)
                cors2_t = - self.weighted_mean(test_fn * B, self.weights2, axis=0)
            cors2 += cors2_t

            # quantities for updating w1
            if d_a * d_d < n**2:
                res1[:d_d] = yd - (alpha[:d_a] - alpha[d_a:]).T @ ad
            else:
                pred_fn = np.dot(A, alpha[:d_a] - alpha[d_a:]).reshape(-1, 1)
                res1[:d_d] = yd - self.weighted_mean(D * pred_fn, self.weights1, axis=0) 
            res1[d_d:] = - res1[:d_d]

            # quantities for updating w2
            if d_c * d_a < n**2:
                res2[:d_c] = (alpha[:d_a] - alpha[d_a:]).T @ ac 
            else:
                pred_fn = np.dot(A * W, alpha[:d_a] - alpha[d_a:]).reshape(-1, 1)
                res2[:d_c] = self.weighted_mean(C * pred_fn, self.weights2, axis=0)
            if d_c * d_b < n**2:
                res2[:d_c] -= (beta[:d_b] - beta[d_b:]).T @ bc
            else:
                pred_fn = np.dot(B, beta[:d_b] - beta[d_b:]).reshape(-1, 1)
                res2[:d_c] -= self.weighted_mean(C * pred_fn, self.weights2, axis=0)
            res2[d_c:] = - res2[:d_c]

            # update alpha
            alpha[:d_a] = np.exp(-1 - eta_alpha *
                                 (cors1 + cors1_t + (t + 1) * mu))
            alpha[d_a:] = np.exp(-1 - eta_alpha *
                                 (- cors1 - cors1_t + (t + 1) * mu))
            normalization = np.sum(alpha)
            if normalization > V1:
                alpha[:] = alpha * V1 / normalization

            # update beta
            beta[:d_b] = np.exp(-1 - eta_beta *
                                (cors2 + cors2_t + (t + 1) * mu))
            beta[d_b:] = np.exp(-1 - eta_beta *
                                (- cors2 - cors2_t + (t + 1) * mu))
            normalization = np.sum(beta)
            if normalization > V2:
                beta[:] = beta * V2 / normalization

            # update w1
            w1[:] = w1 * np.exp(2 * eta_w1 * res1 - eta_w1 * res1_pre)
            w1[:] = w1 / np.sum(w1)

            # update w2
            w2[:] = w2 * np.exp(2 * eta_w2 * res2 - eta_w2 * res2_pre)
            w2[:] = w2 / np.sum(w2)

            alpha_acc = alpha_acc * (t - 1) / t + alpha / t
            beta_acc = beta_acc * (t - 1) / t + beta / t
            w1_acc = w1_acc * (t - 1) / t + w1 / t
            w2_acc = w2_acc * (t - 1) / t + w2 / t

            res1_pre[:] = res1
            res2_pre[:] = res2

            if t % 50 == 0:
                self.alpha_ = alpha_acc[:d_a] - alpha_acc[d_a:]
                self.beta_ = beta_acc[:d_b] - beta_acc[d_b:]
                self.w1_ = w1_acc[:d_d] - w1_acc[d_d:]
                self.w2_ = w2_acc[:d_c] - w2_acc[d_c:]
                if self._check_duality_gap(A, B, C, D, Y, W):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_alpha /= 2
                    eta_beta /= 2
                    eta_w1 /= 2
                    eta_w2 /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_alpha /= 1.01
                    eta_beta /= 1.01
                    eta_w1 /= 1.01
                    eta_w2 /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.alpha_ = alpha_acc[:d_a] - alpha_acc[d_a:]
        self.beta_ = beta_acc[:d_b] - beta_acc[d_b:]
        self.w1_ = w1_acc[:d_d] - w1_acc[d_d:]
        self.w2_ = w2_acc[:d_c] - w2_acc[d_c]
        
        self._post_process(A, B, C, D, Y, W)
        return self


class sparse2_ridge_l1vsl1(_SparseLinear2AdversarialGMM):
    r"""
    Sparse Ridge NPIV estimator using :math:`\ell_1-\ell_1` optimization for nested NPIV.

    This class solves the high-dimensional sparse ridge problem using :math:`\ell_1` relaxations for the minimax optimization problem in a nested NPIV setting.

    Parameters:
        Same as `_SparseLinear2AdversarialGMM`.
    """

    def _check_duality_gap(self, A, B, C, D, Y, W):
        """
        Calculate the duality gap to certify convergence of the algorithm.

        The ensembles can be thought of as primal and dual solutions, and the duality gap can be used as a certificate for convergence of the algorithm.

        Parameters:
            A (array-like): Covariates for the first stage.
            B (array-like): Covariates for the second stage.
            C (array-like): Instrumental variables for the second stage.
            D (array-like): Instrumental variables for the first stage.
            Y (array-like): Outcomes.
            W (array-like): Weights.

        Returns:
            bool: True if the duality gap is below the tolerance level, indicating convergence.
        """
        self.max_response_loss_ = np.linalg.norm(self.weighted_mean(D * (Y - np.dot(A, self.alpha_)).reshape(-1, 1), self.weights1, axis=0), ord=np.inf)\
            + np.linalg.norm(self.weighted_mean(C * (np.dot(A * W, self.alpha_) - np.dot(B, self.beta_)).reshape(-1, 1), self.weights2, axis=0), ord=np.inf)\
            + self.mu * self.alpha_.T @ self.aa @ self.alpha_ + self.mu * self.beta_.T @ self.bb @ self.beta_
            
        self.min_response_loss_ = 2 * self.weighted_mean(Y * np.dot(D, self.w1_), self.weights1)\
            - (self.msvp_a/self.mu) * np.linalg.norm(self.weighted_mean(A * np.dot(D, self.w1_).reshape(-1, 1), self.weights1,
                                                            axis=0)
                                                    - self.weighted_mean(W * A * np.dot(C, self.w2_).reshape(-1, 1), self.weights2,
                                                            axis=0),
                                                    ord=2)\
            - (self.msvp_b/self.mu) * np.linalg.norm(self.weighted_mean(B * np.dot(C, self.w2_).reshape(-1, 1), self.weights2,
                                                            axis=0),
                                                    ord=2)
        
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, A, B, C, D, Y, W):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (A.shape[0])**(2 / 3))
            filt = (np.abs(self.alpha_) < thresh)
            self.alpha_[filt] = 0
            thresh = 1 / (self.sparsity * (B.shape[0])**(2 / 3))
            filt = (np.abs(self.beta_) < thresh)
            self.beta_[filt] = 0
        self._check_duality_gap(A, B, C, D, Y, W)

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        """
        Fit the model.

        Parameters:
            A (array-like): Covariates for the first stage.
            B (array-like): Covariates for the second stage.
            C (array-like): Instrumental variables for the second stage.
            D (array-like): Instrumental variables for the first stage.
            Y (array-like): Outcomes.
            W (array-like, optional): Weights. Defaults to None.
            subsetted (bool, optional): Whether to use subsets. Defaults to False.
            subset_ind1 (array-like, optional): Subset indices for the first stage. Required if subsetted is True.
            subset_ind2 (array-like, optional): Subset indices for the second stage. Defaults to None.

        Returns:
            self: Fitted estimator.
        """
        W = np.ones(Y.shape[0]) if W is None else W
        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W) 
        self.weights1 = np.ones(Y.shape[0])
        self.weights2 = np.ones(Y.shape[0])
        if subsetted:
            if subset_ind1 is None:
                raise ValueError("subset_ind1 must be provided when subsetted is True")
            if len(subset_ind1) != len(Y):
                raise ValueError("subset_ind1 must have the same length as Y")
            ind1 = np.where(subset_ind1==0)[0] 
            ind2 = np.where(subset_ind2==0)[0] if subset_ind2 is not None else np.where(subset_ind1==1)[0]  
            self.weights1[ind1] = 0
            self.weights2[ind2] = 0
        
        T = self.n_iter
        d_a = A.shape[1]
        d_b = B.shape[1]
        d_c = C.shape[1]
        d_d = D.shape[1]
        n = A.shape[0]
        V1 = self.V1
        V2 = self.V2
        eta_alpha = .5 if self.eta_alpha == 'auto' else self.eta_alpha
        eta_beta = .5 if self.eta_beta == 'auto' else self.eta_beta
        eta_w1 = .5 if self.eta_w1 == 'auto' else self.eta_w1
        eta_w2 = .5 if self.eta_w2 == 'auto' else self.eta_w2
        mu = self.mu

        yd = self.weighted_mean(Y.reshape(-1, 1) * D, self.weights1, axis=0)
        aa = np.mean(cross_product(A, A), axis=0).reshape(d_a, d_a).T
        self.aa = aa
        Sigma = np.linalg.svd(aa, compute_uv=False)
        sigma_min = np.min(Sigma[Sigma > 1e-10])  
        self.msvp_a = 1 / sigma_min

        bb = np.mean(cross_product(B, B), axis=0).reshape(d_b, d_b).T
        self.bb = bb
        Sigma = np.linalg.svd(bb, compute_uv=False)
        sigma_min = np.min(Sigma[Sigma > 1e-10])
        self.msvp_b = 1 / sigma_min

        if d_a * d_d < n**2:
            ad = self.weighted_mean(cross_product(A, D), self.weights1,
                         axis=0).reshape(d_d, d_a).T
        if d_a * d_c < n**2:
            ac = self.weighted_mean(cross_product(W*A, C), self.weights2,
                         axis=0).reshape(d_c, d_a).T
        if d_b * d_c < n**2:
            bc = self.weighted_mean(cross_product(B, C), self.weights2,
                         axis=0).reshape(d_c, d_b).T
        
        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                alpha = np.ones(2 * d_a) * V1 / (2 * d_a)
                beta = np.ones(2 * d_b) * V1 / (2 * d_b)
                w1 = np.ones(2 * d_d) / (2 * d_d)
                w2 = np.ones(2 * d_c) / (2 * d_c)
                alpha_acc = np.ones(2 * d_a) * V1 / (2 * d_a)
                beta_acc = np.ones(2 * d_b) * V1 / (2 * d_b)
                w1_acc = np.ones(2 * d_d) / (2 * d_d)
                w2_acc = np.ones(2 * d_c) / (2 * d_c)
                res1 = np.zeros(2 * d_d)
                res2 = np.zeros(2 * d_c)
                res1_pre = np.zeros(2 * d_d)
                res2_pre = np.zeros(2 * d_c)
                cors1 = 0
                cors2 = 0

            # quantities for updating alpha
            if d_a * d_d < n**2:
                cors1_t = - ad @ (w1[:d_d] - w1[d_d:]) + mu * aa @ (alpha[:d_a] - alpha[d_a:])
            else:
                test_fn = np.dot(D, w1[:d_d] - w1[d_d:]).reshape(-1, 1)
                cors1_t = - self.weighted_mean(test_fn * A, self.weights1, axis=0) + mu * aa @ (alpha[:d_a] - alpha[d_a:])
            if d_a * d_c < n**2:
                cors1_t += ac @ (w2[:d_c] - w2[d_c:])
            else:
                test_fn = np.dot(C, w2[:d_c] - w2[d_c:]).reshape(-1, 1)
                cors1_t += self.weighted_mean(test_fn * A * W, self.weights2, axis=0)
            cors1 += cors1_t

            # quantities for updating beta
            if d_b * d_c < n**2:
                cors2_t = - bc @ (w2[:d_c] - w2[d_c:]) + mu * bb @ (beta[:d_b] - beta[d_b:])
            else:
                test_fn = np.dot(C, w2[:d_c] - w2[d_c:]).reshape(-1, 1)
                cors2_t = - self.weighted_mean(test_fn * B, self.weights2, axis=0) + mu * bb @ (beta[:d_b] - beta[d_b:])
            cors2 += cors2_t

            # quantities for updating w1
            if d_a * d_d < n**2:
                res1[:d_d] = yd - (alpha[:d_a] - alpha[d_a:]).T @ ad
            else:
                pred_fn = np.dot(A, alpha[:d_a] - alpha[d_a:]).reshape(-1, 1)
                res1[:d_d] = yd - self.weighted_mean(D * pred_fn, self.weights1, axis=0) 
            res1[d_d:] = - res1[:d_d]

            # quantities for updating w2
            if d_c * d_a < n**2:
                res2[:d_c] = (alpha[:d_a] - alpha[d_a:]).T @ ac 
            else:
                pred_fn = np.dot(A * W, alpha[:d_a] - alpha[d_a:]).reshape(-1, 1)
                res2[:d_c] = self.weighted_mean(C * pred_fn, self.weights2, axis=0)
            if d_c * d_b < n**2:
                res2[:d_c] -= (beta[:d_b] - beta[d_b:]).T @ bc
            else:
                pred_fn = np.dot(B, beta[:d_b] - beta[d_b:]).reshape(-1, 1)
                res2[:d_c] -= self.weighted_mean(C * pred_fn, self.weights2, axis=0)
            res2[d_c:] = - res2[:d_c]

            # update alpha
            alpha[:d_a] = np.exp(-1 - eta_alpha * (cors1 + cors1_t))
            alpha[d_a:] = np.exp(-1 - eta_alpha * (- cors1 - cors1_t))
            normalization = np.sum(alpha)
            if normalization > V1:
                alpha[:] = alpha * V1 / normalization

            # update beta
            beta[:d_b] = np.exp(-1 - eta_beta * (cors2 + cors2_t))
            beta[d_b:] = np.exp(-1 - eta_beta * (- cors2 - cors2_t))
            normalization = np.sum(beta)
            if normalization > V2:
                beta[:] = beta * V2 / normalization

            # update w1
            w1[:] = w1 * np.exp(2 * eta_w1 * res1 - eta_w1 * res1_pre)
            w1[:] = w1 / np.sum(w1)

            # update w2
            w2[:] = w2 * np.exp(2 * eta_w2 * res2 - eta_w2 * res2_pre)
            w2[:] = w2 / np.sum(w2)

            alpha_acc = alpha_acc * (t - 1) / t + alpha / t
            beta_acc = beta_acc * (t - 1) / t + beta / t
            w1_acc = w1_acc * (t - 1) / t + w1 / t
            w2_acc = w2_acc * (t - 1) / t + w2 / t

            res1_pre[:] = res1
            res2_pre[:] = res2

            if t % 50 == 0:
                self.alpha_ = alpha_acc[:d_a] - alpha_acc[d_a:]
                self.beta_ = beta_acc[:d_b] - beta_acc[d_b:]
                self.w1_ = w1_acc[:d_d] - w1_acc[d_d:]
                self.w2_ = w2_acc[:d_c] - w2_acc[d_c:]
                if self._check_duality_gap(A, B, C, D, Y, W):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_alpha /= 2
                    eta_beta /= 2
                    eta_w1 /= 2
                    eta_w2 /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_alpha /= 1.01
                    eta_beta /= 1.01
                    eta_w1 /= 1.01
                    eta_w2 /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.alpha_ = alpha_acc[:d_a] - alpha_acc[d_a:]
        self.beta_ = beta_acc[:d_b] - beta_acc[d_b:]
        self.w1_ = w1_acc[:d_d] - w1_acc[d_d:]
        self.w2_ = w2_acc[:d_c] - w2_acc[d_c]
        
        self._post_process(A, B, C, D, Y, W)
        return self
