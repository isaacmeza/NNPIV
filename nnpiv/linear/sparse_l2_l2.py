"""
This module provides implementations of sparse linear NPIV estimators with L2 norm regularization.

Classes:
    _SparseLinearAdversarialGMM: Base class for sparse linear adversarial GMM.
    sparse_l2vsl2: Sparse Linear NPIV estimator using $\ell_2-\ell_2$ optimization.
    sparse_ridge_l2vsl2: Sparse Ridge NPIV estimator using $\ell_2-\ell_2$ optimization.
"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from nnpiv.linear.utilities import cross_product


class _SparseLinearAdversarialGMM:
    """
    Base class for sparse linear adversarial GMM.

    This class implements common functionality for sparse linear models using adversarial GMM.

    Parameters:
        lambda_theta (float): Regularization parameter.
        B (int): Budget parameter.
        eta_theta (str or float): Learning rate for theta.
        eta_w (str or float): Learning rate for w.
        n_iter (int): Number of iterations.
        tol (float): Tolerance for duality gap.
        sparsity (int or None): Sparsity level for the model.
        fit_intercept (bool): Whether to fit an intercept.

    Methods:
        fit(Z, X, Y): Fit the model.
        predict(X): Predict using the fitted model.
    """

    def __init__(self, lambda_theta=0.01, B=100, eta_theta='auto', eta_w='auto',
                 n_iter=2000, tol=1e-2, sparsity=None, fit_intercept=True):
        """
        Initialize the sparse linear adversarial GMM model.

        Parameters:
            lambda_theta (float, optional): Regularization parameter. Defaults to 0.01.
            B (int, optional): Budget parameter. Defaults to 100.
            eta_theta (str or float, optional): Learning rate for theta. Defaults to 'auto'.
            eta_w (str or float, optional): Learning rate for w. Defaults to 'auto'.
            n_iter (int, optional): Number of iterations. Defaults to 2000.
            tol (float, optional): Tolerance for duality gap. Defaults to 1e-2.
            sparsity (int or None, optional): Sparsity level for the model. Defaults to None.
            fit_intercept (bool, optional): Whether to fit an intercept. Defaults to True.
        """
        self.B = B
        self.lambda_theta = lambda_theta
        self.eta_theta = eta_theta
        self.eta_w = eta_w
        self.n_iter = n_iter
        self.tol = tol
        self.sparsity = sparsity
        self.fit_intercept = fit_intercept

    def _check_input(self, Z, X, Y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            Z = np.hstack([np.ones((X.shape[0], 1)), Z])
        return Z, X, Y.flatten()

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters:
            X (array-like): Covariates.

        Returns:
            array: Predicted values.
        """
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.coef_)

    @property
    def coef(self):
        return self.coef_[1:] if self.fit_intercept else self.coef_

    @property
    def intercept(self):
        return self.coef_[0] if self.fit_intercept else 0


class sparse_l2vsl2(_SparseLinearAdversarialGMM):
    """
    Sparse Linear NPIV estimator using $\ell_2-\ell_2$ optimization.

    This class solves the high-dimensional sparse linear problem using $\ell_2$ relaxations for the minimax optimization problem.

    Parameters:
        Same as `_SparseLinearAdversarialGMM`.
    """

    def _check_duality_gap(self, Z, X, Y):
        """
        Check the duality gap to monitor convergence.

        The ensembles can be thought of as primal and dual solutions, and the duality gap can be used as a certificate for convergence of the algorithm.

        Parameters:
            Z (array-like): Instrumental variables.
            X (array-like): Covariates.
            Y (array-like): Outcomes.

        Returns:
            bool: True if the duality gap is less than the tolerance, otherwise False.
        """
        self.max_response_loss_ = np.linalg.norm(
            np.mean(Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0), ord=2)\
            + self.lambda_theta * np.linalg.norm(self.coef_, ord=2)**2
        self.min_response_loss_ = self.B * np.clip(self.lambda_theta
                                                   - np.linalg.norm(np.mean(X * np.dot(Z, self.w_).reshape(-1, 1),
                                                                            axis=0),
                                                                    ord=2),
                                                   -np.inf, 0)\
            - np.mean(Y * np.dot(Z, self.w_))
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, Z, X, Y):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (X.shape[0])**(2 / 3))
            filt = (np.abs(self.coef_) < thresh)
            self.coef_[filt] = 0
        self.max_violation_ = np.linalg.norm(
            np.mean(Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0), ord=np.inf)
        self._check_duality_gap(Z, X, Y)

    def fit(self, Z, X, Y):
        """
        Fit the model.

        Parameters:
            Z (array-like): Instrumental variables.
            X (array-like): Covariates.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        n = X.shape[0]
        B = self.B
        eta_theta = np.sqrt(
            np.log(d_x + 1) / T) if self.eta_theta == 'auto' else self.eta_theta
        eta_w = np.sqrt(
            np.log(d_z + 1) / T) if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yz = np.mean(Y.reshape(-1, 1) * Z, axis=0)
        if d_x * d_z < n**2:
            xz = np.mean(cross_product(X, Z),
                         axis=0).reshape(d_z, d_x).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.zeros(d_x)
                theta_acc = np.zeros(d_x)
                w = np.zeros(d_z)
                w_acc = np.zeros(d_z)
                res = np.zeros(d_z)
                res_pre = np.zeros(d_z)
                cors = np.zeros(d_x)
                cors_pre = np.zeros(d_x)

            # quantities for updating theta
            if d_x * d_z < n**2:
                cors[:] = xz @ w + lambda_theta * theta
            else:
                test_fn = np.dot(Z, w).reshape(-1, 1)
                cors[:] = np.mean(test_fn * X, axis=0) + lambda_theta * theta
            
            # quantities for updating w
            if d_x * d_z < n**2:
                res[:] = theta.T @ xz - yz
            else:
                pred_fn = np.dot(X, theta).reshape(-1, 1)
                res[:] = np.mean(Z * pred_fn, axis=0) - yz

            # update theta
            theta[:] = theta - 2 * eta_theta * cors + eta_theta * cors_pre
            normalization = np.linalg.norm(theta, ord=2)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w + 2 * eta_w * res - eta_w * res_pre
            norm_w = np.linalg.norm(w, ord=2)
            w[:] = w / norm_w if norm_w > 1 else w

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res
            cors_pre[:] = cors

            if t % 50 == 0:
                self.coef_ = theta_acc
                self.w_ = w_acc
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.coef_ = theta_acc
        self.w_ = w_acc

        self._post_process(Z, X, Y)

        return self
    

class sparse_ridge_l2vsl2(_SparseLinearAdversarialGMM):
    """
    Sparse Ridge NPIV estimator using $\ell_2-\ell_2$ optimization.

    This class solves the high-dimensional sparse ridge problem using $\ell_2$ relaxations for the minimax optimization problem.

    Parameters:
        Same as `_SparseLinearAdversarialGMM`.
    """

    def _check_duality_gap(self, Z, X, Y):
        """
        Check the duality gap to monitor convergence.

        The ensembles can be thought of as primal and dual solutions, and the duality gap can be used as a certificate for convergence of the algorithm.

        Parameters:
            Z (array-like): Instrumental variables.
            X (array-like): Covariates.
            Y (array-like): Outcomes.

        Returns:
            bool: True if the duality gap is less than the tolerance, otherwise False.
        """
        self.max_response_loss_ = np.linalg.norm(
            np.mean(Z * (Y - np.dot(X, self.coef_)).reshape(-1, 1), axis=0), ord=2)\
            + self.lambda_theta * self.coef_.T @ self.xx @ self.coef_
        
        self.min_response_loss_ =  2 * np.mean(Y * np.dot(Z, self.w_))\
            - (self.msvp/self.lambda_theta) * np.linalg.norm(np.mean(X * np.dot(Z, self.w_).reshape(-1, 1),
                                                                            axis=0),
                                                            ord=2)
                                                   
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, Z, X, Y):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (X.shape[0])**(2 / 3))
            filt = (np.abs(self.coef_) < thresh)
            self.coef_[filt] = 0
        self.max_violation_ = np.linalg.norm(
            np.mean(Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0), ord=np.inf)
        self._check_duality_gap(Z, X, Y)

    def fit(self, Z, X, Y):
        """
        Fit the model.

        Parameters:
            Z (array-like): Instrumental variables.
            X (array-like): Covariates.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        n = X.shape[0]
        B = self.B
        eta_theta = np.sqrt(
            np.log(d_x + 1) / T) if self.eta_theta == 'auto' else self.eta_theta
        eta_w = np.sqrt(
            np.log(d_z + 1) / T) if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yz = np.mean(Y.reshape(-1, 1) * Z, axis=0)
        xx = np.mean(cross_product(X, X),
                        axis=0).reshape(d_x, d_x).T
        self.xx = xx
        # Perform SVD on E_n[xx^T]
        Sigma = np.linalg.svd(xx, compute_uv=False)
        # Find the minimum non-zero singular value
        sigma_min = np.min(Sigma[Sigma > 1e-10])  
        # Compute the maximum singular value of the pseudoinverse
        self.msvp = 1 / sigma_min

        if d_x * d_z < n**2:
            xz = np.mean(cross_product(X, Z),
                         axis=0).reshape(d_z, d_x).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.zeros(d_x)
                theta_acc = np.zeros(d_x)
                w = np.zeros(d_z)
                w_acc = np.zeros(d_z)
                res = np.zeros(d_z)
                res_pre = np.zeros(d_z)
                cors = np.zeros(d_x)
                cors_pre = np.zeros(d_x)

            # quantities for updating theta
            if d_x * d_z < n**2:
                cors[:] = xz @ w + lambda_theta * xx @ theta
            else:
                test_fn = np.dot(Z, w).reshape(-1, 1)
                cors[:] = np.mean(test_fn * X, axis=0) + lambda_theta * xx @ theta
            
            # quantities for updating w
            if d_x * d_z < n**2:
                res[:] = theta.T @ xz - yz
            else:
                pred_fn = np.dot(X, theta).reshape(-1, 1)
                res[:] = np.mean(Z * pred_fn, axis=0) - yz

            # update theta
            theta[:] = theta - 2 * eta_theta * cors + eta_theta * cors_pre
            normalization = np.linalg.norm(theta, ord=2)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w + 2 * eta_w * res - eta_w * res_pre
            norm_w = np.linalg.norm(w, ord=2)
            w[:] = w / norm_w if norm_w > 1 else w

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res
            cors_pre[:] = cors

            if t % 50 == 0:
                self.coef_ = theta_acc
                self.w_ = w_acc
                if self._check_duality_gap(Z, X, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.coef_ = theta_acc
        self.w_ = w_acc

        self._post_process(Z, X, Y)

        return self
