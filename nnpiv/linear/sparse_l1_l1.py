r"""Sparse linear NPIV estimators with :math:`\ell_1` learner and critic balls.

Classes
-------
_SparseLinearAdversarialGMM
    Base class for sparse linear adversarial GMM.
sparse_l1vsl1
    Linear-critic estimator with an :math:`\ell_1` coefficient penalty.
sparse_ridge_l1vsl1
    Linear-critic estimator with an empirical-L2 learner penalty.
"""

# Licensed under the MIT License.

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from nnpiv.linear.utilities import cross_product, quadratic_min_l1


class _SparseLinearAdversarialGMM:
    r"""
    Base class for sparse linear adversarial GMM.

    This class implements common functionality for sparse linear models using adversarial GMM.

    Parameters:
        lambda_theta (float): Nonnegative learner regularization coefficient.
        B (float): Nonnegative :math:`\ell_1` learner radius.
        eta_theta (str or float): Learning rate for theta.
        eta_w (str or float): Learning rate for w.
        n_iter (int): Number of iterations.
        tol (float): Tolerance for duality gap.
        sparsity (int or None): Sparsity level for the model.
        fit_intercept (bool): Whether to fit an intercept.
    """

    def __init__(self, lambda_theta=0.01, B=100, eta_theta='auto', eta_w='auto',
                 n_iter=2000, tol=1e-2, sparsity=None, fit_intercept=True):
        self.B = B
        self.lambda_theta = lambda_theta
        self.eta_theta = eta_theta
        self.eta_w = eta_w
        self.n_iter = n_iter
        self.tol = tol
        self.sparsity = sparsity
        self.fit_intercept = fit_intercept

    def _validate_parameters(self):
        if self.n_iter < 2:
            raise ValueError("n_iter must be at least 2")
        if self.B < 0:
            raise ValueError("B must be nonnegative")
        if self.lambda_theta < 0:
            raise ValueError("lambda_theta must be nonnegative")

    def _check_input(self, Z, X, Y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            Z = np.hstack([np.ones((X.shape[0], 1)), Z])
        return Z, X, Y.flatten()

    def predict(self, X):
        """
        Predict.

        Parameters:
            X (array-like): Feature or treatment matrix.

        Returns:
            ndarray: Fitted linear function evaluated at ``X``.
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


class sparse_l1vsl1(_SparseLinearAdversarialGMM):
    r"""Sparse linear NPIV with :math:`\ell_1` learner regularization.

    With :math:`m(\alpha)=\mathbb E_n[Z(X^\top\alpha-Y)]`, the
    implemented linear-critic game is

    .. math::

        \min_{\|\alpha\|_1\leq B}\max_{\|\theta\|_1\leq 1}
        \theta^\top m(\alpha)+\lambda\|\alpha\|_1.

    OFTRL is applied to positive-negative lifts of both players. The returned
    coefficients average the feasible iterates, and ``duality_gap_`` evaluates
    the corresponding exact constrained best-response gap.

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
            np.mean(Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0), ord=np.inf)\
            + self.lambda_theta * np.linalg.norm(self.coef_, ord=1)
        self.min_response_loss_ = self.B * np.clip(self.lambda_theta
                                                   - np.linalg.norm(np.mean(X * np.dot(Z, self.w_).reshape(-1, 1),
                                                                            axis=0),
                                                                    ord=np.inf),
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
        self._validate_parameters()
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        n = X.shape[0]
        B = self.B
        eta_theta = .5 if self.eta_theta == 'auto' else self.eta_theta
        eta_w = .5 if self.eta_w == 'auto' else self.eta_w
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
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.ones(2 * d_z) / (2 * d_z)
                w_acc = np.ones(2 * d_z) / (2 * d_z)
                res = np.zeros(2 * d_z)
                res_pre = np.zeros(2 * d_z)
                cors = 0

            # quantities for updating theta
            if d_x * d_z < n**2:
                cors_t = xz @ (w[:d_z] - w[d_z:])
            else:
                test_fn = np.dot(Z, w[:d_z] -
                                 w[d_z:]).reshape(-1, 1)
                cors_t = np.mean(test_fn * X, axis=0)
            cors += cors_t

            # quantities for updating w
            if d_x * d_z < n**2:
                res[:d_z] = (theta[:d_x] -
                             theta[d_x:]).T @ xz - yz
            else:
                pred_fn = np.dot(X, theta[:d_x] -
                                 theta[d_x:]).reshape(-1, 1)
                res[:d_z] = np.mean(Z * pred_fn, axis=0) - yz
            res[d_z:] = - res[:d_z]

            # update theta
            theta[:d_x] = np.exp(-1 - eta_theta *
                                 (cors + cors_t + (t + 1) * lambda_theta))
            theta[d_x:] = np.exp(-1 - eta_theta *
                                 (- cors - cors_t + (t + 1) * lambda_theta))
            normalization = np.sum(theta)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w * \
                np.exp(2 * eta_w * res - eta_w * res_pre)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc[:d_z] - w_acc[d_z:]
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
        self.rho_ = theta_acc
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc[:d_z] - w_acc[d_z:]

        self._post_process(Z, X, Y)

        return self


class sparse_ridge_l1vsl1(_SparseLinearAdversarialGMM):
    r"""Sparse linear NPIV with empirical-L2 learner regularization.

    Let :math:`Q_X=\mathbb E_n[XX^\top]` and
    :math:`m(\alpha)=\mathbb E_n[Z(X^\top\alpha-Y)]`. This estimator solves

    .. math::

        \min_{\|\alpha\|_1\leq B}\max_{\|\theta\|_1\leq 1}
        \theta^\top m(\alpha)
        +\frac{\lambda}{2}\alpha^\top Q_X\alpha.

    ``duality_gap_`` uses the exact quadratic learner best response over the
    :math:`\ell_1` ball, including the active-boundary case.

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
        moment = np.mean(
            Z * (np.dot(X, self.coef_) - Y).reshape(-1, 1), axis=0)
        learner_gradient = np.mean(
            X * np.dot(Z, self.w_).reshape(-1, 1), axis=0)
        self.max_response_loss_ = np.linalg.norm(moment, ord=np.inf)\
            + .5 * self.lambda_theta * self.coef_.T @ self.xx @ self.coef_
        self.min_response_loss_ = -np.mean(Y * np.dot(Z, self.w_))\
            + quadratic_min_l1(learner_gradient, self.xx,
                               self.lambda_theta, self.B,
                               getattr(self, 'xx_pinv', None))

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
        self._validate_parameters()
        Z, X, Y = self._check_input(Z, X, Y)
        T = self.n_iter
        d_x = X.shape[1]
        d_z = Z.shape[1]
        n = X.shape[0]
        B = self.B
        eta_theta = .5 if self.eta_theta == 'auto' else self.eta_theta
        eta_w = .5 if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        yz = np.mean(Y.reshape(-1, 1) * Z, axis=0)
        xx = np.mean(cross_product(X, X),
                        axis=0).reshape(d_x, d_x).T
        self.xx = xx
        self.xx_pinv = np.linalg.pinv(xx)
        if d_x * d_z < n**2:
            xz = np.mean(cross_product(X, Z),
                         axis=0).reshape(d_z, d_x).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.ones(2 * d_z) / (2 * d_z)
                w_acc = np.ones(2 * d_z) / (2 * d_z)
                res = np.zeros(2 * d_z)
                res_pre = np.zeros(2 * d_z)
                cors = 0

            # quantities for updating theta
            xx_theta = xx @ (theta[:d_x] - theta[d_x:])
            if d_x * d_z < n**2:
                cors_t = xz @ (w[:d_z] - w[d_z:]) + lambda_theta * xx_theta
            else:
                test_fn = np.dot(Z, w[:d_z] -
                                 w[d_z:]).reshape(-1, 1)
                cors_t = np.mean(test_fn * X, axis=0) + lambda_theta * xx_theta
            cors += cors_t

            # quantities for updating w
            if d_x * d_z < n**2:
                res[:d_z] = (theta[:d_x] -
                             theta[d_x:]).T @ xz - yz
            else:
                pred_fn = np.dot(X, theta[:d_x] -
                                 theta[d_x:]).reshape(-1, 1)
                res[:d_z] = np.mean(Z * pred_fn, axis=0) - yz
            res[d_z:] = - res[:d_z]

            # update theta
            theta[:d_x] = np.exp(-1 - eta_theta * (cors + cors_t))
            theta[d_x:] = np.exp(-1 - eta_theta * (- cors - cors_t))
            normalization = np.sum(theta)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w * \
                np.exp(2 * eta_w * res - eta_w * res_pre)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc[:d_z] - w_acc[d_z:]
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
        self.rho_ = theta_acc
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc[:d_z] - w_acc[d_z:]

        self._post_process(Z, X, Y)

        return self
