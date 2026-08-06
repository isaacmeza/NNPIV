r"""Nested linear NPIV estimators with linear :math:`\ell_2` critics.

Classes:
-------
_SparseLinear2AdversarialGMM
    Base class for sparse linear adversarial GMM for nested NPIV.
sparse2_l2vsl2
    Simultaneous estimator with coefficient-L2 penalties.
sparse2_ridge_l2vsl2
    Simultaneous estimator with empirical-L2 learner penalties.
"""


# Licensed under the MIT License.

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from nnpiv.linear.utilities import (
    cross_product, quadratic_min_l2, quadratic_min_l2_identity
)


class _SparseLinear2AdversarialGMM:
    r"""
    Base class for sparse linear adversarial GMM for nested NPIV.

    This class implements common functionality for sparse linear models using adversarial GMM in a nested NPIV setting.

    Parameters:
        mu (float): Nonnegative learner regularization coefficient.
        V1 (float): Nonnegative :math:`\ell_2` radius for ``alpha``.
        V2 (float): Nonnegative :math:`\ell_2` radius for ``beta``.
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

    def _validate_parameters(self):
        if self.n_iter < 2:
            raise ValueError("n_iter must be at least 2")
        if self.V1 < 0 or self.V2 < 0:
            raise ValueError("V1 and V2 must be nonnegative")
        if self.mu < 0:
            raise ValueError("mu must be nonnegative")

    def _set_subset_weights(self, n, subsetted, subset_ind1, subset_ind2):
        self.weights1 = np.ones(n)
        self.weights2 = np.ones(n)
        if not subsetted:
            return
        if subset_ind1 is None:
            raise ValueError("subset_ind1 must be provided when subsetted is True")

        subset_ind1 = np.asarray(subset_ind1).reshape(-1)
        if subset_ind1.size != n:
            raise ValueError("subset_ind1 must have the same length as Y")
        if not np.all(np.isin(subset_ind1, [0, 1])):
            raise ValueError("subset_ind1 must contain only 0 and 1")

        if subset_ind2 is None:
            subset_ind2 = 1 - subset_ind1
        else:
            subset_ind2 = np.asarray(subset_ind2).reshape(-1)
            if subset_ind2.size != n:
                raise ValueError("subset_ind2 must have the same length as Y")
            if not np.all(np.isin(subset_ind2, [0, 1])):
                raise ValueError("subset_ind2 must contain only 0 and 1")

        if np.sum(subset_ind1) == 0 or np.sum(subset_ind2) == 0:
            raise ValueError("each selected subset must contain an observation")
        self.weights1 = subset_ind1.astype(float)
        self.weights2 = subset_ind2.astype(float)

    def weighted_mean(self, arr, weights, axis=0):
        """
        Compute the weighted mean of an array along the specified axis.

        Args:
            arr (array-like): Input array.
            weights (array-like): Weights for the mean computation.
            axis (int, optional): Axis along which to compute the mean. Defaults to 0.

        Returns:
            array: Weighted mean.
        """
        weights = np.array(weights)
        if arr.ndim == 1 or axis is None:
            return np.sum(arr * weights) / np.sum(weights)
        else:
            return np.sum(arr * weights[:, np.newaxis], axis=axis) / np.sum(weights)

    def _check_input(self, A, B, C, D, Y, W):
        if self.fit_intercept:
            A = np.hstack([np.ones((A.shape[0], 1)), A])
            B = np.hstack([np.ones((B.shape[0], 1)), B])
            C = np.hstack([np.ones((C.shape[0], 1)), C])
            D = np.hstack([np.ones((D.shape[0], 1)), D])
        return A, B, C, D, Y.flatten(), W.reshape(-1, 1)

    def predict(self, B, *args):
        """
        Predict using the fitted model.

        Args:
            B (array-like): Features for the second learner ``h``.
            *args: Optional features for the first learner ``g``.

        Returns:
            array or tuple: Predictions for ``h``. If ``A`` is also supplied,
            returns ``(h(B), g(A))``.
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
            return (np.dot(B, self.beta_), np.dot(A, self.alpha_))
        else:
            raise ValueError("predict expects at most two arguments, B_test and optionally A_test")

    @property
    def coef(self):
        return self.beta_[1:] if self.fit_intercept else self.beta_

    @property
    def intercept(self):
        return self.beta_[0] if self.fit_intercept else 0


class sparse2_l2vsl2(_SparseLinear2AdversarialGMM):
    r"""Nested linear NPIV with coefficient-L2 penalties.

    Let :math:`r_1(\alpha)=\mathbb E_p[D(Y-A^\top\alpha)]` and
    :math:`r_2(\alpha,\beta)=
    \mathbb E_q[C((WA)^\top\alpha-B^\top\beta)]`. The estimator solves

    .. math::

        \min_{\substack{\|\alpha\|_2\leq V_1\\
                        \|\beta\|_2\leq V_2}}
        \max_{\substack{\|\theta_1\|_2\leq1\\
                        \|\theta_2\|_2\leq1}}
        \theta_1^\top r_1+\theta_2^\top r_2
        +\frac{\mu}{2}(\|\alpha\|_2^2+\|\beta\|_2^2).

    ``duality_gap_`` and the moment violations use the matching Euclidean
    geometry and exact constrained best responses.

    Parameters:
        Same as `_SparseLinear2AdversarialGMM`.
    """

    def _check_duality_gap(self, A, B, C, D, Y, W):
        """
        Calculate the duality gap to certify convergence of the algorithm.

        The ensembles can be thought of as primal and dual solutions, and the duality gap can be used as a certificate for convergence of the algorithm.

        Parameters:
            A (array-like): Features for the first learner ``g``.
            B (array-like): Features for the second learner ``h``.
            C (array-like): Instruments for the ``h - W*g`` moment.
            D (array-like): Instruments for the ``Y - g`` moment.
            Y (array-like): Outcomes.
            W (array-like): Observation-level bridge multiplier.

        Returns:
            bool: True if the duality gap is below the tolerance level, indicating convergence.
        """
        first_moment = self.weighted_mean(
            D * (Y - np.dot(A, self.alpha_)).reshape(-1, 1),
            self.weights1, axis=0)
        second_moment = self.weighted_mean(
            C * (np.dot(A * W, self.alpha_) - np.dot(B, self.beta_)).reshape(-1, 1),
            self.weights2, axis=0)
        alpha_gradient = -self.weighted_mean(
            A * np.dot(D, self.w1_).reshape(-1, 1), self.weights1, axis=0)\
            + self.weighted_mean(
                W * A * np.dot(C, self.w2_).reshape(-1, 1),
                self.weights2, axis=0)
        beta_gradient = -self.weighted_mean(
            B * np.dot(C, self.w2_).reshape(-1, 1), self.weights2, axis=0)

        self.max_response_loss_ = np.linalg.norm(first_moment, ord=2)\
            + np.linalg.norm(second_moment, ord=2)\
            + .5 * self.mu * np.linalg.norm(self.alpha_, ord=2)**2\
            + .5 * self.mu * np.linalg.norm(self.beta_, ord=2)**2
        self.min_response_loss_ = self.weighted_mean(
            Y * np.dot(D, self.w1_), self.weights1)\
            + quadratic_min_l2_identity(alpha_gradient, self.mu, self.V1)\
            + quadratic_min_l2_identity(beta_gradient, self.mu, self.V2)

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
            A (array-like): Features for the first learner ``g``.
            B (array-like): Features for the second learner ``h``.
            C (array-like): Instruments for the ``h - W*g`` moment.
            D (array-like): Instruments for the ``Y - g`` moment.
            Y (array-like): Outcomes.
            W (array-like, optional): Observation-level multiplier on ``g`` in
                the second bridge moment. ``None`` uses ones.
            subsetted (bool, optional): Use stage-specific empirical means.
            subset_ind1 (array-like, optional): Nonempty binary mask for the
                ``Y - g`` moment; required when ``subsetted=True``.
            subset_ind2 (array-like, optional): Nonempty binary mask for the
                ``h - W*g`` moment. If omitted, uses the complement of
                ``subset_ind1``. Explicit masks need not partition the sample.

        Returns:
            self: Fitted estimator.
        """
        self._validate_parameters()
        W = np.ones(Y.shape[0]) if W is None else W
        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W)
        self._set_subset_weights(
            Y.shape[0], subsetted, subset_ind1, subset_ind2)

        T = self.n_iter
        d_a = A.shape[1]
        d_b = B.shape[1]
        d_c = C.shape[1]
        d_d = D.shape[1]
        n = A.shape[0]
        V1 = self.V1
        V2 = self.V2
        eta_alpha = np.sqrt(np.log(d_a + 1) / T) if self.eta_alpha == 'auto' else self.eta_alpha
        eta_beta = np.sqrt(np.log(d_b + 1) / T) if self.eta_beta == 'auto' else self.eta_beta
        eta_w1 = np.sqrt(np.log(d_d + 1) / T) if self.eta_w1 == 'auto' else self.eta_w1
        eta_w2 = np.sqrt(np.log(d_c + 1) / T) if self.eta_w2 == 'auto' else self.eta_w2
        mu = self.mu

        yd = self.weighted_mean(Y.reshape(-1, 1) * D, self.weights1, axis=0)
        if d_a * d_d < n**2:
            ad = self.weighted_mean(cross_product(A, D), self.weights1, axis=0).reshape(d_d, d_a).T
        if d_a * d_c < n**2:
            ac = self.weighted_mean(cross_product(W * A, C), self.weights2, axis=0).reshape(d_c, d_a).T
        if d_b * d_c < n**2:
            bc = self.weighted_mean(cross_product(B, C), self.weights2, axis=0).reshape(d_c, d_b).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                alpha = np.zeros(d_a)
                beta = np.zeros(d_b)
                w1 = np.zeros(d_d)
                w2 = np.zeros(d_c)
                alpha_acc = np.zeros(d_a)
                beta_acc = np.zeros(d_b)
                w1_acc = np.zeros(d_d)
                w2_acc = np.zeros(d_c)
                res1 = np.zeros(d_d)
                res2 = np.zeros(d_c)
                res1_pre = np.zeros(d_d)
                res2_pre = np.zeros(d_c)
                cors1 = np.zeros(d_a)
                cors2 = np.zeros(d_b)
                cors1_pre = np.zeros(d_a)
                cors2_pre = np.zeros(d_b)

            # quantities for updating alpha
            if d_a * d_d < n**2:
                cors1[:] = - ad @ w1 + mu * alpha
            else:
                test_fn = np.dot(D, w1).reshape(-1, 1)
                cors1[:] = - self.weighted_mean(test_fn * A, self.weights1, axis=0) + mu * alpha
            if d_a * d_c < n**2:
                cors1[:] += ac @ w2
            else:
                test_fn = np.dot(C, w2).reshape(-1, 1)
                cors1[:] += self.weighted_mean(test_fn * A * W, self.weights2, axis=0)

            # quantities for updating beta
            if d_b * d_c < n**2:
                cors2[:] = - bc @ w2 + mu * beta
            else:
                test_fn = np.dot(C, w2).reshape(-1, 1)
                cors2[:] = - self.weighted_mean(test_fn * B, self.weights2, axis=0) + mu * beta

            # quantities for updating w1
            if d_a * d_d < n**2:
                res1[:] = yd - alpha.T @ ad
            else:
                pred_fn = np.dot(A, alpha).reshape(-1, 1)
                res1[:] = yd - self.weighted_mean(D * pred_fn, self.weights1, axis=0)

            # quantities for updating w2
            if d_c * d_a < n**2:
                res2[:] = alpha.T @ ac
            else:
                pred_fn = np.dot(A * W, alpha).reshape(-1, 1)
                res2[:] = self.weighted_mean(C * pred_fn, self.weights2, axis=0)
            if d_c * d_b < n**2:
                res2[:] -= beta.T @ bc
            else:
                pred_fn = np.dot(B, beta).reshape(-1, 1)
                res2[:] -= self.weighted_mean(C * pred_fn, self.weights2, axis=0)

            # update alpha
            alpha[:] = alpha - 2 * eta_alpha * cors1 + eta_alpha * cors1_pre
            normalization = np.linalg.norm(alpha, ord=2)
            if normalization > V1:
                alpha[:] = alpha * V1 / normalization

            # update beta
            beta[:] = beta - 2 * eta_beta * cors2 + eta_beta * cors2_pre
            normalization = np.linalg.norm(beta, ord=2)
            if normalization > V2:
                beta[:] = beta * V2 / normalization

            # update w1
            w1[:] = w1 + 2 * eta_w1 * res1 - eta_w1 * res1_pre
            norm_w1 = np.linalg.norm(w1, ord=2)
            w1[:] = w1 / norm_w1 if norm_w1 > 1 else w1

            # update w2
            w2[:] = w2 + 2 * eta_w2 * res2 - eta_w2 * res2_pre
            norm_w2 = np.linalg.norm(w2, ord=2)
            w2[:] = w2 / norm_w2 if norm_w2 > 1 else w2

            alpha_acc = alpha_acc * (t - 1) / t + alpha / t
            beta_acc = beta_acc * (t - 1) / t + beta / t
            w1_acc = w1_acc * (t - 1) / t + w1 / t
            w2_acc = w2_acc * (t - 1) / t + w2 / t

            res1_pre[:] = res1
            res2_pre[:] = res2
            cors1_pre[:] = cors1
            cors2_pre[:] = cors2

            if t % 50 == 0:
                self.alpha_ = alpha_acc
                self.beta_ = beta_acc
                self.w1_ = w1_acc
                self.w2_ = w2_acc
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
        self.alpha_ = alpha_acc
        self.beta_ = beta_acc
        self.w1_ = w1_acc
        self.w2_ = w2_acc

        self._post_process(A, B, C, D, Y, W)
        return self


class sparse2_ridge_l2vsl2(_SparseLinear2AdversarialGMM):
    r"""Nested linear NPIV with empirical-L2 learner penalties.

    This class uses the same two linear moment games as
    :class:`sparse2_l2vsl2`, with learner penalty

    .. math::

        \frac{\mu}{2}\left(
        \alpha^\top\mathbb E_n[AA^\top]\alpha+
        \beta^\top\mathbb E_n[BB^\top]\beta\right).

    Stage moments use their normalized selected samples, while the two ridge
    matrices use the full sample. The duality gap evaluates the exact
    :math:`\ell_2` trust-region best responses, including singular covariance
    matrices.

    Parameters:
        Same as `_SparseLinear2AdversarialGMM`.
    """

    def _check_duality_gap(self, A, B, C, D, Y, W):
        """
        Calculate the duality gap to certify convergence of the algorithm.

        The ensembles can be thought of as primal and dual solutions, and the duality gap can be used as a certificate for convergence of the algorithm.

        Parameters:
            A (array-like): Features for the first learner ``g``.
            B (array-like): Features for the second learner ``h``.
            C (array-like): Instruments for the ``h - W*g`` moment.
            D (array-like): Instruments for the ``Y - g`` moment.
            Y (array-like): Outcomes.
            W (array-like): Observation-level bridge multiplier.

        Returns:
            bool: True if the duality gap is below the tolerance level, indicating convergence.
        """
        first_moment = self.weighted_mean(
            D * (Y - np.dot(A, self.alpha_)).reshape(-1, 1),
            self.weights1, axis=0)
        second_moment = self.weighted_mean(
            C * (np.dot(A * W, self.alpha_) - np.dot(B, self.beta_)).reshape(-1, 1),
            self.weights2, axis=0)
        alpha_gradient = -self.weighted_mean(
            A * np.dot(D, self.w1_).reshape(-1, 1), self.weights1, axis=0)\
            + self.weighted_mean(
                W * A * np.dot(C, self.w2_).reshape(-1, 1),
                self.weights2, axis=0)
        beta_gradient = -self.weighted_mean(
            B * np.dot(C, self.w2_).reshape(-1, 1), self.weights2, axis=0)

        self.max_response_loss_ = np.linalg.norm(first_moment, ord=2)\
            + np.linalg.norm(second_moment, ord=2)\
            + .5 * self.mu * self.alpha_.T @ self.aa @ self.alpha_\
            + .5 * self.mu * self.beta_.T @ self.bb @ self.beta_
        self.min_response_loss_ = self.weighted_mean(
            Y * np.dot(D, self.w1_), self.weights1)\
            + quadratic_min_l2(
                alpha_gradient, self.aa, self.mu, self.V1, self.aa_eigh)\
            + quadratic_min_l2(
                beta_gradient, self.bb, self.mu, self.V2, self.bb_eigh)

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
            A (array-like): Features for the first learner ``g``.
            B (array-like): Features for the second learner ``h``.
            C (array-like): Instruments for the ``h - W*g`` moment.
            D (array-like): Instruments for the ``Y - g`` moment.
            Y (array-like): Outcomes.
            W (array-like, optional): Observation-level multiplier on ``g`` in
                the second bridge moment. ``None`` uses ones.
            subsetted (bool, optional): Use stage-specific empirical means.
            subset_ind1 (array-like, optional): Nonempty binary mask for the
                ``Y - g`` moment; required when ``subsetted=True``.
            subset_ind2 (array-like, optional): Nonempty binary mask for the
                ``h - W*g`` moment. If omitted, uses the complement of
                ``subset_ind1``. Explicit masks need not partition the sample.

        Returns:
            self: Fitted estimator.
        """
        self._validate_parameters()
        W = np.ones(Y.shape[0]) if W is None else W
        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W)
        self._set_subset_weights(
            Y.shape[0], subsetted, subset_ind1, subset_ind2)

        T = self.n_iter
        d_a = A.shape[1]
        d_b = B.shape[1]
        d_c = C.shape[1]
        d_d = D.shape[1]
        n = A.shape[0]
        V1 = self.V1
        V2 = self.V2
        eta_alpha = np.sqrt(np.log(d_a + 1) / T) if self.eta_alpha == 'auto' else self.eta_alpha
        eta_beta = np.sqrt(np.log(d_b + 1) / T) if self.eta_beta == 'auto' else self.eta_beta
        eta_w1 = np.sqrt(np.log(d_d + 1) / T) if self.eta_w1 == 'auto' else self.eta_w1
        eta_w2 = np.sqrt(np.log(d_c + 1) / T) if self.eta_w2 == 'auto' else self.eta_w2
        mu = self.mu

        yd = self.weighted_mean(Y.reshape(-1, 1) * D, self.weights1, axis=0)
        aa = np.mean(cross_product(A, A), axis=0).reshape(d_a, d_a).T
        self.aa = aa
        self.aa_eigh = np.linalg.eigh(.5 * (aa + aa.T))
        bb = np.mean(cross_product(B, B), axis=0).reshape(d_b, d_b).T
        self.bb = bb
        self.bb_eigh = np.linalg.eigh(.5 * (bb + bb.T))

        if d_a * d_d < n**2:
            ad = self.weighted_mean(cross_product(A, D), self.weights1, axis=0).reshape(d_d, d_a).T
        if d_a * d_c < n**2:
            ac = self.weighted_mean(cross_product(W * A, C), self.weights2, axis=0).reshape(d_c, d_a).T
        if d_b * d_c < n**2:
            bc = self.weighted_mean(cross_product(B, C), self.weights2, axis=0).reshape(d_c, d_b).T

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                alpha = np.zeros(d_a)
                beta = np.zeros(d_b)
                w1 = np.zeros(d_d)
                w2 = np.zeros(d_c)
                alpha_acc = np.zeros(d_a)
                beta_acc = np.zeros(d_b)
                w1_acc = np.zeros(d_d)
                w2_acc = np.zeros(d_c)
                res1 = np.zeros(d_d)
                res2 = np.zeros(d_c)
                res1_pre = np.zeros(d_d)
                res2_pre = np.zeros(d_c)
                cors1 = np.zeros(d_a)
                cors2 = np.zeros(d_b)
                cors1_pre = np.zeros(d_a)
                cors2_pre = np.zeros(d_b)

            # quantities for updating alpha
            if d_a * d_d < n**2:
                cors1[:] = - ad @ w1 + mu * self.aa @ alpha
            else:
                test_fn = np.dot(D, w1).reshape(-1, 1)
                cors1[:] = - self.weighted_mean(test_fn * A, self.weights1, axis=0) + mu * self.aa @ alpha
            if d_a * d_c < n**2:
                cors1[:] += ac @ w2
            else:
                test_fn = np.dot(C, w2).reshape(-1, 1)
                cors1[:] += self.weighted_mean(test_fn * A * W, self.weights2, axis=0)

            # quantities for updating beta
            if d_b * d_c < n**2:
                cors2[:] = - bc @ w2 + mu * self.bb @ beta
            else:
                test_fn = np.dot(C, w2).reshape(-1, 1)
                cors2[:] = - self.weighted_mean(test_fn * B, self.weights2, axis=0) + mu * self.bb @ beta

            # quantities for updating w1
            if d_a * d_d < n**2:
                res1[:] = yd - alpha.T @ ad
            else:
                pred_fn = np.dot(A, alpha).reshape(-1, 1)
                res1[:] = yd - self.weighted_mean(D * pred_fn, self.weights1, axis=0)

            # quantities for updating w2
            if d_c * d_a < n**2:
                res2[:] = alpha.T @ ac
            else:
                pred_fn = np.dot(A * W, alpha).reshape(-1, 1)
                res2[:] = self.weighted_mean(C * pred_fn, self.weights2, axis=0)
            if d_c * d_b < n**2:
                res2[:] -= beta.T @ bc
            else:
                pred_fn = np.dot(B, beta).reshape(-1, 1)
                res2[:] -= self.weighted_mean(C * pred_fn, self.weights2, axis=0)

            # update alpha
            alpha[:] = alpha - 2 * eta_alpha * cors1 + eta_alpha * cors1_pre
            normalization = np.linalg.norm(alpha, ord=2)
            if normalization > V1:
                alpha[:] = alpha * V1 / normalization

            # update beta
            beta[:] = beta - 2 * eta_beta * cors2 + eta_beta * cors2_pre
            normalization = np.linalg.norm(beta, ord=2)
            if normalization > V2:
                beta[:] = beta * V2 / normalization

            # update w1
            w1[:] = w1 + 2 * eta_w1 * res1 - eta_w1 * res1_pre
            norm_w1 = np.linalg.norm(w1, ord=2)
            w1[:] = w1 / norm_w1 if norm_w1 > 1 else w1

            # update w2
            w2[:] = w2 + 2 * eta_w2 * res2 - eta_w2 * res2_pre
            norm_w2 = np.linalg.norm(w2, ord=2)
            w2[:] = w2 / norm_w2 if norm_w2 > 1 else w2

            alpha_acc = alpha_acc * (t - 1) / t + alpha / t
            beta_acc = beta_acc * (t - 1) / t + beta / t
            w1_acc = w1_acc * (t - 1) / t + w1 / t
            w2_acc = w2_acc * (t - 1) / t + w2 / t

            res1_pre[:] = res1
            res2_pre[:] = res2
            cors1_pre[:] = cors1
            cors2_pre[:] = cors2

            if t % 50 == 0:
                self.alpha_ = alpha_acc
                self.beta_ = beta_acc
                self.w1_ = w1_acc
                self.w2_ = w2_acc
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
        self.alpha_ = alpha_acc
        self.beta_ = beta_acc
        self.w1_ = w1_acc
        self.w2_ = w2_acc

        self._post_process(A, B, C, D, Y, W)
        return self
