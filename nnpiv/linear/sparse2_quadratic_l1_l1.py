r"""Sparse nested NPIV with quadratically penalized linear critics.

Classes
-------
sparse2_ridge_quadratic_l1vsl1
    Simultaneous sparse estimator with empirical-L2 learner penalties and
    quadratic critic penalties.
"""

# Licensed under the MIT License.

import numpy as np

from .sparse2_l1_l1 import _SparseLinear2AdversarialGMM


def _weighted_cross(left, right, weights):
    """Return a normalized weighted cross moment."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    return left.T @ (weights[:, None] * right) / np.sum(weights)


def _weighted_covariance_action(features, vector, weights):
    """Apply a normalized weighted feature covariance to a vector."""
    weights = np.asarray(weights, dtype=float).reshape(-1)
    values = features @ vector
    return features.T @ (weights * values) / np.sum(weights)


def _entropy_subsimplex(log_values, radius):
    """Map unconstrained entropy scores to a nonnegative l1 sub-simplex."""
    log_values = np.asarray(log_values, dtype=float)
    if radius == 0:
        return np.zeros_like(log_values)
    if np.any(np.isnan(log_values)):
        raise FloatingPointError("nonfinite learner entropy scores")

    positive_infinite = np.isposinf(log_values)
    if np.any(positive_infinite):
        result = np.zeros_like(log_values)
        result[positive_infinite] = radius / np.sum(positive_infinite)
        return result

    maximum = np.max(log_values)
    if np.isneginf(maximum):
        return np.zeros_like(log_values)
    shifted = np.exp(log_values - maximum)
    log_total = maximum + np.log(np.sum(shifted))
    if log_total > np.log(radius):
        return radius * shifted / np.sum(shifted)
    return np.exp(log_values)


def _normalized_log_weights(log_values):
    """Normalize log weights while retaining their log-domain state."""
    log_values = np.asarray(log_values, dtype=float)
    if np.any(np.isnan(log_values)):
        raise FloatingPointError("nonfinite critic entropy scores")

    positive_infinite = np.isposinf(log_values)
    if np.any(positive_infinite):
        normalized = np.full_like(log_values, -np.inf)
        normalized[positive_infinite] = -np.log(np.sum(positive_infinite))
        return normalized, np.exp(normalized)

    maximum = np.max(log_values)
    if np.isneginf(maximum):
        raise FloatingPointError("all critic entropy weights vanished")
    log_total = maximum + np.log(np.sum(np.exp(log_values - maximum)))
    normalized = log_values - log_total
    return normalized, np.exp(normalized)


class sparse2_ridge_quadratic_l1vsl1(_SparseLinear2AdversarialGMM):
    r"""Nested sparse estimator with quadratically penalized critics.

    Let :math:`\mathbb E_p` and :math:`\mathbb E_q` denote the empirical means
    selected for the first and second moments, and let :math:`\mathbb E_n`
    denote the full-sample mean. Define

    .. math::

        \begin{aligned}
        Q_A&=\mathbb E_n[AA^\top],& Q_B&=\mathbb E_n[BB^\top],\\
        Q_D&=\mathbb E_p[DD^\top],& Q_C&=\mathbb E_q[CC^\top],\\
        M_{AD}&=\mathbb E_p[AD^\top],&
        M_{AC}&=\mathbb E_q[(WA)C^\top],\\
        M_{BC}&=\mathbb E_q[BC^\top],& y_D&=\mathbb E_p[YD],\\
        r_1(\alpha)&=y_D-M_{AD}^\top\alpha,&
        r_2(\alpha,\beta)&=M_{AC}^\top\alpha-M_{BC}^\top\beta.
        \end{aligned}

    The estimator minimizes over :math:`\|\alpha\|_1\leq V_1` and
    :math:`\|\beta\|_1\leq V_2`, and maximizes over unit
    :math:`\ell_1` critic balls, the objective

    .. math::

        2r_1^\top\theta_1-\theta_1^\top Q_D\theta_1
        +2r_2^\top\theta_2-\theta_2^\top Q_C\theta_2
        +\mu'\alpha^\top Q_A\alpha+\mu\beta^\top Q_B\beta.

    ``W`` is the observation-level bridge multiplier in the second moment; it
    is not a sample weight. The OFTRL implementation uses the equivalent
    half-scaled objective and averages feasible iterates. With ``tol=None`` all
    iterations are used. A nonnegative tolerance checks the conservative
    ``duality_gap_upper_bound_`` every 50 iterations.

    Parameters:
        mu (float): Second learner ridge coefficient. Defaults to 0.01.
        V1 (float): :math:`\ell_1` radius for ``alpha``. Defaults to 3.
        V2 (float): :math:`\ell_1` radius for ``beta``. Defaults to 3.
        eta_alpha (str or float): First learner rate. ``"auto"`` uses the
            theoretical rate described below.
        eta_w1 (str or float): First critic rate. Defaults to ``"auto"``.
        eta_beta (str or float): Second learner rate. Defaults to ``"auto"``.
        eta_w2 (str or float): Second critic rate. Defaults to ``"auto"``.
        n_iter (int): Number of played iterates. Defaults to 3000.
        tol (float or None): Optional nonnegative stopping bound. Defaults to
            ``None``.
        sparsity (int or None): Optional coefficient post-thresholding level.
            Defaults to ``None``.
        fit_intercept (bool): Add intercept columns to all feature and
            instrument matrices. Defaults to ``True``.
        mu_prime (float or None): First learner ridge coefficient. ``None``
            uses ``mu``.

    Attributes:
        alpha_ (array-like): Averaged coefficients for the first learner.
        beta_ (array-like): Averaged coefficients for the second learner.
        w1_ (array-like): Averaged first critic coefficients.
        w2_ (array-like): Averaged second critic coefficients.
        n_iters_ (int): Number of averaged iterates.
        duality_gap_upper_bound_ (float): Conservative full-scale saddle-gap
            bound at the returned averages.
        duality_gap_upper_bounds_ (list): Bounds recorded at 50-iteration
            checkpoints when ``tol`` is enabled.
        eta_base_ (float): Common automatic-rate scale.
        eta_alpha_, eta_beta_, eta_w1_, eta_w2_ (float): Resolved learner and
            critic rates used during fitting.

    Notes:
        For automatic rates, let :math:`\|M\|_{\max}=\max_{jk}|M_{jk}|` and
        :math:`m=\max\{\|M_{AD}\|_{\max},\|M_{AC}\|_{\max},
        \|M_{BC}\|_{\max}\}`. Then :math:`\eta=(16m)^{-1}` (or :math:`1`
        when :math:`m=0`), the learner rates are :math:`2\eta/V_j`, and both
        critic rates are :math:`2\eta`. A learner with :math:`V_j=0` uses a
        zero rate.
    """

    def __init__(self, mu=0.01, V1=3, V2=3,
                 eta_alpha='auto', eta_w1='auto', eta_beta='auto',
                 eta_w2='auto', n_iter=3000, tol=None, sparsity=None,
                 fit_intercept=True, mu_prime=None):
        super().__init__(
            mu=mu, V1=V1, V2=V2, eta_alpha=eta_alpha,
            eta_w1=eta_w1, eta_beta=eta_beta, eta_w2=eta_w2,
            n_iter=n_iter, tol=tol, sparsity=sparsity,
            fit_intercept=fit_intercept
        )
        self.mu_prime = mu_prime

    def _validate_parameters(self):
        super()._validate_parameters()
        mu_prime = self.mu if self.mu_prime is None else self.mu_prime
        if not np.isfinite(self.mu) or self.mu < 0:
            raise ValueError("mu must be finite and nonnegative")
        if not np.isfinite(mu_prime) or mu_prime < 0:
            raise ValueError("mu_prime must be finite and nonnegative")
        if (not np.isfinite(self.V1) or not np.isfinite(self.V2)
                or self.V1 < 0 or self.V2 < 0):
            raise ValueError("V1 and V2 must be finite and nonnegative")
        for name in ('eta_alpha', 'eta_w1', 'eta_beta', 'eta_w2'):
            value = getattr(self, name)
            if isinstance(value, str) and value == 'auto':
                continue
            if (not np.isscalar(value) or isinstance(value, str)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be 'auto' or a finite positive number")
        if (self.tol is not None
                and (not np.isfinite(self.tol) or self.tol < 0)):
            raise ValueError("tol must be None or finite and nonnegative")
        self.mu_prime_ = float(mu_prime)

    @staticmethod
    def _resolve_rate(value, automatic):
        return float(automatic if value == 'auto' else value)

    def _half_gradients(self, alpha, beta, theta1, theta2):
        """Return gradients of the half-scaled saddle objective."""
        grad_alpha = (-self._ad @ theta1 + self._ac @ theta2
                      + self.mu_prime_ * self._aa @ alpha)
        grad_beta = -self._bc @ theta2 + self.mu * self._bb @ beta
        score1 = (self._yd - self._ad.T @ alpha
                  - _weighted_covariance_action(
                      self._D_fit, theta1, self.weights1))
        score2 = (self._ac.T @ alpha - self._bc.T @ beta
                  - _weighted_covariance_action(
                      self._C_fit, theta2, self.weights2))
        return grad_alpha, grad_beta, score1, score2

    def _duality_gap_upper_bound(self, alpha, beta, theta1, theta2):
        """Return a conservative gap bound on the full objective scale."""
        moment1 = self._yd - self._ad.T @ alpha
        moment2 = self._ac.T @ alpha - self._bc.T @ beta
        max_response = (
            2 * np.linalg.norm(moment1, ord=np.inf)
            + 2 * np.linalg.norm(moment2, ord=np.inf)
            + self.mu_prime_ * alpha @ self._aa @ alpha
            + self.mu * beta @ self._bb @ beta
        )

        alpha_gradient = -self._ad @ theta1 + self._ac @ theta2
        beta_gradient = -self._bc @ theta2
        min_response = (
            2 * self._yd @ theta1
            - theta1 @ _weighted_covariance_action(
                self._D_fit, theta1, self.weights1)
            - theta2 @ _weighted_covariance_action(
                self._C_fit, theta2, self.weights2)
            - 2 * self.V1 * np.linalg.norm(alpha_gradient, ord=np.inf)
            - 2 * self.V2 * np.linalg.norm(beta_gradient, ord=np.inf)
        )
        self.max_response_loss_upper_bound_ = max_response
        self.min_response_loss_lower_bound_ = min_response
        gap = float(max_response - min_response)
        roundoff = 1e-10 * max(
            1.0, abs(float(max_response)), abs(float(min_response)))
        if gap < -roundoff:
            raise RuntimeError("computed saddle-gap upper bound is negative")
        return max(gap, 0.0)

    def _set_fitted_averages(self, sums, count):
        self.alpha_ = sums[0] / count
        self.beta_ = sums[1] / count
        self.w1_ = sums[2] / count
        self.w2_ = sums[3] / count

    def _post_process(self):
        if self.sparsity is not None:
            n = self._A_fit.shape[0]
            alpha_threshold = 1 / (self.sparsity * n**(2 / 3))
            beta_threshold = 1 / (self.sparsity * n**(2 / 3))
            self.alpha_[np.abs(self.alpha_) < alpha_threshold] = 0
            self.beta_[np.abs(self.beta_) < beta_threshold] = 0
        self.duality_gap_upper_bound_ = self._duality_gap_upper_bound(
            self.alpha_, self.beta_, self.w1_, self.w2_)

    def fit(self, A, B, C, D, Y, W=None, subsetted=False,
            subset_ind1=None, subset_ind2=None):
        """Fit the nested quadratic-critic sparse estimator.

        Parameters:
            A (array-like): Features for the first learner ``g``.
            B (array-like): Features for the second learner ``h``.
            C (array-like): Instruments for the ``h - W*g`` moment.
            D (array-like): Instruments for the ``Y - g`` moment.
            Y (array-like): Outcome vector.
            W (array-like, optional): Observation-level multiplier on ``g`` in
                the second moment. ``None`` uses ones.
            subsetted (bool, default=False): Use stage-specific empirical
                means selected by binary masks.
            subset_ind1 (array-like, optional): Nonempty binary mask for the
                ``Y - g`` moment. Required when ``subsetted=True``.
            subset_ind2 (array-like, optional): Nonempty binary mask for the
                ``h - W*g`` moment. If omitted, uses the complement of
                ``subset_ind1``. Explicit masks need not partition the sample.

        Returns:
            sparse2_ridge_quadratic_l1vsl1: Fitted estimator.
        """
        self._validate_parameters()
        W = np.ones(Y.shape[0]) if W is None else W
        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W)
        self._set_subset_weights(
            Y.shape[0], subsetted, subset_ind1, subset_ind2)

        self._A_fit = A
        self._D_fit = D
        self._C_fit = C
        self._yd = D.T @ (self.weights1 * Y) / np.sum(self.weights1)
        self._ad = _weighted_cross(A, D, self.weights1)
        self._ac = _weighted_cross(W * A, C, self.weights2)
        self._bc = _weighted_cross(B, C, self.weights2)
        self._aa = A.T @ A / A.shape[0]
        self._bb = B.T @ B / B.shape[0]

        cross_scale = max(
            np.max(np.abs(self._ad)),
            np.max(np.abs(self._ac)),
            np.max(np.abs(self._bc))
        )
        self.eta_base_ = 1 / (16 * cross_scale) if cross_scale > 0 else 1.0
        auto_alpha = 0.0 if self.V1 == 0 else 2 * self.eta_base_ / self.V1
        auto_beta = 0.0 if self.V2 == 0 else 2 * self.eta_base_ / self.V2
        self.eta_alpha_ = self._resolve_rate(self.eta_alpha, auto_alpha)
        self.eta_beta_ = self._resolve_rate(self.eta_beta, auto_beta)
        self.eta_w1_ = self._resolve_rate(self.eta_w1, 2 * self.eta_base_)
        self.eta_w2_ = self._resolve_rate(self.eta_w2, 2 * self.eta_base_)

        d_a, d_b = A.shape[1], B.shape[1]
        d_c, d_d = C.shape[1], D.shape[1]
        rho1 = _entropy_subsimplex(np.full(2 * d_a, -1.0), self.V1)
        rho2 = _entropy_subsimplex(np.full(2 * d_b, -1.0), self.V2)
        log_omega1 = np.full(2 * d_d, -np.log(2 * d_d))
        log_omega2 = np.full(2 * d_c, -np.log(2 * d_c))
        omega1 = np.exp(log_omega1)
        omega2 = np.exp(log_omega2)

        cumulative_alpha = np.zeros(d_a)
        cumulative_beta = np.zeros(d_b)
        previous_score1 = np.zeros(d_d)
        previous_score2 = np.zeros(d_c)
        sums = [np.zeros(d_a), np.zeros(d_b),
                np.zeros(d_d), np.zeros(d_c)]
        self.duality_gap_upper_bounds_ = []

        played = 0
        for iteration in range(self.n_iter):
            alpha = rho1[:d_a] - rho1[d_a:]
            beta = rho2[:d_b] - rho2[d_b:]
            theta1 = omega1[:d_d] - omega1[d_d:]
            theta2 = omega2[:d_c] - omega2[d_c:]
            for total, action in zip(sums, (alpha, beta, theta1, theta2)):
                total += action
            played += 1

            if self.tol is not None and played % 50 == 0:
                self._set_fitted_averages(sums, played)
                bound = self._duality_gap_upper_bound(
                    self.alpha_, self.beta_, self.w1_, self.w2_)
                self.duality_gap_upper_bounds_.append(bound)
                if bound <= self.tol:
                    break
            if iteration + 1 == self.n_iter:
                break

            grad_alpha, grad_beta, score1, score2 = self._half_gradients(
                alpha, beta, theta1, theta2)
            cumulative_alpha += grad_alpha
            cumulative_beta += grad_beta
            optimistic_alpha = cumulative_alpha + grad_alpha
            optimistic_beta = cumulative_beta + grad_beta
            rho1 = _entropy_subsimplex(
                np.concatenate([
                    -1 - self.eta_alpha_ * optimistic_alpha,
                    -1 + self.eta_alpha_ * optimistic_alpha
                ]), self.V1)
            rho2 = _entropy_subsimplex(
                np.concatenate([
                    -1 - self.eta_beta_ * optimistic_beta,
                    -1 + self.eta_beta_ * optimistic_beta
                ]), self.V2)

            optimistic_score1 = 2 * score1 - previous_score1
            optimistic_score2 = 2 * score2 - previous_score2
            log_omega1 += self.eta_w1_ * np.concatenate([
                optimistic_score1, -optimistic_score1
            ])
            log_omega2 += self.eta_w2_ * np.concatenate([
                optimistic_score2, -optimistic_score2
            ])
            log_omega1, omega1 = _normalized_log_weights(log_omega1)
            log_omega2, omega2 = _normalized_log_weights(log_omega2)
            previous_score1 = score1.copy()
            previous_score2 = score2.copy()

        self.n_iters_ = played
        self._set_fitted_averages(sums, played)
        self._post_process()
        return self
