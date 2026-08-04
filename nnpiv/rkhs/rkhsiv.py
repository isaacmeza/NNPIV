"""
This module provides implementations of RKHS Instrumental Variable (IV) estimators.

Classes:
    _BaseRKHSIV: Base class for RKHS IV methods.
    RKHSIV: RKHS IV estimator.
    RKHSIVCV: RKHS IV estimator with cross-validation.
    RKHSIVL2: RKHS IV estimator with L2 regularization.
    RKHSIVL2CV: RKHS IV estimator with L2 regularization and cross-validation.
    ApproxRKHSIV: Approximate RKHS IV estimator using kernel approximations.
    ApproxRKHSIVCV: Approximate RKHS IV estimator with cross-validation using kernel approximations.
    ApproxRKHSIVL2: Approximate RKHS IV estimator with L2 regularization.
    ApproxRKHSIVL2CV: Approximate RKHS IV estimator with L2 regularization and cross-validation.
"""

# Licensed under the MIT License.

from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
from sklearn.model_selection import KFold
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.utils import _safe_indexing
import numpy as np


_DEFAULT_PINV_RCOND = 1e-15


def _check_auto(param):
    return (isinstance(param, str) and (param == 'auto'))


def _to_column_vector(y):
    arr = np.asarray(y)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    raise ValueError(
        "`Y` must be a 1D array or a 2D column vector with shape (n, 1). "
        f"Got shape={arr.shape!r}."
    )


def _to_scalar(x):
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(
            "Expected scalar quadratic form, got array with "
            f"shape={arr.shape!r} and size={arr.size}."
        )
    return float(arr.reshape(-1)[0])


def _sqrt_psd_matrix(K):
    """
    Numerically stable real square-root for symmetric PSD matrices.

    Kernel matrices can be theoretically PSD but have tiny negative/complex
    artifacts in floating point arithmetic. This routine symmetrizes ``K``,
    clips negative eigenvalues to zero, and returns a real square-root.
    """
    K_sym = 0.5 * (K + K.T)
    evals, evecs = np.linalg.eigh(K_sym)
    evals = np.clip(evals, 0.0, None)
    return (evecs * np.sqrt(evals)) @ evecs.T


def _pinv_symmetric(matrix):
    """Return the pseudoinverse of a numerically symmetrized matrix."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    return np.linalg.pinv(matrix, hermitian=True)


def _solve_symmetric(matrix, rhs):
    """Solve a nonsingular numerically symmetrized linear system."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    return np.linalg.solve(matrix, rhs)


class _BaseRKHSIV:
    """
    Base class for RKHS IV methods.

    This class provides common functionality for RKHS IV estimators.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Length scale for the kernel.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scale (str or float): Scale of the regularization parameter.
        kernel_params (dict): Additional parameters for the kernel.
    """

    def __init__(self, *args, **kwargs):
        return

    def _get_delta(self, n):
        """
        Compute the critical radius.

        Parameters:
            n (int): Number of samples.

        Returns:
            float: Critical radius.
        """
        delta_scale = 5 if _check_auto(self.delta_scale) else self.delta_scale
        delta_exp = .4 if _check_auto(self.delta_exp) else self.delta_exp
        return delta_scale / (n**(delta_exp))

    def _get_alpha_scale(self):
        return 60 if _check_auto(self.alpha_scale) else self.alpha_scale

    def _get_alpha_scales(self):
        return ([c for c in np.geomspace(0.1, 1e4, self.n_alphas)]
                if _check_auto(self.alpha_scales) else self.alpha_scales)

    def _get_alpha(self, delta, alpha_scale):
        return alpha_scale * (delta**4)

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            if _check_auto(self.gamma):
                pairwise_dists = euclidean_distances(X, X)
                median_dist = np.median(pairwise_dists)
                gamma = 1.0 / (2 * median_dist)
            else:
                gamma = self.gamma
            params = {"gamma": gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def _resolve_fitted_gamma(self, X, kernel_role):
        """Resolve one kernel bandwidth from fitting data."""
        if callable(self.kernel):
            return None
        if not _check_auto(self.gamma):
            return self.gamma

        pairwise_dists = euclidean_distances(X, X)
        median_dist = float(np.median(pairwise_dists))
        if not np.isfinite(median_dist) or median_dist <= 0:
            raise ValueError(
                f"Cannot resolve `gamma='auto'` for the {kernel_role} kernel: "
                "the median pairwise distance in the fitting data must be "
                "finite and strictly positive."
            )
        # Use the same median-distance convention as the base kernel helper.
        return 1.0 / (2 * median_dist)

    def _get_kernel_at_fitted_gamma(self, X, Y=None, fitted_gamma=None):
        """Evaluate a kernel using the bandwidth fixed during ``fit``."""
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {
                "gamma": fitted_gamma,
                "degree": self.degree,
                "coef0": self.coef0,
            }
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, **params
        )

    @staticmethod
    def _symmetric_range_basis(K):
        """Return the positive eigenspace retained at the default pinv cutoff."""
        K = np.asarray(K, dtype=float)
        K = 0.5 * (K + K.T)
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        scale = (
            float(np.max(np.abs(eigenvalues)))
            if eigenvalues.size
            else 0.0
        )
        cutoff = _DEFAULT_PINV_RCOND * scale
        retained = eigenvalues > cutoff
        return eigenvectors[:, retained], eigenvalues[retained]


class RKHSIV(_BaseRKHSIV):
    """
    RKHS IV estimator.

    This class implements an RKHS IV estimator.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Length scale for the kernel.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scale (str or float): Scale of the regularization parameter.
        kernel_params (dict): Additional parameters for the kernel.
    """

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1,
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto',
                 kernel_params=None):
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale

    @staticmethod
    def _compute_instrument_operator(Kf, n, delta):
        """Compute ``M_delta`` without explicitly forming a matrix inverse."""
        RootKf = _sqrt_psd_matrix(Kf)
        regularized_Kf = (
            Kf / (2 * n * delta**2) + np.eye(n) / 2
        )
        return RootKf @ np.linalg.solve(regularized_Kf, RootKf)

    def fit(self, Z, T, Y):
        """
        Fit the RKHS IV estimator.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        self.gamma_t_ = self._resolve_fitted_gamma(T, "treatment")
        self.gamma_z_ = self._resolve_fitted_gamma(Z, "instrument")
        Kh = self._get_kernel_at_fitted_gamma(
            T, fitted_gamma=self.gamma_t_
        )
        Kf = self._get_kernel_at_fitted_gamma(
            Z, fitted_gamma=self.gamma_z_
        )

        M = self._compute_instrument_operator(Kf, n, delta)
        self.T = T.copy()
        # NumPy's default pseudoinverse tolerance defines the numerical rank.
        self.a = np.linalg.pinv(Kh @ M @ Kh + alpha * Kh) @ Kh @ M @ Y
        return self

    def predict(self, T_test):
        """
        Predict outcomes for new treatments.

        Parameters:
            T_test (array-like): New treatments.

        Returns:
            array-like: Predicted outcomes.
        """
        if hasattr(self, "gamma_t_"):
            kernel = self._get_kernel_at_fitted_gamma(
                T_test, Y=self.T, fitted_gamma=self.gamma_t_
            )
        else:
            # Some subclasses construct kernels without storing a bandwidth.
            kernel = self._get_kernel(T_test, Y=self.T)
        return kernel @ self.a

    def score(self, Z, T, Y, delta='auto'):
        """
        Compute the score of the fitted estimator.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.
            delta (str or float): Critical radius.

        Returns:
            float: Score.
        """
        Y = _to_column_vector(Y)
        n = Y.shape[0]

        if not hasattr(self, "gamma_z_"):
            # Subclasses without a stored bandwidth use their configured
            # kernel construction and sample-dependent critical radius.
            score_delta = self._get_delta(n)
            Kf = self._get_kernel(Z)
            RootKf = _sqrt_psd_matrix(Kf)
            M = RootKf @ np.linalg.inv(
                Kf / (2 * n * score_delta**2) + np.eye(n) / 2
            ) @ RootKf
            Y_pred = self.predict(T)
            return (
                _to_scalar((Y - Y_pred).T @ M @ (Y - Y_pred)) / n**2
            )

        if _check_auto(delta):
            delta = self._get_delta(n)
        else:
            delta = float(delta)
            if not np.isfinite(delta) or delta <= 0:
                raise ValueError("`delta` must be finite and strictly positive.")

        Kf = self._get_kernel_at_fitted_gamma(
            Z, fitted_gamma=self.gamma_z_
        )
        M = self._compute_instrument_operator(Kf, n, delta)
        Y_pred = self.predict(T)
        return _to_scalar((Y - Y_pred).T @ M @ (Y - Y_pred)) / n**2


class RKHSIVCV(RKHSIV):
    """
    RKHS IV estimator with cross-validation.

    This class implements an RKHS IV estimator with cross-validation.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Length scale for the kernel.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        n_alphas (int): Number of alpha scales to try.
        cv (int): Number of folds for cross-validation.
        kernel_params (dict): Additional parameters for the kernel.
    """

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = n_alphas
        self.cv = cv

    def fit(self, Z, T, Y):
        """
        Fit the RKHS IV estimator with cross-validation.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Y = _to_column_vector(Y)
        Z = np.asarray(Z)
        T = np.asarray(T)
        n = Y.shape[0]

        alpha_scales = self._get_alpha_scales()
        fold_specific_gamma = (
            not callable(self.kernel) and _check_auto(self.gamma)
        )

        if not fold_specific_gamma:
            gamma_t = self._resolve_fitted_gamma(T, "treatment")
            gamma_z = self._resolve_fitted_gamma(Z, "instrument")
            Kh = self._get_kernel_at_fitted_gamma(
                T, fitted_gamma=gamma_t
            )
            Kf = self._get_kernel_at_fitted_gamma(
                Z, fitted_gamma=gamma_z
            )

        scores = []
        for train, test in KFold(n_splits=self.cv).split(Z):
            n_train = len(train)
            n_test = len(test)
            delta_train = self._get_delta(n_train)
            delta_test = self._get_delta(n_test)

            if fold_specific_gamma:
                gamma_t_train = self._resolve_fitted_gamma(
                    T[train], "treatment"
                )
                gamma_z_train = self._resolve_fitted_gamma(
                    Z[train], "instrument"
                )
                Kh_train = self._get_kernel_at_fitted_gamma(
                    T[train], fitted_gamma=gamma_t_train
                )
                Kh_test_train = self._get_kernel_at_fitted_gamma(
                    T[test], Y=T[train], fitted_gamma=gamma_t_train
                )
                Kf_train = self._get_kernel_at_fitted_gamma(
                    Z[train], fitted_gamma=gamma_z_train
                )
                Kf_test = self._get_kernel_at_fitted_gamma(
                    Z[test], fitted_gamma=gamma_z_train
                )
            else:
                Kh_train = Kh[np.ix_(train, train)]
                Kh_test_train = Kh[np.ix_(test, train)]
                Kf_train = Kf[np.ix_(train, train)]
                Kf_test = Kf[np.ix_(test, test)]

            M_train = self._compute_instrument_operator(
                Kf_train, n_train, delta_train
            )
            M_test = self._compute_instrument_operator(
                Kf_test, n_test, delta_test
            )
            KMK_train = Kh_train @ M_train @ Kh_train
            B_train = Kh_train @ M_train @ Y[train]
            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = self._get_alpha(delta_train, alpha_scale)
                a = np.linalg.pinv(KMK_train + alpha * Kh_train) @ B_train
                res = Y[test] - Kh_test_train @ a
                fold_scores.append(
                    _to_scalar(res.T @ M_test @ res) / (n_test**2)
                )
            scores.append(fold_scores)

        self.alpha_scales_ = np.asarray(alpha_scales, dtype=float).copy()
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = self.alpha_scales_[
            np.argmin(self.avg_scores)
        ]

        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        if fold_specific_gamma:
            self.gamma_t_ = self._resolve_fitted_gamma(T, "treatment")
            self.gamma_z_ = self._resolve_fitted_gamma(Z, "instrument")
            Kh = self._get_kernel_at_fitted_gamma(
                T, fitted_gamma=self.gamma_t_
            )
            Kf = self._get_kernel_at_fitted_gamma(
                Z, fitted_gamma=self.gamma_z_
            )
        else:
            self.gamma_t_ = gamma_t
            self.gamma_z_ = gamma_z
        M = self._compute_instrument_operator(Kf, n, delta)

        self.T = T.copy()
        # NumPy's default pseudoinverse tolerance defines the numerical rank.
        self.a = np.linalg.pinv(
            Kh @ M @ Kh + self.best_alpha * Kh) @ Kh @ M @ Y
        return self


class RKHSIVL2(_BaseRKHSIV):
    """
    RKHS IV estimator with L2 regularization.

    The instrument projection is constructed from the positive eigenspace of
    its Gram matrix. The kernel coefficients are obtained from the
    L2-regularized normal equation.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Length scale for the kernel.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        kernel_params (dict): Additional parameters for the kernel.
    """

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1,
                 delta_scale='auto', delta_exp='auto', kernel_params=None):
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp

    def fit(self, Z, T, Y):
        """
        Fit the RKHS IV estimator with L2 regularization.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Y = _to_column_vector(Y)
        Z = np.asarray(Z)
        T = np.asarray(T)
        n = Y.shape[0]
        delta = self._get_delta(n)
        alpha = delta**4

        self.gamma_t_ = self._resolve_fitted_gamma(T, "treatment")
        self.gamma_z_ = self._resolve_fitted_gamma(Z, "instrument")
        Kh = self._get_kernel_at_fitted_gamma(
            T, fitted_gamma=self.gamma_t_
        )
        Kf = self._get_kernel_at_fitted_gamma(
            Z, fitted_gamma=self.gamma_z_
        )

        instrument_basis, _ = self._symmetric_range_basis(Kf)
        Pz = instrument_basis @ instrument_basis.T

        self.T = T.copy()
        self.a = np.linalg.pinv(
            Kh @ Pz @ Kh + alpha * Kh @ Kh
        ) @ Kh @ Pz @ Y
        return self

    def predict(self, T_test):
        """
        Predict outcomes for new treatments.

        Parameters:
            T_test (array-like): New treatments.

        Returns:
            array-like: Predicted outcomes.
        """
        if hasattr(self, "gamma_t_"):
            kernel = self._get_kernel_at_fitted_gamma(
                T_test, Y=self.T, fitted_gamma=self.gamma_t_
            )
        else:
            kernel = self._get_kernel(T_test, Y=self.T)
        return kernel @ self.a


class RKHSIVL2CV(RKHSIVL2):
    """
    RKHS IV estimator with L2 regularization and cross-validation.

    Instrument projections are constructed from the positive eigenspaces of
    their Gram matrices. Candidate models and the final model retain the
    coefficient-space L2 normal equations.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Length scale for the kernel.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        n_alphas (int): Number of alpha scales to try.
        cv (int): Number of folds for cross-validation.
        kernel_params (dict): Additional parameters for the kernel.
    """

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = n_alphas
        self.cv = cv

    def fit(self, Z, T, Y):
        """
        Fit the RKHS IV estimator with L2 regularization and cross-validation.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Y = _to_column_vector(Y)
        Z = np.asarray(Z)
        T = np.asarray(T)
        n = Y.shape[0]

        alpha_scales = self._get_alpha_scales()
        fold_specific_gamma = (
            not callable(self.kernel) and _check_auto(self.gamma)
        )

        if not fold_specific_gamma:
            gamma_t = self._resolve_fitted_gamma(T, "treatment")
            gamma_z = self._resolve_fitted_gamma(Z, "instrument")
            Kh = self._get_kernel_at_fitted_gamma(
                T, fitted_gamma=gamma_t
            )
            Kf = self._get_kernel_at_fitted_gamma(
                Z, fitted_gamma=gamma_z
            )

        scores = []
        for train, test in KFold(n_splits=self.cv).split(Z):
            n_train = len(train)
            n_test = len(test)
            delta_train = self._get_delta(n_train)

            if fold_specific_gamma:
                gamma_t_train = self._resolve_fitted_gamma(
                    T[train], "treatment"
                )
                gamma_z_train = self._resolve_fitted_gamma(
                    Z[train], "instrument"
                )
                Kh_train = self._get_kernel_at_fitted_gamma(
                    T[train], fitted_gamma=gamma_t_train
                )
                Kh_test_train = self._get_kernel_at_fitted_gamma(
                    T[test], Y=T[train], fitted_gamma=gamma_t_train
                )
                Kf_train = self._get_kernel_at_fitted_gamma(
                    Z[train], fitted_gamma=gamma_z_train
                )
                Kf_test = self._get_kernel_at_fitted_gamma(
                    Z[test], fitted_gamma=gamma_z_train
                )
            else:
                Kh_train = Kh[np.ix_(train, train)]
                Kh_test_train = Kh[np.ix_(test, train)]
                Kf_train = Kf[np.ix_(train, train)]
                Kf_test = Kf[np.ix_(test, test)]

            train_basis, _ = self._symmetric_range_basis(Kf_train)
            test_basis, _ = self._symmetric_range_basis(Kf_test)
            Pz_train = train_basis @ train_basis.T
            Pz_test = test_basis @ test_basis.T
            KMK_train = Kh_train @ Pz_train @ Kh_train
            B_train = Kh_train @ Pz_train @ Y[train]
            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = alpha_scale * delta_train**4
                a = np.linalg.pinv(
                    KMK_train + alpha * Kh_train @ Kh_train
                ) @ B_train
                res = Y[test] - Kh_test_train @ a
                fold_scores.append(
                    _to_scalar(res.T @ Pz_test @ res) / (n_test**2)
                )
            scores.append(fold_scores)

        self.alpha_scales_ = np.asarray(alpha_scales, dtype=float).copy()
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = self.alpha_scales_[
            np.argmin(self.avg_scores)
        ]

        delta = self._get_delta(n)
        self.best_alpha = self.best_alpha_scale * delta**4

        if fold_specific_gamma:
            self.gamma_t_ = self._resolve_fitted_gamma(T, "treatment")
            self.gamma_z_ = self._resolve_fitted_gamma(Z, "instrument")
            Kh = self._get_kernel_at_fitted_gamma(
                T, fitted_gamma=self.gamma_t_
            )
            Kf = self._get_kernel_at_fitted_gamma(
                Z, fitted_gamma=self.gamma_z_
            )
        else:
            self.gamma_t_ = gamma_t
            self.gamma_z_ = gamma_z

        instrument_basis, _ = self._symmetric_range_basis(Kf)
        Pz = instrument_basis @ instrument_basis.T

        self.T = T.copy()
        self.a = np.linalg.pinv(
            Kh @ Pz @ Kh + self.best_alpha * Kh @ Kh
        ) @ Kh @ Pz @ Y
        return self


class ApproxRKHSIV(_BaseRKHSIV):
    """
    Approximate RKHS IV estimator using kernel approximations.

    This class implements an approximate RKHS IV estimator using kernel approximations.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int or float): Number of approximation components.
            Values in (0, 1] are sample fractions with a floor of 10;
            integer-like values greater than 1 are fixed component counts.
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Length scale for the kernel.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scale (str or float): Scale of the regularization parameter.
        kernel_params (dict): Additional parameters for the kernel.
    """

    def __init__(self, kernel_approx='nystrom', n_components=10,
                 kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto'):
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale

    def _resolve_n_components(self, n_samples=None):
        """
        Resolve the effective approximation dimension from ``self.n_components``.

        Supports two input modes:
        - values in (0, 1]: fraction of sample size (rounded, with floor 10)
        - integer-like values greater than 1: fixed component count

        The resolved count is always capped by ``n_samples`` when provided.
        """
        try:
            value = float(self.n_components)
        except Exception as exc:
            raise ValueError("`n_components` must be numeric.") from exc

        if value <= 0:
            raise ValueError("`n_components` must be > 0.")

        if value <= 1:
            if n_samples is None:
                raise ValueError("Fractional `n_components` requires `n_samples`.")
            n_samples_i = int(n_samples)
            if n_samples_i <= 0:
                raise ValueError("`n_samples` must be a positive integer.")
            resolved = max(10, int(round(n_samples_i * value)))
        elif value.is_integer():
            resolved = int(value)
        else:
            raise ValueError(
                "`n_components` must be integer-like > 1 or a fraction in (0, 1]."
            )

        if n_samples is not None:
            resolved = min(resolved, int(n_samples))

        return max(1, resolved)

    def _get_new_approx_instance(self, n_samples=None, fitted_gamma=None):
        """
        Create a new kernel approximation instance.

        Parameters:
            n_samples (int, optional): Sample count used to resolve/cap components.
            fitted_gamma (float, optional): Bandwidth resolved from fitting data.

        Returns:
            object: Kernel approximation instance.
        """
        gamma = self.gamma if fitted_gamma is None else fitted_gamma
        if (self.kernel_approx == 'rbfsampler') and (self.kernel == 'rbf'):
            n_components = self._resolve_n_components(n_samples=n_samples)
            return RBFSampler(gamma=gamma, n_components=n_components, random_state=1)
        elif self.kernel_approx == 'nystrom':
            n_components = self._resolve_n_components(n_samples=n_samples)
            return Nystroem(kernel=self.kernel, gamma=gamma, coef0=self.coef0, degree=self.degree, kernel_params=self.kernel_params,
                            random_state=1, n_components=n_components)
        else:
            raise AttributeError("Invalid kernel approximator")

    def fit(self, Z, T, Y):
        """
        Fit the approximate RKHS IV estimator.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())
        self.gamma_z_ = self._resolve_fitted_gamma(Z, "instrument")
        self.gamma_t_ = self._resolve_fitted_gamma(T, "treatment")
        self.featZ = self._get_new_approx_instance(
            n_samples=n, fitted_gamma=self.gamma_z_
        )
        RootKf = self.featZ.fit_transform(Z)
        self.featT = self._get_new_approx_instance(
            n_samples=n, fitted_gamma=self.gamma_t_
        )
        RootKh = self.featT.fit_transform(T)
        n_feat_f = RootKf.shape[1]
        n_feat_h = RootKh.shape[1]
        Q_system = (
            RootKf.T @ RootKf /
            (2 * n * delta**2) + np.eye(n_feat_f) / 2
        )
        A = RootKh.T @ RootKf
        W = (
            A @ _solve_symmetric(Q_system, A.T)
            + alpha * np.eye(n_feat_h)
        )
        B = A @ _solve_symmetric(Q_system, RootKf.T @ Y)
        self.a = _pinv_symmetric(W) @ B
        self.fitted_delta = delta
        return self

    def predict(self, T):
        """
        Predict outcomes for new treatments.

        Parameters:
            T (array-like): New treatments.

        Returns:
            array-like: Predicted outcomes.
        """
        return self.featT.transform(T) @ self.a

    def score(self, Z, T, Y, delta='auto'):
        """
        Compute the score of the fitted estimator.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.
            delta (str or float): Critical radius.

        Returns:
            float: Score.
        """
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        if _check_auto(delta):
            delta = self._get_delta(n)
        else:
            delta = float(delta)
            if not np.isfinite(delta) or delta <= 0:
                raise ValueError("`delta` must be finite and strictly positive.")

        featZ = self._get_new_approx_instance(
            n_samples=n,
            fitted_gamma=getattr(self, "gamma_z_", None),
        )
        RootKf = featZ.fit_transform(Z)
        n_feat_f = RootKf.shape[1]
        Q_system = (
            RootKf.T @ RootKf /
            (2 * n * delta**2) + np.eye(n_feat_f) / 2
        )
        Y_pred = self.predict(T)
        res = RootKf.T @ (Y - Y_pred)
        return _to_scalar(
            res.T @ _solve_symmetric(Q_system, res)
        ) / n**2


class ApproxRKHSIVCV(ApproxRKHSIV):
    """
    Approximate RKHS IV estimator with cross-validation using kernel approximations.

    Each feature approximation is fitted on its training fold and then used to
    transform the corresponding held-out fold.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int or float): Number of approximation components.
            Values in (0, 1] are sample fractions with a floor of 10;
            integer-like values greater than 1 are fixed component counts.
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Length scale for the kernel.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        n_alphas (int): Number of alpha scales to try.
        cv (int): Number of folds for cross-validation.
        kernel_params (dict): Additional parameters for the kernel.
    """

    def __init__(self, kernel_approx='nystrom', n_components=10,
                 kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = n_alphas
        self.cv = cv

    def fit(self, Z, T, Y):
        """
        Fit the approximate RKHS IV estimator with cross-validation.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Y = _to_column_vector(Y)
        n = Y.shape[0]

        alpha_scales = np.asarray(
            self._get_alpha_scales(), dtype=float
        )
        scores = []
        for train, test in KFold(n_splits=self.cv).split(np.arange(n)):
            n_train = len(train)
            n_test = len(test)
            delta_train = self._get_delta(n_train)
            delta_test = self._get_delta(n_test)

            Z_train = _safe_indexing(Z, train)
            Z_test = _safe_indexing(Z, test)
            T_train = _safe_indexing(T, train)
            T_test = _safe_indexing(T, test)

            gamma_z_train = self._resolve_fitted_gamma(
                Z_train, "instrument"
            )
            gamma_t_train = self._resolve_fitted_gamma(
                T_train, "treatment"
            )
            feat_z_train = self._get_new_approx_instance(
                n_samples=n_train, fitted_gamma=gamma_z_train
            )
            feat_t_train = self._get_new_approx_instance(
                n_samples=n_train, fitted_gamma=gamma_t_train
            )
            RootKf_train = feat_z_train.fit_transform(Z_train)
            RootKf_test = feat_z_train.transform(Z_test)
            RootKh_train = feat_t_train.fit_transform(T_train)
            RootKh_test = feat_t_train.transform(T_test)

            n_feat_f_train = RootKf_train.shape[1]
            n_feat_f_test = RootKf_test.shape[1]
            n_feat_h_train = RootKh_train.shape[1]
            Q_train_system = (
                RootKf_train.T @ RootKf_train /
                (2 * n_train * (delta_train**2))
                + np.eye(n_feat_f_train) / 2
            )
            Q_test_system = (
                RootKf_test.T @ RootKf_test /
                (2 * n_test * (delta_test**2))
                + np.eye(n_feat_f_test) / 2
            )
            A_train = RootKh_train.T @ RootKf_train
            AQA_train = A_train @ _solve_symmetric(
                Q_train_system, A_train.T
            )
            B_train = A_train @ _solve_symmetric(
                Q_train_system, RootKf_train.T @ Y[train]
            )
            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = self._get_alpha(delta_train, alpha_scale)
                a = _pinv_symmetric(
                    AQA_train + alpha * np.eye(n_feat_h_train)
                ) @ B_train
                res = RootKf_test.T @ (Y[test] - RootKh_test @ a)
                fold_scores.append(
                    _to_scalar(
                        res.T @ _solve_symmetric(Q_test_system, res)
                    ) / (n_test**2)
                )
            scores.append(fold_scores)

        self.alpha_scales_ = alpha_scales.copy()
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = float(
            self.alpha_scales_[np.argmin(self.avg_scores)]
        )

        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        self.gamma_z_ = self._resolve_fitted_gamma(Z, "instrument")
        self.gamma_t_ = self._resolve_fitted_gamma(T, "treatment")
        self.featZ = self._get_new_approx_instance(
            n_samples=n, fitted_gamma=self.gamma_z_
        )
        RootKf = self.featZ.fit_transform(Z)
        self.featT = self._get_new_approx_instance(
            n_samples=n, fitted_gamma=self.gamma_t_
        )
        RootKh = self.featT.fit_transform(T)
        n_feat_f = RootKf.shape[1]
        n_feat_h = RootKh.shape[1]
        Q_system = (
            RootKf.T @ RootKf /
            (2 * n * delta**2) + np.eye(n_feat_f) / 2
        )
        A = RootKh.T @ RootKf
        W = (
            A @ _solve_symmetric(Q_system, A.T)
            + self.best_alpha * np.eye(n_feat_h)
        )
        B = A @ _solve_symmetric(Q_system, RootKf.T @ Y)
        self.a = _pinv_symmetric(W) @ B
        self.fitted_delta = delta
        return self


class ApproxRKHSIVL2(ApproxRKHSIV):
    """
    Approximate RKHS IV estimator with L2 regularization.

    Instrument projections are represented by thin-SVD range bases. The
    sample-space empirical-L2 equation is contracted through the learner thin
    SVD so its pseudoinverse truncation is retained in a feature-sized system.
    """

    @staticmethod
    def _feature_gram_range_basis(features):
        """Return the range retained by the default Gram pinv cutoff."""
        left_vectors, singular_values, _ = np.linalg.svd(
            features, full_matrices=False
        )
        if not singular_values.size or singular_values[0] == 0:
            return left_vectors[:, :0], singular_values[:0]
        cutoff = np.sqrt(_DEFAULT_PINV_RCOND) * singular_values[0]
        retained = singular_values > cutoff
        return left_vectors[:, retained], singular_values[retained]

    @classmethod
    def _l2_reduced_system_terms(
        cls,
        instrument_features,
        learner_features,
        outcome,
    ):
        instrument_basis, _ = cls._feature_gram_range_basis(
            instrument_features
        )
        learner_left, learner_singular, learner_right_t = np.linalg.svd(
            learner_features, full_matrices=False
        )
        projected_learner_basis = instrument_basis.T @ learner_left
        projected_outcome = instrument_basis.T @ outcome
        squared_singular = learner_singular**2
        weighted_projected_basis = (
            projected_learner_basis * squared_singular[None, :]
        )
        return (
            weighted_projected_basis.T @ weighted_projected_basis,
            np.diag(squared_singular**2),
            squared_singular[:, None]
            * (projected_learner_basis.T @ projected_outcome),
            learner_singular,
            learner_right_t,
            learner_left,
        )

    @staticmethod
    def _solve_l2_feature_system(
        projected_moment,
        l2_penalty,
        rhs,
        learner_singular,
        learner_right_t,
        learner_left,
        alpha,
    ):
        reduced_coefficient = np.linalg.pinv(
            projected_moment + alpha * l2_penalty
        ) @ rhs
        feature_coefficient = learner_right_t.T @ (
            learner_singular[:, None] * reduced_coefficient
        )
        dual_coefficient = learner_left @ reduced_coefficient
        return feature_coefficient, dual_coefficient

    def fit(self, Z, T, Y):
        """
        Fit the approximate RKHS IV L2 estimator.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        delta = self._get_delta(n)
        alpha = (
            delta**4
            if _check_auto(self.alpha_scale)
            else self._get_alpha(delta, self.alpha_scale)
        )

        self.gamma_z_ = self._resolve_fitted_gamma(Z, "instrument")
        self.gamma_t_ = self._resolve_fitted_gamma(T, "treatment")
        self.featZ = self._get_new_approx_instance(
            n_samples=n, fitted_gamma=self.gamma_z_
        )
        RootKf = self.featZ.fit_transform(Z)
        self.featT = self._get_new_approx_instance(
            n_samples=n, fitted_gamma=self.gamma_t_
        )
        RootKh = self.featT.fit_transform(T)

        equation_terms = self._l2_reduced_system_terms(
            RootKf, RootKh, Y
        )
        self.theta_, self.a = self._solve_l2_feature_system(
            *equation_terms, alpha
        )
        self.RootKh_train_ = RootKh
        self.fitted_delta = delta
        return self

    def predict(self, T):
        """
        Predict outcomes for new treatments.

        Parameters:
            T (array-like): New treatments.

        Returns:
            array-like: Predicted outcomes.
        """
        return self.featT.transform(T) @ self.theta_

    def score(self, Z, T, Y, delta='auto'):
        """
        Compute the L2 score of the fitted estimator.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.
            delta (str or float): Kept for API compatibility.

        Returns:
            float: Score.
        """
        _ = delta
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        featZ = self._get_new_approx_instance(
            n_samples=n,
            fitted_gamma=getattr(self, "gamma_z_", None),
        )
        RootKf = featZ.fit_transform(Z)
        instrument_basis, _ = self._feature_gram_range_basis(RootKf)
        Y_pred = self.predict(T)
        projected_residual = instrument_basis.T @ (Y - Y_pred)
        return _to_scalar(
            projected_residual.T @ projected_residual
        ) / n**2


class ApproxRKHSIVL2CV(ApproxRKHSIVL2):
    """
    Approximate RKHS IV L2 estimator with cross-validation.

    Each feature approximation is fitted on its training fold and then used to
    transform the corresponding held-out fold.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int or float): Number of approximation components.
            Values in (0, 1] are sample fractions with a floor of 10;
            integer-like values greater than 1 are fixed component counts.
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Length scale for the kernel.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        kernel_params (dict): Additional parameters for the kernel.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        n_alphas (int): Number of alpha scales to try.
        cv (int): Number of folds for cross-validation.
    """

    def __init__(self, kernel_approx='nystrom', n_components=10,
                 kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = n_alphas
        self.cv = cv

    def fit(self, Z, T, Y):
        """
        Fit the approximate RKHS IV L2 estimator with cross-validation.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        Y = _to_column_vector(Y)
        n = Y.shape[0]

        alpha_scales = np.asarray(
            self._get_alpha_scales(), dtype=float
        )
        scores = []

        for train, test in KFold(n_splits=self.cv).split(np.arange(n)):
            n_train = len(train)
            n_test = len(test)
            delta_train = self._get_delta(n_train)

            Z_train = _safe_indexing(Z, train)
            Z_test = _safe_indexing(Z, test)
            T_train = _safe_indexing(T, train)
            T_test = _safe_indexing(T, test)

            gamma_z_train = self._resolve_fitted_gamma(
                Z_train, "instrument"
            )
            gamma_t_train = self._resolve_fitted_gamma(
                T_train, "treatment"
            )
            feat_z_train = self._get_new_approx_instance(
                n_samples=n_train, fitted_gamma=gamma_z_train
            )
            feat_t_train = self._get_new_approx_instance(
                n_samples=n_train, fitted_gamma=gamma_t_train
            )
            RootKf_train = feat_z_train.fit_transform(Z_train)
            RootKf_test = feat_z_train.transform(Z_test)
            RootKh_train = feat_t_train.fit_transform(T_train)
            RootKh_test = feat_t_train.transform(T_test)
            test_basis, _ = self._feature_gram_range_basis(RootKf_test)
            equation_terms = self._l2_reduced_system_terms(
                RootKf_train, RootKh_train, Y[train]
            )

            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = float(alpha_scale) * (delta_train**4)
                coefficient, _ = self._solve_l2_feature_system(
                    *equation_terms, alpha
                )
                residual = Y[test] - RootKh_test @ coefficient
                projected_residual = test_basis.T @ residual
                fold_scores.append(
                    _to_scalar(
                        projected_residual.T @ projected_residual
                    ) / (n_test**2)
                )
            scores.append(fold_scores)

        self.alpha_scales_ = alpha_scales.copy()
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = float(
            self.alpha_scales_[np.argmin(self.avg_scores)]
        )
        delta = self._get_delta(n)
        self.best_alpha = self.best_alpha_scale * (delta**4)

        self.gamma_z_ = self._resolve_fitted_gamma(Z, "instrument")
        self.gamma_t_ = self._resolve_fitted_gamma(T, "treatment")
        self.featZ = self._get_new_approx_instance(
            n_samples=n, fitted_gamma=self.gamma_z_
        )
        RootKf = self.featZ.fit_transform(Z)
        self.featT = self._get_new_approx_instance(
            n_samples=n, fitted_gamma=self.gamma_t_
        )
        RootKh = self.featT.fit_transform(T)
        equation_terms = self._l2_reduced_system_terms(
            RootKf, RootKh, Y
        )
        self.theta_, self.a = self._solve_l2_feature_system(
            *equation_terms, self.best_alpha
        )
        self.RootKh_train_ = RootKh
        self.fitted_delta = delta
        return self
