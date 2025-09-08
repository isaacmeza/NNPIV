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
"""

# Licensed under the MIT License.

from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
from sklearn.model_selection import KFold
from sklearn.kernel_approximation import Nystroem, RBFSampler
import numpy as np
import scipy


def _check_auto(param):
    return (isinstance(param, str) and (param == 'auto'))


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
        n = Y.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        Kh = self._get_kernel(T)
        Kf = self._get_kernel(Z)

        RootKf = scipy.linalg.sqrtm(Kf).astype(float)
        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        self.T = T.copy()
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
        return self._get_kernel(T_test, Y=self.T) @ self.a

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
        n = Y.shape[0]
        delta = self._get_delta(n)
        Kf = self._get_kernel(Z)
        RootKf = scipy.linalg.sqrtm(Kf).astype(float)
        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf
        Y_pred = self.predict(T)
        return ((Y - Y_pred).T @ M @ (Y - Y_pred))[0, 0] / n**2


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
        n = Y.shape[0]

        Kh = self._get_kernel(T)
        Kf = self._get_kernel(Z)

        RootKf = scipy.linalg.sqrtm(Kf).astype(float)

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta_test = self._get_delta(n_test)
        delta = self._get_delta(n)
        scores = []
        for it, (train, test) in enumerate(KFold(n_splits=self.cv).split(Z)):
            M_train = RootKf[np.ix_(train, train)] @ np.linalg.inv(
                Kf[np.ix_(train, train)] / (2 * n_train * (delta_train**2)) + np.eye(len(train)) / 2) @ RootKf[np.ix_(train, train)]
            M_test = RootKf[np.ix_(test, test)] @ np.linalg.inv(
                Kf[np.ix_(test, test)] / (2 * n_test * (delta_test**2)) + np.eye(len(test)) / 2) @ RootKf[np.ix_(test, test)]
            Kh_train = Kh[np.ix_(train, train)]
            KMK_train = Kh_train @ M_train @ Kh_train
            B_train = Kh_train @ M_train @ Y[train]
            scores.append([])
            for alpha_scale in alpha_scales:
                alpha = self._get_alpha(delta_train, alpha_scale)
                a = np.linalg.pinv(KMK_train + alpha * Kh_train) @ B_train
                res = Y[test] - Kh[np.ix_(test, train)] @ a
                scores[it].append((res.T @ M_test @ res)[
                                  0, 0] / (res.shape[0]**2))

        self.alpha_scales = alpha_scales
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = alpha_scales[np.argmin(self.avg_scores)]

        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        M = RootKf @ np.linalg.inv(
            Kf / (2 * n * delta**2) + np.eye(n) / 2) @ RootKf

        self.T = T.copy()
        self.a = np.linalg.pinv(
            Kh @ M @ Kh + self.best_alpha * Kh) @ Kh @ M @ Y
        return self


class RKHSIVL2(_BaseRKHSIV):
    """
    RKHS IV estimator with L2 regularization.

    This class implements an RKHS IV estimator with L2 regularization.

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
        n = Y.shape[0]
        delta = self._get_delta(n)
        alpha = delta**4

        Kh = self._get_kernel(T)
        Kf = self._get_kernel(Z)

        M = np.linalg.pinv(Kf) @ Kf
        self.T = T.copy()
        self.a = np.linalg.pinv(Kh @ M @ Kh + alpha * Kh @ Kh) @ Kh @ M @ Y
        return self

    def predict(self, T_test):
        """
        Predict outcomes for new treatments.

        Parameters:
            T_test (array-like): New treatments.

        Returns:
            array-like: Predicted outcomes.
        """
        return self._get_kernel(T_test, Y=self.T) @ self.a


class RKHSIVL2CV(RKHSIVL2):
    """
    RKHS IV estimator with L2 regularization and cross-validation.

    This class implements an RKHS IV estimator with L2 regularization and cross-validation.

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
        n = Y.shape[0]

        Kh = self._get_kernel(T)
        Kf = self._get_kernel(Z)

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta = self._get_delta(n)
        scores = []
        for it, (train, test) in enumerate(KFold(n_splits=self.cv).split(Z)):
            M_train = np.linalg.pinv(Kf[np.ix_(train, train)]) @ Kf[np.ix_(train, train)]
            M_test = np.linalg.pinv(Kf[np.ix_(test, test)]) @ Kf[np.ix_(test, test)]
            Kh_train = Kh[np.ix_(train, train)]
            KMK_train = Kh_train @ M_train @ Kh_train
            B_train = Kh_train @ M_train @ Y[train]
            scores.append([])
            for alpha_scale in alpha_scales:
                alpha = alpha_scale * delta_train**4
                a = np.linalg.pinv(KMK_train + alpha * Kh_train @ Kh_train) @ B_train
                res = Y[test] - Kh[np.ix_(test, train)] @ a
                scores[it].append((res.T @ M_test @ res)[
                                  0, 0] / (res.shape[0]**2))

        self.alpha_scales = alpha_scales
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = alpha_scales[np.argmin(self.avg_scores)]

        delta = self._get_delta(n)
        self.best_alpha = self.best_alpha_scale * delta**4

        M = np.linalg.pinv(Kf) @ Kf

        self.T = T.copy()
        self.a = np.linalg.pinv(
            Kh @ M @ Kh + self.best_alpha * Kh @ Kh) @ Kh @ M @ Y
        return self


class ApproxRKHSIV(_BaseRKHSIV):
    """
    Approximate RKHS IV estimator using kernel approximations.

    This class implements an approximate RKHS IV estimator using kernel approximations.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int): Number of approximation components.
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

    def _get_new_approx_instance(self):
        """
        Create a new kernel approximation instance.

        Returns:
            object: Kernel approximation instance.
        """
        if (self.kernel_approx == 'rbfsampler') and (self.kernel == 'rbf'):
            return RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=1)
        elif self.kernel_approx == 'nystrom':
            return Nystroem(kernel=self.kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree, kernel_params=self.kernel_params,
                            random_state=1, n_components=self.n_components)
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
        n = Y.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())
        self.featZ = self._get_new_approx_instance()
        RootKf = self.featZ.fit_transform(Z)
        self.featT = self._get_new_approx_instance()
        RootKh = self.featT.fit_transform(T)
        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ Y
        self.a = np.linalg.pinv(W) @ B
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
        n = Y.shape[0]
        delta = self._get_delta(n)
        featZ = self._get_new_approx_instance()
        RootKf = featZ.fit_transform(Z)
        RootKh = self.featT.fit_transform(T)
        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        Y_pred = self.predict(T)
        res = RootKf.T @ (Y - Y_pred)
        return (res.T @ Q @ res)[0, 0] / n**2


class ApproxRKHSIVCV(ApproxRKHSIV):
    """
    Approximate RKHS IV estimator with cross-validation using kernel approximations.

    This class implements an approximate RKHS IV estimator with cross-validation using kernel approximations.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int): Number of approximation components.
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
        n = Y.shape[0]

        self.featZ = self._get_new_approx_instance()
        RootKf = self.featZ.fit_transform(Z)
        self.featT = self._get_new_approx_instance()
        RootKh = self.featT.fit_transform(T)

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta_test = self._get_delta(n_test)
        delta = self._get_delta(n)
        scores = []
        for it, (train, test) in enumerate(KFold(n_splits=self.cv).split(Z)):
            RootKf_train, RootKf_test = RootKf[train], RootKf[test]
            RootKh_train, RootKh_test = RootKh[train], RootKh[test]
            Q_train = np.linalg.pinv(
                RootKf_train.T @ RootKf_train / (2 * n_train * (delta_train**2)) + np.eye(self.n_components) / 2)
            Q_test = np.linalg.pinv(
                RootKf_test.T @ RootKf_test / (2 * n_test * (delta_test**2)) + np.eye(self.n_components) / 2)
            A_train = RootKh_train.T @ RootKf_train
            AQA_train = A_train @ Q_train @ A_train.T
            B_train = A_train @ Q_train @ RootKf_train.T @ Y[train]
            scores.append([])
            for alpha_scale in alpha_scales:
                alpha = self._get_alpha(delta_train, alpha_scale)
                a = np.linalg.pinv(AQA_train + alpha *
                                   np.eye(self.n_components)) @ B_train
                res = RootKf_test.T @ (Y[test] - RootKh_test @ a)
                scores[it].append((res.T @ Q_test @ res)[
                                  0, 0] / (len(test)**2))

        self.alpha_scales = alpha_scales
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = alpha_scales[np.argmin(self.avg_scores)]

        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + self.best_alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ Y
        self.a = np.linalg.pinv(W) @ B
        self.fitted_delta = delta
        return self
