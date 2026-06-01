"""
This module provides implementations of nested NPIV estimators for RKHS function classes.

Classes:
    _BaseRKHS2IV: Base class for nested RKHS IV methods.
    RKHS2IV: Nested RKHS IV estimator (alternate simultaneous variant).
    RKHS2IVCV: Cross-validated RKHS2IV estimator.
    RKHS2IVL2: Nested RKHS IV estimator aligned with Appendix J / Algorithm 2.
    RKHS2IVL2CV: Cross-validated RKHS2IVL2 estimator.
    ApproxRKHS2IV: Nystrom/RFF approximate RKHS2IV estimator.
    ApproxRKHS2IVCV: Cross-validated approximate RKHS2IV estimator.
    ApproxRKHS2IVL2: Nystrom/RFF approximate RKHS2IVL2 estimator.
    ApproxRKHS2IVL2CV: Cross-validated approximate RKHS2IVL2 estimator.
"""

# Licensed under the MIT License.

from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
from sklearn.model_selection import KFold
from sklearn.kernel_approximation import Nystroem, RBFSampler
import numpy as np


def _check_auto(param):
    return (isinstance(param, str) and (param == 'auto'))


def _to_column_vector(y):
    arr = np.asarray(y)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    raise ValueError("Y must be a 1D array or a 2D column vector.")


def _to_scalar(x):
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(
            "Expected scalar quadratic form, got array with "
            f"shape={arr.shape!r} and size={arr.size}."
        )
    return float(arr.reshape(-1)[0])


class _BaseRKHS2IV:
    """
    Base class for nested RKHS IV methods.

    This class provides common functionality for nested RKHS IV estimators.
    """

    def __init__(self, *args, **kwargs):
        return

    def _get_delta(self, n):
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

    def _resolve_n_components(self, n_samples=None):
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
                "`n_components` must be integer-like >= 1 or a fraction in (0, 1]."
            )

        if n_samples is not None:
            resolved = min(resolved, int(n_samples))

        return max(1, resolved)

    def _get_new_approx_instance(self, n_samples=None):
        if (self.kernel_approx == 'rbfsampler') and (self.kernel == 'rbf'):
            n_components = self._resolve_n_components(n_samples=n_samples)
            return RBFSampler(gamma=self.gamma, n_components=n_components, random_state=1)
        if self.kernel_approx == 'nystrom':
            n_components = self._resolve_n_components(n_samples=n_samples)
            return Nystroem(kernel=self.kernel, gamma=self.gamma, coef0=self.coef0,
                            degree=self.degree, kernel_params=self.kernel_params,
                            random_state=1, n_components=n_components)
        raise AttributeError("Invalid kernel approximator")

    def _validate_subset_inputs(self, n, subsetted=False, subset_ind1=None, subset_ind2=None):
        if not subsetted:
            return None, None

        if subset_ind1 is None:
            raise ValueError("subset_ind1 must be provided when subsetted is True")

        subset_ind1 = np.asarray(subset_ind1).reshape(-1)
        if subset_ind1.shape[0] != n:
            raise ValueError("subset_ind1 must have the same length as Y")

        if subset_ind2 is not None:
            subset_ind2 = np.asarray(subset_ind2).reshape(-1)
            if subset_ind2.shape[0] != n:
                raise ValueError("subset_ind2 must have the same length as Y")

        ind1 = np.flatnonzero(subset_ind1 == 1)
        ind2 = (np.flatnonzero(subset_ind2 == 1)
                if subset_ind2 is not None else np.flatnonzero(subset_ind1 == 0))

        if ind1.size == 0:
            raise ValueError("subset_ind1 selects zero observations.")
        if ind2.size == 0:
            raise ValueError("subset_ind2/subset_ind1 complement selects zero observations.")

        return ind1, ind2

    def _local_subset_indices(self, fold_indices, global_indices):
        return np.flatnonzero(np.isin(fold_indices, global_indices, assume_unique=False))

    def _projector_from_kernel(self, K, ridge):
        n = K.shape[0]
        if ridge:
            return np.linalg.pinv(K + np.eye(n)) @ K
        return np.linalg.pinv(K) @ K

    def _lifted_subset_projector(self, K_block, subset_local_indices, scale_n, ridge):
        subset_local_indices = np.asarray(subset_local_indices, dtype=int)
        if subset_local_indices.size == 0:
            raise ValueError("Subset projector requested with zero selected rows.")

        n_block = K_block.shape[0]
        I_subset = np.eye(n_block)[subset_local_indices, :]
        K_subset = I_subset @ K_block @ I_subset.T
        P_subset = self._projector_from_kernel(K_subset, ridge=ridge)

        return (scale_n / subset_local_indices.size) * I_subset.T @ P_subset @ I_subset

    def _build_projectors(self, Kc, Kd, n_scale, ridge, subsetted=False,
                          ind1_local=None, ind2_local=None):
        if not subsetted:
            Pc = self._projector_from_kernel(Kc, ridge=ridge)
            Pd = self._projector_from_kernel(Kd, ridge=ridge)
            return Pc, Pd

        Pc = self._lifted_subset_projector(Kc, ind2_local, scale_n=n_scale, ridge=ridge)
        Pd = self._lifted_subset_projector(Kd, ind1_local, scale_n=n_scale, ridge=ridge)
        return Pc, Pd

    def _solve_coefficients(self, Ka, Kb, Pc, Pd, Iw, Y, alpha, l2_variant):
        n = Y.shape[0]
        Id = np.eye(n)

        KbPcKa_inv = np.linalg.pinv(Kb @ Pc @ Iw @ Ka)

        if l2_variant:
            M = Ka @ (
                - Iw @ Pc
                + (Pd + Iw @ Pc @ Iw + alpha * Id)
                @ Ka @ KbPcKa_inv @ Kb
                @ (Pc + alpha * Id)
            ) @ Kb
            b = np.linalg.pinv(M) @ Ka @ Pd @ Y
            a = KbPcKa_inv @ Kb @ (Pc + alpha * Id) @ Kb @ b
        else:
            M = Ka @ (
                - Iw @ Pc
                + (Pd @ Ka + Iw @ Pc @ Iw @ Ka + alpha * Id)
                @ KbPcKa_inv
                @ (Kb @ Pc + alpha * Id)
            ) @ Kb
            b = np.linalg.pinv(M) @ Ka @ Pd @ Y
            a = KbPcKa_inv @ (Kb @ Pc + alpha * Id) @ Kb @ b

        return a, b

    def _as_candidate_values(self, value, name, allow_auto=False, positive=True):
        if isinstance(value, np.ndarray):
            raw_vals = list(value.reshape(-1))
        elif isinstance(value, (list, tuple)):
            raw_vals = list(value)
        else:
            raw_vals = [value]

        if len(raw_vals) == 0:
            raise ValueError(f"`{name}` candidate grid must be non-empty.")

        parsed = []
        for raw in raw_vals:
            if isinstance(raw, str):
                if allow_auto and raw == 'auto':
                    parsed.append(raw)
                    continue
                raise ValueError(f"`{name}` candidates must be numeric; got {raw!r}.")
            value_f = float(raw)
            if not np.isfinite(value_f):
                raise ValueError(f"`{name}` candidates must be finite; got {raw!r}.")
            if positive and value_f <= 0:
                raise ValueError(f"`{name}` candidates must be > 0; got {raw!r}.")
            parsed.append(value_f)
        return parsed

    def _normalize_positive_grid(self, values, name):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"`{name}` must be non-empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"`{name}` must be finite.")
        if np.any(arr <= 0):
            raise ValueError(f"`{name}` must contain strictly positive values.")
        return arr

    def _expand_alpha_grid_once(self, alpha_scales, best_index):
        alpha_scales = self._normalize_positive_grid(alpha_scales, "alpha_scales")
        if alpha_scales.size < 2:
            return alpha_scales, False

        if best_index == 0:
            ratio = alpha_scales[1] / alpha_scales[0]
            ratio = ratio if np.isfinite(ratio) and ratio > 0 else 10.0
            expanded = np.concatenate([[alpha_scales[0] / ratio], alpha_scales])
            return expanded, True

        if best_index == alpha_scales.size - 1:
            ratio = alpha_scales[-1] / alpha_scales[-2]
            ratio = ratio if np.isfinite(ratio) and ratio > 0 else 10.0
            expanded = np.concatenate([alpha_scales, [alpha_scales[-1] * ratio]])
            return expanded, True

        return alpha_scales, False

    def _run_alpha_cv_with_optional_expansion(self, cv_runner, alpha_scales):
        alpha_initial = self._normalize_positive_grid(alpha_scales, "alpha_scales")
        cv_result = cv_runner(alpha_initial)
        best_index = int(np.argmin(cv_result["avg_scores"]))
        best_is_boundary = best_index in (0, alpha_initial.size - 1)

        expanded = False
        alpha_used = alpha_initial
        if getattr(self, "expand_alpha_grid", True) and best_is_boundary:
            alpha_expanded, expanded = self._expand_alpha_grid_once(alpha_initial, best_index)
            if expanded:
                cv_result = cv_runner(alpha_expanded)
                alpha_used = alpha_expanded
                best_index = int(np.argmin(cv_result["avg_scores"]))
                best_is_boundary = best_index in (0, alpha_used.size - 1)

        cv_result["alpha_scales_initial"] = alpha_initial
        cv_result["alpha_scales_used"] = alpha_used
        cv_result["alpha_grid_expanded"] = expanded
        cv_result["best_index"] = best_index
        cv_result["best_alpha_scale"] = float(alpha_used[best_index])
        cv_result["best_alpha_is_boundary"] = best_is_boundary
        cv_result["best_score"] = float(cv_result["avg_scores"][best_index])
        return cv_result

    def _set_cv_diagnostics(self, cv_result):
        self.cv_n_valid_folds_ = int(cv_result["n_valid_folds"])
        self.cv_fold_scores_ = np.asarray(cv_result["fold_scores"], dtype=float)
        self.cv_alpha_scales_initial_ = np.asarray(cv_result["alpha_scales_initial"], dtype=float)
        self.cv_alpha_scales_used_ = np.asarray(cv_result["alpha_scales_used"], dtype=float)
        self.cv_alpha_grid_expanded_ = bool(cv_result["alpha_grid_expanded"])
        self.cv_best_alpha_is_boundary_ = bool(cv_result["best_alpha_is_boundary"])

    def _run_exact_cv(self, Ka, Kb, Kc, Kd, Iw, Y, alpha_scales,
                      n_train, n_test, delta_train, subsetted, ind1, ind2,
                      ridge, l2_variant):
        alpha_scales = self._normalize_positive_grid(alpha_scales, "alpha_scales")
        fold_scores_all = []

        for train, test in KFold(n_splits=self.cv).split(Y):
            Ka_train = Ka[np.ix_(train, train)]
            Kb_train = Kb[np.ix_(train, train)]
            Kc_train = Kc[np.ix_(train, train)]
            Kd_train = Kd[np.ix_(train, train)]

            Kc_test = Kc[np.ix_(test, test)]
            Kd_test = Kd[np.ix_(test, test)]

            if subsetted:
                train_ind1 = self._local_subset_indices(train, ind1)
                train_ind2 = self._local_subset_indices(train, ind2)
                test_ind1 = self._local_subset_indices(test, ind1)
                test_ind2 = self._local_subset_indices(test, ind2)

                if (train_ind1.size == 0 or train_ind2.size == 0
                        or test_ind1.size == 0 or test_ind2.size == 0):
                    continue

                Pc_train, Pd_train = self._build_projectors(
                    Kc_train, Kd_train, n_scale=n_train, ridge=ridge, subsetted=True,
                    ind1_local=train_ind1, ind2_local=train_ind2
                )
                Pc_test, Pd_test = self._build_projectors(
                    Kc_test, Kd_test, n_scale=n_test, ridge=ridge, subsetted=True,
                    ind1_local=test_ind1, ind2_local=test_ind2
                )
            else:
                Pc_train, Pd_train = self._build_projectors(
                    Kc_train, Kd_train, n_scale=n_train, ridge=ridge, subsetted=False
                )
                Pc_test, Pd_test = self._build_projectors(
                    Kc_test, Kd_test, n_scale=n_test, ridge=ridge, subsetted=False
                )

            Iw_train = Iw[np.ix_(train, train)]
            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = float(alpha_scale) * (delta_train**4)
                a, b = self._solve_coefficients(
                    Ka_train, Kb_train, Pc_train, Pd_train,
                    Iw_train, Y[train], alpha=alpha, l2_variant=l2_variant
                )

                res1 = Y[test] - Ka[np.ix_(test, train)] @ a
                res2 = Ka[np.ix_(test, train)] @ a - Kb[np.ix_(test, train)] @ b
                fold_scores.append(
                    (res1.T @ Pd_test @ res1)[0, 0] / (res1.shape[0]**2)
                    + (res2.T @ Pc_test @ res2)[0, 0] / (res2.shape[0]**2)
                )
            fold_scores_all.append(fold_scores)

        n_valid_folds = len(fold_scores_all)
        if n_valid_folds == 0:
            raise ValueError(
                "No valid CV folds remain under subset constraints. "
                "Ensure both subsets are represented in each fold or reduce cv."
            )

        fold_scores_arr = np.asarray(fold_scores_all, dtype=float)
        return {
            "fold_scores": fold_scores_arr,
            "avg_scores": np.mean(fold_scores_arr, axis=0),
            "n_valid_folds": n_valid_folds,
        }


class RKHS2IV(_BaseRKHS2IV):
    """
    Nested RKHS IV estimator.

    Note:
        This class is an alternate simultaneous variant.
        The Appendix J / Algorithm 2 closed-form implementation target is ``RKHS2IVL2``.
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

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )

        Iw = np.eye(n) if W is None else np.diag(np.asarray(W).reshape(-1))

        delta = self._get_delta(n)
        alpha = delta**4

        Ka = self._get_kernel(A)
        Kb = self._get_kernel(B)
        Kc = self._get_kernel(C)
        Kd = self._get_kernel(D)

        Pc, Pd = self._build_projectors(
            Kc, Kd, n_scale=n, ridge=True, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )

        self.a, self.b = self._solve_coefficients(
            Ka, Kb, Pc, Pd, Iw, Y, alpha=alpha, l2_variant=False
        )
        self.A = A.copy()
        self.B = B.copy()
        return self

    def predict(self, B_test, *args):
        if len(args) == 0:
            return self._get_kernel(B_test, Y=self.B) @ self.b
        if len(args) == 1:
            A_test = args[0]
            return (self._get_kernel(B_test, Y=self.B) @ self.b,
                    self._get_kernel(A_test, Y=self.A) @ self.a)
        raise ValueError("predict expects at most two arguments, B_test and optionally A_test")


class RKHS2IVCV(RKHS2IV):
    """
    Cross-validated RKHS2IV estimator.

    Note:
        This class cross-validates the alternate simultaneous variant.
        The Appendix J / Algorithm 2 closed-form implementation target is ``RKHS2IVL2CV``.
    """

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6,
                 expand_alpha_grid=True):
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
        self.expand_alpha_grid = expand_alpha_grid

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )

        Iw = np.eye(n) if W is None else np.diag(np.asarray(W).reshape(-1))

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta = self._get_delta(n)

        gamma_candidates = self._as_candidate_values(self.gamma, "gamma", allow_auto=True, positive=True)
        candidate_summaries = []
        candidate_states = []

        for gamma_candidate in gamma_candidates:
            self.gamma = gamma_candidate
            Ka = self._get_kernel(A)
            Kb = self._get_kernel(B)
            Kc = self._get_kernel(C)
            Kd = self._get_kernel(D)

            cv_result = self._run_alpha_cv_with_optional_expansion(
                lambda alpha_grid: self._run_exact_cv(
                    Ka, Kb, Kc, Kd, Iw, Y, alpha_grid,
                    n_train=n_train, n_test=n_test, delta_train=delta_train,
                    subsetted=subsetted, ind1=ind1, ind2=ind2,
                    ridge=True, l2_variant=False,
                ),
                alpha_scales,
            )
            cv_result["best_alpha"] = cv_result["best_alpha_scale"] * (delta**4)
            candidate_summaries.append({
                "gamma": gamma_candidate,
                "best_alpha_scale": cv_result["best_alpha_scale"],
                "best_score": cv_result["best_score"],
                "n_valid_folds": cv_result["n_valid_folds"],
                "alpha_grid_expanded": cv_result["alpha_grid_expanded"],
                "best_alpha_is_boundary": cv_result["best_alpha_is_boundary"],
            })
            candidate_states.append((gamma_candidate, Ka, Kb, Kc, Kd, cv_result))

        best_candidate_idx = int(np.argmin([state[-1]["best_score"] for state in candidate_states]))
        best_gamma, Ka, Kb, Kc, Kd, cv_result = candidate_states[best_candidate_idx]
        self.gamma = best_gamma
        self.best_gamma_ = best_gamma
        self.cv_gamma_grid_ = list(gamma_candidates)
        self.cv_candidate_summaries_ = candidate_summaries

        self.alpha_scales = cv_result["alpha_scales_used"]
        self.avg_scores = cv_result["avg_scores"]
        self.best_alpha_scale = cv_result["best_alpha_scale"]
        self.best_alpha = cv_result["best_alpha"]
        self._set_cv_diagnostics(cv_result)

        Pc, Pd = self._build_projectors(
            Kc, Kd, n_scale=n, ridge=True, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )
        self.a, self.b = self._solve_coefficients(
            Ka, Kb, Pc, Pd, Iw, Y, alpha=self.best_alpha, l2_variant=False
        )

        self.A = A.copy()
        self.B = B.copy()
        return self


class RKHS2IVL2(_BaseRKHS2IV):
    """
    Nested RKHS IV estimator with L2 regularization.

    Note:
        This class implements the Appendix J / Algorithm 2 RKHS closed form.
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

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )

        Iw = np.eye(n) if W is None else np.diag(np.asarray(W).reshape(-1))

        delta = self._get_delta(n)
        alpha = delta**4

        Ka = self._get_kernel(A)
        Kb = self._get_kernel(B)
        Kc = self._get_kernel(C)
        Kd = self._get_kernel(D)

        Pc, Pd = self._build_projectors(
            Kc, Kd, n_scale=n, ridge=False, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )

        self.a, self.b = self._solve_coefficients(
            Ka, Kb, Pc, Pd, Iw, Y, alpha=alpha, l2_variant=True
        )
        self.A = A.copy()
        self.B = B.copy()
        return self

    def predict(self, B_test, *args):
        if len(args) == 0:
            return self._get_kernel(B_test, Y=self.B) @ self.b
        if len(args) == 1:
            A_test = args[0]
            return (self._get_kernel(B_test, Y=self.B) @ self.b,
                    self._get_kernel(A_test, Y=self.A) @ self.a)
        raise ValueError("predict expects at most two arguments, B_test and optionally A_test")


class RKHS2IVL2CV(RKHS2IVL2):
    """
    Cross-validated RKHS2IVL2 estimator.

    Note:
        This class cross-validates the Appendix J / Algorithm 2 RKHS closed form.
    """

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6,
                 expand_alpha_grid=True):
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
        self.expand_alpha_grid = expand_alpha_grid

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )

        Iw = np.eye(n) if W is None else np.diag(np.asarray(W).reshape(-1))

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta = self._get_delta(n)

        gamma_candidates = self._as_candidate_values(self.gamma, "gamma", allow_auto=True, positive=True)
        candidate_summaries = []
        candidate_states = []

        for gamma_candidate in gamma_candidates:
            self.gamma = gamma_candidate
            Ka = self._get_kernel(A)
            Kb = self._get_kernel(B)
            Kc = self._get_kernel(C)
            Kd = self._get_kernel(D)

            cv_result = self._run_alpha_cv_with_optional_expansion(
                lambda alpha_grid: self._run_exact_cv(
                    Ka, Kb, Kc, Kd, Iw, Y, alpha_grid,
                    n_train=n_train, n_test=n_test, delta_train=delta_train,
                    subsetted=subsetted, ind1=ind1, ind2=ind2,
                    ridge=False, l2_variant=True,
                ),
                alpha_scales,
            )
            cv_result["best_alpha"] = cv_result["best_alpha_scale"] * (delta**4)
            candidate_summaries.append({
                "gamma": gamma_candidate,
                "best_alpha_scale": cv_result["best_alpha_scale"],
                "best_score": cv_result["best_score"],
                "n_valid_folds": cv_result["n_valid_folds"],
                "alpha_grid_expanded": cv_result["alpha_grid_expanded"],
                "best_alpha_is_boundary": cv_result["best_alpha_is_boundary"],
            })
            candidate_states.append((gamma_candidate, Ka, Kb, Kc, Kd, cv_result))

        best_candidate_idx = int(np.argmin([state[-1]["best_score"] for state in candidate_states]))
        best_gamma, Ka, Kb, Kc, Kd, cv_result = candidate_states[best_candidate_idx]
        self.gamma = best_gamma
        self.best_gamma_ = best_gamma
        self.cv_gamma_grid_ = list(gamma_candidates)
        self.cv_candidate_summaries_ = candidate_summaries

        self.alpha_scales = cv_result["alpha_scales_used"]
        self.avg_scores = cv_result["avg_scores"]
        self.best_alpha_scale = cv_result["best_alpha_scale"]
        self.best_alpha = cv_result["best_alpha"]
        self._set_cv_diagnostics(cv_result)

        Pc, Pd = self._build_projectors(
            Kc, Kd, n_scale=n, ridge=False, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )
        self.a, self.b = self._solve_coefficients(
            Ka, Kb, Pc, Pd, Iw, Y, alpha=self.best_alpha, l2_variant=True
        )

        self.A = A.copy()
        self.B = B.copy()
        return self


class ApproxRKHS2IVL2(_BaseRKHS2IV):
    """
    Approximate Appendix J / Algorithm 2 RKHS estimator using finite kernel features.

    This class mirrors ``RKHS2IVL2`` with Nystrom/RFF feature approximations.
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

    def _fit_feature_maps(self, A, B, C, D):
        n = A.shape[0]

        self.featA = self._get_new_approx_instance(n_samples=n)
        self.featB = self._get_new_approx_instance(n_samples=n)
        self.featC = self._get_new_approx_instance(n_samples=n)
        self.featD = self._get_new_approx_instance(n_samples=n)

        FA = self.featA.fit_transform(A)
        FB = self.featB.fit_transform(B)
        FC = self.featC.fit_transform(C)
        FD = self.featD.fit_transform(D)
        return FA, FB, FC, FD

    def _projector_from_features(self, F):
        return F @ np.linalg.pinv(F)

    def _lifted_subset_projector_from_features(self, F, subset_indices, scale_n):
        subset_indices = np.asarray(subset_indices, dtype=int)
        if subset_indices.size == 0:
            raise ValueError("Subset projector requested with zero selected rows.")

        n = F.shape[0]
        I_subset = np.eye(n)[subset_indices, :]
        F_subset = F[subset_indices, :]
        P_subset = self._projector_from_features(F_subset)
        return (scale_n / subset_indices.size) * I_subset.T @ P_subset @ I_subset

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )

        Iw = np.eye(n) if W is None else np.diag(np.asarray(W).reshape(-1))

        delta = self._get_delta(n)
        # Align `auto` with exact RKHS2IVL2 default (alpha = delta^4).
        # Keep manual alpha_scale support for explicit finite-sample tuning.
        alpha = delta**4 if _check_auto(self.alpha_scale) else self._get_alpha(delta, self.alpha_scale)

        FA, FB, FC, FD = self._fit_feature_maps(A, B, C, D)

        Ka = FA @ FA.T
        Kb = FB @ FB.T

        if subsetted:
            Pc = self._lifted_subset_projector_from_features(FC, ind2, scale_n=n)
            Pd = self._lifted_subset_projector_from_features(FD, ind1, scale_n=n)
        else:
            Pc = self._projector_from_features(FC)
            Pd = self._projector_from_features(FD)

        self.a, self.b = self._solve_coefficients(
            Ka, Kb, Pc, Pd, Iw, Y, alpha=alpha, l2_variant=True
        )

        # Cache feature-space prediction coefficients.
        self.FA = FA
        self.FB = FB
        self.theta_a = FA.T @ self.a
        self.theta_b = FB.T @ self.b

        self.A = A.copy()
        self.B = B.copy()
        return self

    def predict(self, B_test, A_test=None):
        pred_b = self.featB.transform(B_test) @ self.theta_b
        if A_test is None:
            return pred_b
        pred_a = self.featA.transform(A_test) @ self.theta_a
        return pred_b, pred_a


class ApproxRKHS2IVL2CV(ApproxRKHS2IVL2):
    """
    Cross-validated approximate Appendix J / Algorithm 2 RKHS estimator.
    """

    def __init__(self, kernel_approx='nystrom', n_components=10,
                 kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6,
                 expand_alpha_grid=True):
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
        self.expand_alpha_grid = expand_alpha_grid

    def _run_feature_cv(self, FA, FB, FC, FD, Iw, Y, alpha_scales,
                        n_train, n_test, delta_train, subsetted, ind1, ind2):
        alpha_scales = self._normalize_positive_grid(alpha_scales, "alpha_scales")
        fold_scores_all = []

        for train, test in KFold(n_splits=self.cv).split(Y):
            FA_train, FB_train = FA[train], FB[train]
            FC_train, FD_train = FC[train], FD[train]
            FA_test, FB_test = FA[test], FB[test]
            FC_test, FD_test = FC[test], FD[test]

            Ka_train = FA_train @ FA_train.T
            Kb_train = FB_train @ FB_train.T

            if subsetted:
                train_ind1 = self._local_subset_indices(train, ind1)
                train_ind2 = self._local_subset_indices(train, ind2)
                test_ind1 = self._local_subset_indices(test, ind1)
                test_ind2 = self._local_subset_indices(test, ind2)

                if (train_ind1.size == 0 or train_ind2.size == 0
                        or test_ind1.size == 0 or test_ind2.size == 0):
                    continue

                Pc_train = self._lifted_subset_projector_from_features(
                    FC_train, train_ind2, scale_n=n_train
                )
                Pd_train = self._lifted_subset_projector_from_features(
                    FD_train, train_ind1, scale_n=n_train
                )
                Pc_test = self._lifted_subset_projector_from_features(
                    FC_test, test_ind2, scale_n=n_test
                )
                Pd_test = self._lifted_subset_projector_from_features(
                    FD_test, test_ind1, scale_n=n_test
                )
            else:
                Pc_train = self._projector_from_features(FC_train)
                Pd_train = self._projector_from_features(FD_train)
                Pc_test = self._projector_from_features(FC_test)
                Pd_test = self._projector_from_features(FD_test)

            Iw_train = Iw[np.ix_(train, train)]
            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = float(alpha_scale) * (delta_train**4)
                a, b = self._solve_coefficients(
                    Ka_train, Kb_train, Pc_train, Pd_train,
                    Iw_train, Y[train], alpha=alpha, l2_variant=True
                )

                Ka_test_train = FA_test @ FA_train.T
                Kb_test_train = FB_test @ FB_train.T
                res1 = Y[test] - Ka_test_train @ a
                res2 = Ka_test_train @ a - Kb_test_train @ b
                fold_scores.append(
                    (res1.T @ Pd_test @ res1)[0, 0] / (res1.shape[0]**2)
                    + (res2.T @ Pc_test @ res2)[0, 0] / (res2.shape[0]**2)
                )
            fold_scores_all.append(fold_scores)

        n_valid_folds = len(fold_scores_all)
        if n_valid_folds == 0:
            raise ValueError(
                "No valid CV folds remain under subset constraints. "
                "Ensure both subsets are represented in each fold or reduce cv."
            )

        fold_scores_arr = np.asarray(fold_scores_all, dtype=float)
        return {
            "fold_scores": fold_scores_arr,
            "avg_scores": np.mean(fold_scores_arr, axis=0),
            "n_valid_folds": n_valid_folds,
        }

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )

        Iw = np.eye(n) if W is None else np.diag(np.asarray(W).reshape(-1))
        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta = self._get_delta(n)
        gamma_candidates = self._as_candidate_values(self.gamma, "gamma", allow_auto=True, positive=True)
        ncomp_candidates = self._as_candidate_values(self.n_components, "n_components", allow_auto=False, positive=True)
        candidate_summaries = []
        candidate_states = []

        for gamma_candidate in gamma_candidates:
            for n_components_candidate in ncomp_candidates:
                self.gamma = gamma_candidate
                self.n_components = n_components_candidate
                FA, FB, FC, FD = self._fit_feature_maps(A, B, C, D)
                cv_result = self._run_alpha_cv_with_optional_expansion(
                    lambda alpha_grid: self._run_feature_cv(
                        FA, FB, FC, FD, Iw, Y, alpha_grid,
                        n_train=n_train, n_test=n_test, delta_train=delta_train,
                        subsetted=subsetted, ind1=ind1, ind2=ind2,
                    ),
                    alpha_scales,
                )
                cv_result["best_alpha"] = cv_result["best_alpha_scale"] * (delta**4)
                candidate_summaries.append({
                    "gamma": gamma_candidate,
                    "n_components": n_components_candidate,
                    "best_alpha_scale": cv_result["best_alpha_scale"],
                    "best_score": cv_result["best_score"],
                    "n_valid_folds": cv_result["n_valid_folds"],
                    "alpha_grid_expanded": cv_result["alpha_grid_expanded"],
                    "best_alpha_is_boundary": cv_result["best_alpha_is_boundary"],
                })
                candidate_states.append((gamma_candidate, n_components_candidate, cv_result))

        best_candidate_idx = int(np.argmin([state[-1]["best_score"] for state in candidate_states]))
        best_gamma, best_n_components, cv_result = candidate_states[best_candidate_idx]
        self.gamma = best_gamma
        self.n_components = best_n_components
        self.best_gamma_ = best_gamma
        self.best_n_components_ = best_n_components
        self.cv_gamma_grid_ = list(gamma_candidates)
        self.cv_n_components_grid_ = list(ncomp_candidates)
        self.cv_candidate_summaries_ = candidate_summaries

        self.alpha_scales = cv_result["alpha_scales_used"]
        self.avg_scores = cv_result["avg_scores"]
        self.best_alpha_scale = cv_result["best_alpha_scale"]
        self.best_alpha = cv_result["best_alpha"]
        self._set_cv_diagnostics(cv_result)

        # Rebuild selected features deterministically for final refit.
        FA, FB, FC, FD = self._fit_feature_maps(A, B, C, D)
        Ka = FA @ FA.T
        Kb = FB @ FB.T
        if subsetted:
            Pc = self._lifted_subset_projector_from_features(FC, ind2, scale_n=n)
            Pd = self._lifted_subset_projector_from_features(FD, ind1, scale_n=n)
        else:
            Pc = self._projector_from_features(FC)
            Pd = self._projector_from_features(FD)

        self.a, self.b = self._solve_coefficients(
            Ka, Kb, Pc, Pd, Iw, Y, alpha=self.best_alpha, l2_variant=True
        )
        self.FA = FA
        self.FB = FB
        self.theta_a = FA.T @ self.a
        self.theta_b = FB.T @ self.b
        self.A = A.copy()
        self.B = B.copy()
        return self


class ApproxRKHS2IV(ApproxRKHS2IVL2):
    """
    Approximate alternate simultaneous RKHS estimator using finite kernel features.

    This class mirrors ``RKHS2IV`` (non-Appendix J alternate objective) with
    Nystrom/RFF feature approximations.
    """

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )

        Iw = np.eye(n) if W is None else np.diag(np.asarray(W).reshape(-1))

        delta = self._get_delta(n)
        alpha = delta**4

        FA, FB, FC, FD = self._fit_feature_maps(A, B, C, D)

        Ka = FA @ FA.T
        Kb = FB @ FB.T
        Kc = FC @ FC.T
        Kd = FD @ FD.T

        Pc, Pd = self._build_projectors(
            Kc, Kd, n_scale=n, ridge=True, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )

        self.a, self.b = self._solve_coefficients(
            Ka, Kb, Pc, Pd, Iw, Y, alpha=alpha, l2_variant=False
        )

        self.FA = FA
        self.FB = FB
        self.theta_a = FA.T @ self.a
        self.theta_b = FB.T @ self.b

        self.A = A.copy()
        self.B = B.copy()
        return self


class ApproxRKHS2IVCV(ApproxRKHS2IV):
    """
    Cross-validated approximate alternate simultaneous RKHS estimator.
    """

    def __init__(self, kernel_approx='nystrom', n_components=10,
                 kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6,
                 expand_alpha_grid=True):
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
        self.expand_alpha_grid = expand_alpha_grid

    def _run_feature_cv(self, FA, FB, FC, FD, Iw, Y, alpha_scales,
                        n_train, n_test, delta_train, subsetted, ind1, ind2):
        alpha_scales = self._normalize_positive_grid(alpha_scales, "alpha_scales")
        fold_scores_all = []

        for train, test in KFold(n_splits=self.cv).split(Y):
            FA_train, FB_train = FA[train], FB[train]
            FC_train, FD_train = FC[train], FD[train]
            FA_test, FB_test = FA[test], FB[test]
            FC_test, FD_test = FC[test], FD[test]

            Ka_train = FA_train @ FA_train.T
            Kb_train = FB_train @ FB_train.T
            Kc_train = FC_train @ FC_train.T
            Kd_train = FD_train @ FD_train.T
            Kc_test = FC_test @ FC_test.T
            Kd_test = FD_test @ FD_test.T

            if subsetted:
                train_ind1 = self._local_subset_indices(train, ind1)
                train_ind2 = self._local_subset_indices(train, ind2)
                test_ind1 = self._local_subset_indices(test, ind1)
                test_ind2 = self._local_subset_indices(test, ind2)

                if (train_ind1.size == 0 or train_ind2.size == 0
                        or test_ind1.size == 0 or test_ind2.size == 0):
                    continue

                Pc_train, Pd_train = self._build_projectors(
                    Kc_train, Kd_train, n_scale=n_train, ridge=True, subsetted=True,
                    ind1_local=train_ind1, ind2_local=train_ind2
                )
                Pc_test, Pd_test = self._build_projectors(
                    Kc_test, Kd_test, n_scale=n_test, ridge=True, subsetted=True,
                    ind1_local=test_ind1, ind2_local=test_ind2
                )
            else:
                Pc_train, Pd_train = self._build_projectors(
                    Kc_train, Kd_train, n_scale=n_train, ridge=True, subsetted=False
                )
                Pc_test, Pd_test = self._build_projectors(
                    Kc_test, Kd_test, n_scale=n_test, ridge=True, subsetted=False
                )

            Iw_train = Iw[np.ix_(train, train)]
            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = float(alpha_scale) * (delta_train**4)
                a, b = self._solve_coefficients(
                    Ka_train, Kb_train, Pc_train, Pd_train,
                    Iw_train, Y[train], alpha=alpha, l2_variant=False
                )

                Ka_test_train = FA_test @ FA_train.T
                Kb_test_train = FB_test @ FB_train.T
                res1 = Y[test] - Ka_test_train @ a
                res2 = Ka_test_train @ a - Kb_test_train @ b
                fold_scores.append(
                    _to_scalar(res1.T @ Pd_test @ res1) / (res1.shape[0]**2)
                    + _to_scalar(res2.T @ Pc_test @ res2) / (res2.shape[0]**2)
                )
            fold_scores_all.append(fold_scores)

        n_valid_folds = len(fold_scores_all)
        if n_valid_folds == 0:
            raise ValueError(
                "No valid CV folds remain under subset constraints. "
                "Ensure both subsets are represented in each fold or reduce cv."
            )

        fold_scores_arr = np.asarray(fold_scores_all, dtype=float)
        return {
            "fold_scores": fold_scores_arr,
            "avg_scores": np.mean(fold_scores_arr, axis=0),
            "n_valid_folds": n_valid_folds,
        }

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        Y = _to_column_vector(Y)
        n = Y.shape[0]
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )

        Iw = np.eye(n) if W is None else np.diag(np.asarray(W).reshape(-1))
        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta = self._get_delta(n)

        gamma_candidates = self._as_candidate_values(self.gamma, "gamma", allow_auto=True, positive=True)
        ncomp_candidates = self._as_candidate_values(self.n_components, "n_components", allow_auto=False, positive=True)
        candidate_summaries = []
        candidate_states = []

        for gamma_candidate in gamma_candidates:
            for n_components_candidate in ncomp_candidates:
                self.gamma = gamma_candidate
                self.n_components = n_components_candidate
                FA, FB, FC, FD = self._fit_feature_maps(A, B, C, D)
                cv_result = self._run_alpha_cv_with_optional_expansion(
                    lambda alpha_grid: self._run_feature_cv(
                        FA, FB, FC, FD, Iw, Y, alpha_grid,
                        n_train=n_train, n_test=n_test, delta_train=delta_train,
                        subsetted=subsetted, ind1=ind1, ind2=ind2,
                    ),
                    alpha_scales,
                )
                cv_result["best_alpha"] = cv_result["best_alpha_scale"] * (delta**4)
                candidate_summaries.append({
                    "gamma": gamma_candidate,
                    "n_components": n_components_candidate,
                    "best_alpha_scale": cv_result["best_alpha_scale"],
                    "best_score": cv_result["best_score"],
                    "n_valid_folds": cv_result["n_valid_folds"],
                    "alpha_grid_expanded": cv_result["alpha_grid_expanded"],
                    "best_alpha_is_boundary": cv_result["best_alpha_is_boundary"],
                })
                candidate_states.append((gamma_candidate, n_components_candidate, cv_result))

        best_candidate_idx = int(np.argmin([state[-1]["best_score"] for state in candidate_states]))
        best_gamma, best_n_components, cv_result = candidate_states[best_candidate_idx]
        self.gamma = best_gamma
        self.n_components = best_n_components
        self.best_gamma_ = best_gamma
        self.best_n_components_ = best_n_components
        self.cv_gamma_grid_ = list(gamma_candidates)
        self.cv_n_components_grid_ = list(ncomp_candidates)
        self.cv_candidate_summaries_ = candidate_summaries

        self.alpha_scales = cv_result["alpha_scales_used"]
        self.avg_scores = cv_result["avg_scores"]
        self.best_alpha_scale = cv_result["best_alpha_scale"]
        self.best_alpha = cv_result["best_alpha"]
        self._set_cv_diagnostics(cv_result)

        FA, FB, FC, FD = self._fit_feature_maps(A, B, C, D)
        Ka = FA @ FA.T
        Kb = FB @ FB.T
        Kc = FC @ FC.T
        Kd = FD @ FD.T
        Pc, Pd = self._build_projectors(
            Kc, Kd, n_scale=n, ridge=True, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )

        self.a, self.b = self._solve_coefficients(
            Ka, Kb, Pc, Pd, Iw, Y, alpha=self.best_alpha, l2_variant=False
        )
        self.FA = FA
        self.FB = FB
        self.theta_a = FA.T @ self.a
        self.theta_b = FB.T @ self.b
        self.A = A.copy()
        self.B = B.copy()
        return self
