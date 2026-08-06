"""
This module provides implementations of nested NPIV estimators for RKHS function classes.

Classes:
    _BaseRKHS2IV: Base class for nested RKHS IV methods.
    RKHS2IV: Nested RKHS IV estimator with RKHS-norm regularization.
    RKHS2IVCV: Cross-validated RKHS2IV estimator.
    RKHS2IVL2: Nested RKHS IV estimator aligned with the common-penalty
        specialization of Appendix L.1 / Algorithm 2.
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
from scipy.sparse import issparse
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
    raise ValueError("Y must be a 1D array or a 2D column vector.")


def _as_feature_input(values):
    return values if issparse(values) else np.asarray(values)


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

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Kernel coefficient passed to scikit-learn; for RBF,
            the kernel is ``exp(-gamma * ||x - x'||^2)``.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scale (str or float): Scale of the regularization parameter.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        kernel_params (dict): Additional parameters for the kernel.
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int or float): Number of approximation components.
            Values in (0, 1] are sample fractions with a floor of 10 and are
            then capped at ``n_samples``; integer-like values greater than 1
            are fixed component counts.
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

    def _resolve_fitted_gamma(self, X, kernel_role):
        """Resolve one kernel coefficient from fitting data."""
        return self._resolve_gamma_value(X, kernel_role, self.gamma)

    def _resolve_gamma_value(self, X, kernel_role, gamma):
        """Resolve a specified kernel coefficient from fitting data."""
        if callable(self.kernel):
            return None
        if not _check_auto(gamma):
            return gamma

        pairwise_dists = euclidean_distances(X, X)
        median_dist = float(np.median(pairwise_dists))
        if not np.isfinite(median_dist) or median_dist <= 0:
            raise ValueError(
                f"Cannot resolve `gamma='auto'` for the {kernel_role} kernel: "
                "the median pairwise distance in the fitting data must be "
                "finite and strictly positive."
            )
        return 1.0 / (2 * median_dist)

    def _get_kernel_at_fitted_gamma(self, X, Y=None, fitted_gamma=None):
        """Evaluate a kernel using the coefficient fixed during ``fit``."""
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

    def _resolve_n_components(self, n_samples=None, n_components=None):
        configured_components = (
            self.n_components if n_components is None else n_components
        )
        try:
            value = float(configured_components)
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

    def _get_new_approx_instance(
        self, n_samples=None, fitted_gamma=None, n_components=None
    ):
        gamma = self.gamma if fitted_gamma is None else fitted_gamma
        resolved_components = self._resolve_n_components(
            n_samples=n_samples, n_components=n_components
        )
        if (self.kernel_approx == 'rbfsampler') and (self.kernel == 'rbf'):
            return RBFSampler(
                gamma=gamma,
                n_components=resolved_components,
                random_state=1,
            )
        if self.kernel_approx == 'nystrom':
            return Nystroem(
                kernel=self.kernel,
                gamma=gamma,
                coef0=self.coef0,
                degree=self.degree,
                kernel_params=self.kernel_params,
                random_state=1,
                n_components=resolved_components,
            )
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
        K = np.asarray(K, dtype=float)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("Kernel matrix must be square.")
        if not np.all(np.isfinite(K)):
            raise ValueError("Kernel matrix contains non-finite values.")

        # Numerical symmetry guard: pairwise kernels can carry tiny asymmetry.
        K = 0.5 * (K + K.T)
        n = K.shape[0]

        if ridge:
            A = K + np.eye(n)
            try:
                return np.linalg.solve(A, K)
            except np.linalg.LinAlgError:
                pass

            # Jitter fallback for rare SVD/solve failures on ill-conditioned folds.
            scale = max(1.0, float(np.trace(A)) / max(n, 1))
            for eps in (1e-12, 1e-10, 1e-8, 1e-6):
                try:
                    return np.linalg.solve(A + (eps * scale) * np.eye(n), K)
                except np.linalg.LinAlgError:
                    continue
            return np.linalg.pinv(A, hermitian=True) @ K

        try:
            return np.linalg.pinv(K, hermitian=True) @ K
        except np.linalg.LinAlgError:
            scale = max(1.0, float(np.trace(K)) / max(n, 1))
            for eps in (1e-12, 1e-10, 1e-8, 1e-6):
                try:
                    Kj = K + (eps * scale) * np.eye(n)
                    return np.linalg.pinv(Kj, hermitian=True) @ K
                except np.linalg.LinAlgError:
                    continue
            raise

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

    @staticmethod
    def _as_multiplier_vector(W, n):
        if W is None:
            return np.ones(n, dtype=float)

        multipliers = np.asarray(W, dtype=float).reshape(-1)
        if multipliers.shape[0] != n:
            raise ValueError("W must contain one multiplier per observation.")
        if not np.all(np.isfinite(multipliers)):
            raise ValueError("W must contain only finite multipliers.")
        return multipliers

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
        retained = eigenvalues > _DEFAULT_PINV_RCOND * scale
        eigenvalues = eigenvalues[retained]
        eigenvectors = eigenvectors[:, retained]
        return eigenvectors, eigenvalues

    @staticmethod
    def _rkhs_feature_coordinates(K):
        """Return retained empirical RKHS features and their dual map."""
        eigenvectors, eigenvalues = _BaseRKHS2IV._symmetric_range_basis(K)
        square_roots = np.sqrt(eigenvalues)
        features = eigenvectors * square_roots
        dual_map = eigenvectors / square_roots
        return features, dual_map

    @classmethod
    def _lifted_range_projector(
        cls, K_block, subset_local_indices, scale_n
    ):
        subset_local_indices = np.asarray(subset_local_indices, dtype=int)
        if subset_local_indices.size == 0:
            raise ValueError("Subset projector requested with zero selected rows.")

        K_subset = K_block[np.ix_(
            subset_local_indices, subset_local_indices
        )]
        basis, _ = cls._symmetric_range_basis(K_subset)
        projector_subset = basis @ basis.T
        projector = np.zeros_like(K_block, dtype=float)
        projector[np.ix_(
            subset_local_indices, subset_local_indices
        )] = (
            scale_n / subset_local_indices.size
        ) * projector_subset
        return projector

    @classmethod
    def _build_range_projectors(
        cls, Kc, Kd, n_scale, subsetted=False,
        ind1_local=None, ind2_local=None
    ):
        if not subsetted:
            basis_c, _ = cls._symmetric_range_basis(Kc)
            basis_d, _ = cls._symmetric_range_basis(Kd)
            return basis_c @ basis_c.T, basis_d @ basis_d.T

        Pc = cls._lifted_range_projector(
            Kc, ind2_local, scale_n=n_scale
        )
        Pd = cls._lifted_range_projector(
            Kd, ind1_local, scale_n=n_scale
        )
        return Pc, Pd

    @staticmethod
    def _solve_rkhs_norm_joint(Ka, Kb, Pc, Pd, multipliers, Y, alpha):
        """Solve the two bridge first-order conditions as one block system."""
        Pc = 0.5 * (Pc + Pc.T)
        Pd = 0.5 * (Pd + Pd.T)
        features_a, dual_map_a = _BaseRKHS2IV._rkhs_feature_coordinates(Ka)
        features_b, dual_map_b = _BaseRKHS2IV._rkhs_feature_coordinates(Kb)
        weighted_features_a = multipliers[:, None] * features_a
        rank_a = features_a.shape[1]
        rank_b = features_b.shape[1]

        system_aa = (
            features_a.T @ Pd @ features_a
            + weighted_features_a.T @ Pc @ weighted_features_a
            + alpha * np.eye(rank_a)
        )
        system_ab = -weighted_features_a.T @ Pc @ features_b
        system_bb = (
            features_b.T @ Pc @ features_b
            + alpha * np.eye(rank_b)
        )
        system = np.block([
            [system_aa, system_ab],
            [system_ab.T, system_bb],
        ])
        rhs = np.vstack([
            features_a.T @ Pd @ Y,
            np.zeros((rank_b, Y.shape[1]), dtype=float),
        ])

        # NumPy's default pseudoinverse tolerance defines the numerical rank.
        coefficients = np.linalg.pinv(system) @ rhs
        coefficients_a = dual_map_a @ coefficients[:rank_a]
        coefficients_b = dual_map_b @ coefficients[rank_a:]
        return coefficients_a, coefficients_b

    @staticmethod
    def _solve_l2_joint(Ka, Kb, Pc, Pd, multipliers, Y, alpha):
        """Solve the empirical-L2 first-order conditions jointly."""
        Pc = 0.5 * (Pc + Pc.T)
        Pd = 0.5 * (Pd + Pd.T)
        weighted_Ka = multipliers[:, None] * Ka

        system_aa = (
            Ka @ Pd @ Ka
            + weighted_Ka.T @ Pc @ weighted_Ka
            + alpha * (Ka @ Ka)
        )
        system_ab = -weighted_Ka.T @ Pc @ Kb
        system_bb = Kb @ Pc @ Kb + alpha * (Kb @ Kb)
        system = np.block([
            [system_aa, system_ab],
            [system_ab.T, system_bb],
        ])
        rhs = np.vstack([
            Ka @ Pd @ Y,
            np.zeros((Y.shape[0], Y.shape[1]), dtype=float),
        ])

        # NumPy's default pseudoinverse tolerance defines the numerical rank.
        coefficients = np.linalg.pinv(system) @ rhs
        n = Y.shape[0]
        return coefficients[:n], coefficients[n:]

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
        self.alpha_scales_ = self.cv_alpha_scales_used_.copy()
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

    def _run_rkhs_norm_cv(
        self, A, B, C, D, Y, multipliers, alpha_scales,
        gamma_candidate, subsetted, ind1, ind2
    ):
        alpha_scales = self._normalize_positive_grid(
            alpha_scales, "alpha_scales"
        )
        fold_specific_gamma = (
            not callable(self.kernel) and _check_auto(gamma_candidate)
        )

        if not fold_specific_gamma:
            gamma_a = self._resolve_gamma_value(
                A, "A", gamma_candidate
            )
            gamma_b = self._resolve_gamma_value(
                B, "B", gamma_candidate
            )
            gamma_c = self._resolve_gamma_value(
                C, "C", gamma_candidate
            )
            gamma_d = self._resolve_gamma_value(
                D, "D", gamma_candidate
            )
            Ka = self._get_kernel_at_fitted_gamma(
                A, fitted_gamma=gamma_a
            )
            Kb = self._get_kernel_at_fitted_gamma(
                B, fitted_gamma=gamma_b
            )
            Kc = self._get_kernel_at_fitted_gamma(
                C, fitted_gamma=gamma_c
            )
            Kd = self._get_kernel_at_fitted_gamma(
                D, fitted_gamma=gamma_d
            )

        fold_scores_all = []
        for train, test in KFold(n_splits=self.cv).split(Y):
            n_train = len(train)
            n_test = len(test)
            delta_train = self._get_delta(n_train)

            if fold_specific_gamma:
                gamma_a_train = self._resolve_gamma_value(
                    A[train], "A", gamma_candidate
                )
                gamma_b_train = self._resolve_gamma_value(
                    B[train], "B", gamma_candidate
                )
                gamma_c_train = self._resolve_gamma_value(
                    C[train], "C", gamma_candidate
                )
                gamma_d_train = self._resolve_gamma_value(
                    D[train], "D", gamma_candidate
                )
                Ka_train = self._get_kernel_at_fitted_gamma(
                    A[train], fitted_gamma=gamma_a_train
                )
                Kb_train = self._get_kernel_at_fitted_gamma(
                    B[train], fitted_gamma=gamma_b_train
                )
                Kc_train = self._get_kernel_at_fitted_gamma(
                    C[train], fitted_gamma=gamma_c_train
                )
                Kd_train = self._get_kernel_at_fitted_gamma(
                    D[train], fitted_gamma=gamma_d_train
                )
                Ka_test_train = self._get_kernel_at_fitted_gamma(
                    A[test], Y=A[train], fitted_gamma=gamma_a_train
                )
                Kb_test_train = self._get_kernel_at_fitted_gamma(
                    B[test], Y=B[train], fitted_gamma=gamma_b_train
                )
                Kc_test = self._get_kernel_at_fitted_gamma(
                    C[test], fitted_gamma=gamma_c_train
                )
                Kd_test = self._get_kernel_at_fitted_gamma(
                    D[test], fitted_gamma=gamma_d_train
                )
            else:
                Ka_train = Ka[np.ix_(train, train)]
                Kb_train = Kb[np.ix_(train, train)]
                Kc_train = Kc[np.ix_(train, train)]
                Kd_train = Kd[np.ix_(train, train)]
                Ka_test_train = Ka[np.ix_(test, train)]
                Kb_test_train = Kb[np.ix_(test, train)]
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
                    Kc_train, Kd_train, n_scale=n_train, ridge=True,
                    subsetted=True, ind1_local=train_ind1,
                    ind2_local=train_ind2
                )
                Pc_test, Pd_test = self._build_projectors(
                    Kc_test, Kd_test, n_scale=n_test, ridge=True,
                    subsetted=True, ind1_local=test_ind1,
                    ind2_local=test_ind2
                )
            else:
                Pc_train, Pd_train = self._build_projectors(
                    Kc_train, Kd_train, n_scale=n_train, ridge=True
                )
                Pc_test, Pd_test = self._build_projectors(
                    Kc_test, Kd_test, n_scale=n_test, ridge=True
                )

            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = float(alpha_scale) * (delta_train ** 4)
                a, b = self._solve_rkhs_norm_joint(
                    Ka_train, Kb_train, Pc_train, Pd_train,
                    multipliers[train], Y[train], alpha
                )
                prediction_a = Ka_test_train @ a
                prediction_b = Kb_test_train @ b
                residual_y = Y[test] - prediction_a
                residual_bridge = (
                    prediction_b
                    - multipliers[test, None] * prediction_a
                )
                fold_scores.append(
                    _to_scalar(residual_y.T @ Pd_test @ residual_y)
                    / (n_test ** 2)
                    + _to_scalar(
                        residual_bridge.T @ Pc_test @ residual_bridge
                    ) / (n_test ** 2)
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

    def _run_l2_joint_cv(
        self, A, B, C, D, Y, multipliers, alpha_scales,
        gamma_candidate, subsetted, ind1, ind2
    ):
        alpha_scales = self._normalize_positive_grid(
            alpha_scales, "alpha_scales"
        )
        fold_specific_gamma = (
            not callable(self.kernel) and _check_auto(gamma_candidate)
        )

        if not fold_specific_gamma:
            gamma_a = self._resolve_gamma_value(
                A, "A", gamma_candidate
            )
            gamma_b = self._resolve_gamma_value(
                B, "B", gamma_candidate
            )
            gamma_c = self._resolve_gamma_value(
                C, "C", gamma_candidate
            )
            gamma_d = self._resolve_gamma_value(
                D, "D", gamma_candidate
            )
            Ka = self._get_kernel_at_fitted_gamma(
                A, fitted_gamma=gamma_a
            )
            Kb = self._get_kernel_at_fitted_gamma(
                B, fitted_gamma=gamma_b
            )
            Kc = self._get_kernel_at_fitted_gamma(
                C, fitted_gamma=gamma_c
            )
            Kd = self._get_kernel_at_fitted_gamma(
                D, fitted_gamma=gamma_d
            )

        fold_scores_all = []
        for train, test in KFold(n_splits=self.cv).split(Y):
            n_train = len(train)
            n_test = len(test)
            delta_train = self._get_delta(n_train)

            if fold_specific_gamma:
                gamma_a_train = self._resolve_gamma_value(
                    A[train], "A", gamma_candidate
                )
                gamma_b_train = self._resolve_gamma_value(
                    B[train], "B", gamma_candidate
                )
                gamma_c_train = self._resolve_gamma_value(
                    C[train], "C", gamma_candidate
                )
                gamma_d_train = self._resolve_gamma_value(
                    D[train], "D", gamma_candidate
                )
                Ka_train = self._get_kernel_at_fitted_gamma(
                    A[train], fitted_gamma=gamma_a_train
                )
                Kb_train = self._get_kernel_at_fitted_gamma(
                    B[train], fitted_gamma=gamma_b_train
                )
                Kc_train = self._get_kernel_at_fitted_gamma(
                    C[train], fitted_gamma=gamma_c_train
                )
                Kd_train = self._get_kernel_at_fitted_gamma(
                    D[train], fitted_gamma=gamma_d_train
                )
                Ka_test_train = self._get_kernel_at_fitted_gamma(
                    A[test], Y=A[train], fitted_gamma=gamma_a_train
                )
                Kb_test_train = self._get_kernel_at_fitted_gamma(
                    B[test], Y=B[train], fitted_gamma=gamma_b_train
                )
                Kc_test = self._get_kernel_at_fitted_gamma(
                    C[test], fitted_gamma=gamma_c_train
                )
                Kd_test = self._get_kernel_at_fitted_gamma(
                    D[test], fitted_gamma=gamma_d_train
                )
            else:
                Ka_train = Ka[np.ix_(train, train)]
                Kb_train = Kb[np.ix_(train, train)]
                Kc_train = Kc[np.ix_(train, train)]
                Kd_train = Kd[np.ix_(train, train)]
                Ka_test_train = Ka[np.ix_(test, train)]
                Kb_test_train = Kb[np.ix_(test, train)]
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
                Pc_train, Pd_train = self._build_range_projectors(
                    Kc_train, Kd_train, n_scale=n_train,
                    subsetted=True, ind1_local=train_ind1,
                    ind2_local=train_ind2
                )
                Pc_test, Pd_test = self._build_range_projectors(
                    Kc_test, Kd_test, n_scale=n_test,
                    subsetted=True, ind1_local=test_ind1,
                    ind2_local=test_ind2
                )
            else:
                Pc_train, Pd_train = self._build_range_projectors(
                    Kc_train, Kd_train, n_scale=n_train
                )
                Pc_test, Pd_test = self._build_range_projectors(
                    Kc_test, Kd_test, n_scale=n_test
                )

            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = float(alpha_scale) * (delta_train ** 4)
                a, b = self._solve_l2_joint(
                    Ka_train, Kb_train, Pc_train, Pd_train,
                    multipliers[train], Y[train], alpha
                )
                prediction_a = Ka_test_train @ a
                prediction_b = Kb_test_train @ b
                residual_y = Y[test] - prediction_a
                residual_bridge = (
                    prediction_b
                    - multipliers[test, None] * prediction_a
                )
                fold_scores.append(
                    _to_scalar(residual_y.T @ Pd_test @ residual_y)
                    / (n_test ** 2)
                    + _to_scalar(
                        residual_bridge.T @ Pc_test @ residual_bridge
                    ) / (n_test ** 2)
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

    This class jointly solves the two bridge equations with RKHS-norm learner
    penalties and unit-ridge instrument actions.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Kernel coefficient passed to scikit-learn; for RBF,
            the kernel is ``exp(-gamma * ||x - x'||^2)``.
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

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        """
        Fit the nested RKHS IV estimator.

        Parameters:
            A (array-like): First nested-stage treatment or endogenous block.
            B (array-like): Second nested-stage treatment or endogenous block.
            C (array-like): Second nested-stage instrument block.
            D (array-like): First nested-stage instrument block.
            Y (array-like): Outcome values.
            W (array-like or None): Optional observation-level multipliers in
                the bridge residual ``h(B) - W*g(A)``; defaults to one.
            subsetted (bool): Whether to apply stage-specific subset restrictions.
            subset_ind1 (array-like or None): Indicator or mask selecting the
                first-stage ``D``-moment rows.
            subset_ind2 (array-like or None): Indicator or mask selecting the
                second-stage ``C``-moment rows; defaults to the complement of
                ``subset_ind1``.
        """
        Y = _to_column_vector(Y)
        A, B, C, D = map(np.asarray, (A, B, C, D))
        n = Y.shape[0]
        if any(values.ndim == 0 or values.shape[0] != n
               for values in (A, B, C, D)):
            raise ValueError("A, B, C, D, and Y must have the same number of observations.")
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )
        multipliers = self._as_multiplier_vector(W, n)

        delta = self._get_delta(n)
        alpha = delta**4

        self.gamma_a_ = self._resolve_fitted_gamma(A, "A")
        self.gamma_b_ = self._resolve_fitted_gamma(B, "B")
        self.gamma_c_ = self._resolve_fitted_gamma(C, "C")
        self.gamma_d_ = self._resolve_fitted_gamma(D, "D")
        Ka = self._get_kernel_at_fitted_gamma(
            A, fitted_gamma=self.gamma_a_
        )
        Kb = self._get_kernel_at_fitted_gamma(
            B, fitted_gamma=self.gamma_b_
        )
        Kc = self._get_kernel_at_fitted_gamma(
            C, fitted_gamma=self.gamma_c_
        )
        Kd = self._get_kernel_at_fitted_gamma(
            D, fitted_gamma=self.gamma_d_
        )

        Pc, Pd = self._build_projectors(
            Kc, Kd, n_scale=n, ridge=True, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )

        self.a, self.b = self._solve_rkhs_norm_joint(
            Ka, Kb, Pc, Pd, multipliers, Y, alpha
        )
        self.A = A.copy()
        self.B = B.copy()
        return self

    def predict(self, B_test, *args):
        """
        Predict fitted bridge values for test data.

        Parameters:
            B_test (array-like): Test data for the second nested-stage block.
            A_test (array-like, optional): If supplied as the second positional
                argument, test data for the first nested-stage block.

        Returns:
            numpy.ndarray or tuple: ``h_hat(B_test)`` when only ``B_test`` is
            supplied; otherwise ``(h_hat(B_test), g_hat(A_test))``.
        """
        if hasattr(self, "gamma_b_"):
            kernel_b = self._get_kernel_at_fitted_gamma(
                B_test, Y=self.B, fitted_gamma=self.gamma_b_
            )
        else:
            kernel_b = self._get_kernel(B_test, Y=self.B)

        if len(args) == 0:
            return kernel_b @ self.b
        if len(args) == 1:
            A_test = args[0]
            if hasattr(self, "gamma_a_"):
                kernel_a = self._get_kernel_at_fitted_gamma(
                    A_test, Y=self.A, fitted_gamma=self.gamma_a_
                )
            else:
                kernel_a = self._get_kernel(A_test, Y=self.A)
            return (kernel_b @ self.b, kernel_a @ self.a)
        raise ValueError("predict expects at most two arguments, B_test and optionally A_test")


class RKHS2IVCV(RKHS2IV):
    """
    Cross-validated RKHS2IV estimator.

    This class cross-validates the simultaneous RKHS-norm estimator.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str, float, or array-like): Automatic RBF coefficient, fixed
            coefficient, or candidate coefficient grid; the RBF kernel is
            ``exp(-gamma * ||x - x'||^2)``.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        kernel_params (dict): Additional parameters for the kernel.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        n_alphas (int): Number of alpha scales to try.
        cv (int): Number of folds for cross-validation.
        expand_alpha_grid (bool): Whether to expand the alpha grid when the CV optimum lies on a boundary.
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
        """
        Fit the nested RKHS IV estimator.

        Parameters:
            A (array-like): First nested-stage treatment or endogenous block.
            B (array-like): Second nested-stage treatment or endogenous block.
            C (array-like): Second nested-stage instrument block.
            D (array-like): First nested-stage instrument block.
            Y (array-like): Outcome values.
            W (array-like or None): Optional observation-level multipliers in
                the bridge residual ``h(B) - W*g(A)``; defaults to one.
            subsetted (bool): Whether to apply stage-specific subset restrictions.
            subset_ind1 (array-like or None): Indicator or mask selecting the
                first-stage ``D``-moment rows.
            subset_ind2 (array-like or None): Indicator or mask selecting the
                second-stage ``C``-moment rows; defaults to the complement of
                ``subset_ind1``.
        """
        Y = _to_column_vector(Y)
        A, B, C, D = map(np.asarray, (A, B, C, D))
        n = Y.shape[0]
        if any(values.ndim == 0 or values.shape[0] != n
               for values in (A, B, C, D)):
            raise ValueError(
                "A, B, C, D, and Y must have the same number of observations."
            )
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1,
            subset_ind2=subset_ind2
        )
        multipliers = self._as_multiplier_vector(W, n)
        alpha_scales = self._get_alpha_scales()
        delta = self._get_delta(n)
        gamma_candidates = self._as_candidate_values(
            self.gamma, "gamma", allow_auto=True, positive=True
        )
        candidate_summaries = []
        candidate_states = []

        for gamma_candidate in gamma_candidates:
            cv_result = self._run_alpha_cv_with_optional_expansion(
                lambda alpha_grid, candidate=gamma_candidate:
                    self._run_rkhs_norm_cv(
                        A, B, C, D, Y, multipliers, alpha_grid,
                        candidate, subsetted, ind1, ind2
                    ),
                alpha_scales,
            )
            cv_result["best_alpha"] = (
                cv_result["best_alpha_scale"] * (delta ** 4)
            )
            candidate_summaries.append({
                "gamma": gamma_candidate,
                "best_alpha_scale": cv_result["best_alpha_scale"],
                "best_score": cv_result["best_score"],
                "n_valid_folds": cv_result["n_valid_folds"],
                "alpha_grid_expanded": cv_result["alpha_grid_expanded"],
                "best_alpha_is_boundary": cv_result[
                    "best_alpha_is_boundary"
                ],
            })
            candidate_states.append((gamma_candidate, cv_result))

        best_candidate_idx = int(np.argmin([
            state[1]["best_score"] for state in candidate_states
        ]))
        best_gamma, cv_result = candidate_states[best_candidate_idx]
        self.best_gamma_ = best_gamma
        self.cv_gamma_grid_ = list(gamma_candidates)
        self.cv_candidate_summaries_ = candidate_summaries
        self.avg_scores = cv_result["avg_scores"]
        self.best_alpha_scale = cv_result["best_alpha_scale"]
        self.best_alpha = cv_result["best_alpha"]
        self._set_cv_diagnostics(cv_result)

        self.gamma_a_ = self._resolve_gamma_value(
            A, "A", best_gamma
        )
        self.gamma_b_ = self._resolve_gamma_value(
            B, "B", best_gamma
        )
        self.gamma_c_ = self._resolve_gamma_value(
            C, "C", best_gamma
        )
        self.gamma_d_ = self._resolve_gamma_value(
            D, "D", best_gamma
        )
        Ka = self._get_kernel_at_fitted_gamma(
            A, fitted_gamma=self.gamma_a_
        )
        Kb = self._get_kernel_at_fitted_gamma(
            B, fitted_gamma=self.gamma_b_
        )
        Kc = self._get_kernel_at_fitted_gamma(
            C, fitted_gamma=self.gamma_c_
        )
        Kd = self._get_kernel_at_fitted_gamma(
            D, fitted_gamma=self.gamma_d_
        )
        Pc, Pd = self._build_projectors(
            Kc, Kd, n_scale=n, ridge=True, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )
        self.a, self.b = self._solve_rkhs_norm_joint(
            Ka, Kb, Pc, Pd, multipliers, Y, self.best_alpha
        )
        self.A = A.copy()
        self.B = B.copy()
        return self


class RKHS2IVL2(_BaseRKHS2IV):
    """
    Nested RKHS IV estimator with L2 regularization.

    Note:
        This class implements the common-penalty specialization
        ``mu_prime = mu = alpha`` of the Appendix L.1 / Algorithm 2 RKHS
        block solution.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Kernel coefficient passed to scikit-learn; for RBF,
            the kernel is ``exp(-gamma * ||x - x'||^2)``.
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

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        """
        Fit the nested RKHS IV estimator.

        Parameters:
            A (array-like): First nested-stage treatment or endogenous block.
            B (array-like): Second nested-stage treatment or endogenous block.
            C (array-like): Second nested-stage instrument block.
            D (array-like): First nested-stage instrument block.
            Y (array-like): Outcome values.
            W (array-like or None): Optional observation-level multipliers in
                the bridge residual ``h(B) - W*g(A)``; defaults to one.
            subsetted (bool): Whether to apply stage-specific subset restrictions.
            subset_ind1 (array-like or None): Indicator or mask selecting the
                first-stage ``D``-moment rows.
            subset_ind2 (array-like or None): Indicator or mask selecting the
                second-stage ``C``-moment rows; defaults to the complement of
                ``subset_ind1``.
        """
        Y = _to_column_vector(Y)
        A, B, C, D = map(np.asarray, (A, B, C, D))
        n = Y.shape[0]
        if any(values.ndim == 0 or values.shape[0] != n
               for values in (A, B, C, D)):
            raise ValueError(
                "A, B, C, D, and Y must have the same number of observations."
            )
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1, subset_ind2=subset_ind2
        )
        multipliers = self._as_multiplier_vector(W, n)

        delta = self._get_delta(n)
        alpha = delta**4

        self.gamma_a_ = self._resolve_fitted_gamma(A, "A")
        self.gamma_b_ = self._resolve_fitted_gamma(B, "B")
        self.gamma_c_ = self._resolve_fitted_gamma(C, "C")
        self.gamma_d_ = self._resolve_fitted_gamma(D, "D")
        Ka = self._get_kernel_at_fitted_gamma(
            A, fitted_gamma=self.gamma_a_
        )
        Kb = self._get_kernel_at_fitted_gamma(
            B, fitted_gamma=self.gamma_b_
        )
        Kc = self._get_kernel_at_fitted_gamma(
            C, fitted_gamma=self.gamma_c_
        )
        Kd = self._get_kernel_at_fitted_gamma(
            D, fitted_gamma=self.gamma_d_
        )

        Pc, Pd = self._build_range_projectors(
            Kc, Kd, n_scale=n, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )

        self.a, self.b = self._solve_l2_joint(
            Ka, Kb, Pc, Pd, multipliers, Y, alpha
        )
        self.A = A.copy()
        self.B = B.copy()
        return self

    def predict(self, B_test, *args):
        """
        Predict fitted bridge values for test data.

        Parameters:
            B_test (array-like): Test data for the second nested-stage block.
            A_test (array-like, optional): If supplied as the second positional
                argument, test data for the first nested-stage block.

        Returns:
            numpy.ndarray or tuple: ``h_hat(B_test)`` when only ``B_test`` is
            supplied; otherwise ``(h_hat(B_test), g_hat(A_test))``.
        """
        if hasattr(self, "gamma_b_"):
            kernel_b = self._get_kernel_at_fitted_gamma(
                B_test, Y=self.B, fitted_gamma=self.gamma_b_
            )
        else:
            kernel_b = self._get_kernel(B_test, Y=self.B)

        if len(args) == 0:
            return kernel_b @ self.b
        if len(args) == 1:
            A_test = args[0]
            if hasattr(self, "gamma_a_"):
                kernel_a = self._get_kernel_at_fitted_gamma(
                    A_test, Y=self.A, fitted_gamma=self.gamma_a_
                )
            else:
                kernel_a = self._get_kernel(A_test, Y=self.A)
            return (kernel_b @ self.b, kernel_a @ self.a)
        raise ValueError("predict expects at most two arguments, B_test and optionally A_test")


class RKHS2IVL2CV(RKHS2IVL2):
    """
    Cross-validated RKHS2IVL2 estimator.

    Note:
        This class cross-validates the common-penalty specialization
        ``mu_prime = mu = alpha`` of the Appendix L.1 / Algorithm 2 RKHS
        block solution.

    Parameters:
        kernel (str or callable): Kernel function or string identifier.
        gamma (str, float, or array-like): Automatic RBF coefficient, fixed
            coefficient, or candidate coefficient grid; the RBF kernel is
            ``exp(-gamma * ||x - x'||^2)``.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        kernel_params (dict): Additional parameters for the kernel.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        n_alphas (int): Number of alpha scales to try.
        cv (int): Number of folds for cross-validation.
        expand_alpha_grid (bool): Whether to expand the alpha grid when the CV optimum lies on a boundary.
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
        """
        Fit the nested RKHS IV estimator.

        Parameters:
            A (array-like): First nested-stage treatment or endogenous block.
            B (array-like): Second nested-stage treatment or endogenous block.
            C (array-like): Second nested-stage instrument block.
            D (array-like): First nested-stage instrument block.
            Y (array-like): Outcome values.
            W (array-like or None): Optional observation-level multipliers in
                the bridge residual ``h(B) - W*g(A)``; defaults to one.
            subsetted (bool): Whether to apply stage-specific subset restrictions.
            subset_ind1 (array-like or None): Indicator or mask selecting the
                first-stage ``D``-moment rows.
            subset_ind2 (array-like or None): Indicator or mask selecting the
                second-stage ``C``-moment rows; defaults to the complement of
                ``subset_ind1``.
        """
        Y = _to_column_vector(Y)
        A, B, C, D = map(np.asarray, (A, B, C, D))
        n = Y.shape[0]
        if any(values.ndim == 0 or values.shape[0] != n
               for values in (A, B, C, D)):
            raise ValueError(
                "A, B, C, D, and Y must have the same number of observations."
            )
        ind1, ind2 = self._validate_subset_inputs(
            n, subsetted=subsetted, subset_ind1=subset_ind1,
            subset_ind2=subset_ind2
        )
        multipliers = self._as_multiplier_vector(W, n)
        alpha_scales = self._get_alpha_scales()
        delta = self._get_delta(n)
        gamma_candidates = self._as_candidate_values(
            self.gamma, "gamma", allow_auto=True, positive=True
        )
        candidate_summaries = []
        candidate_states = []

        for gamma_candidate in gamma_candidates:
            cv_result = self._run_alpha_cv_with_optional_expansion(
                lambda alpha_grid, candidate=gamma_candidate:
                    self._run_l2_joint_cv(
                        A, B, C, D, Y, multipliers, alpha_grid,
                        candidate, subsetted, ind1, ind2
                    ),
                alpha_scales,
            )
            cv_result["best_alpha"] = (
                cv_result["best_alpha_scale"] * (delta ** 4)
            )
            candidate_summaries.append({
                "gamma": gamma_candidate,
                "best_alpha_scale": cv_result["best_alpha_scale"],
                "best_score": cv_result["best_score"],
                "n_valid_folds": cv_result["n_valid_folds"],
                "alpha_grid_expanded": cv_result["alpha_grid_expanded"],
                "best_alpha_is_boundary": cv_result[
                    "best_alpha_is_boundary"
                ],
            })
            candidate_states.append((gamma_candidate, cv_result))

        best_candidate_idx = int(np.argmin([
            state[1]["best_score"] for state in candidate_states
        ]))
        best_gamma, cv_result = candidate_states[best_candidate_idx]
        self.best_gamma_ = best_gamma
        self.cv_gamma_grid_ = list(gamma_candidates)
        self.cv_candidate_summaries_ = candidate_summaries
        self.avg_scores = cv_result["avg_scores"]
        self.best_alpha_scale = cv_result["best_alpha_scale"]
        self.best_alpha = cv_result["best_alpha"]
        self._set_cv_diagnostics(cv_result)

        self.gamma_a_ = self._resolve_gamma_value(
            A, "A", best_gamma
        )
        self.gamma_b_ = self._resolve_gamma_value(
            B, "B", best_gamma
        )
        self.gamma_c_ = self._resolve_gamma_value(
            C, "C", best_gamma
        )
        self.gamma_d_ = self._resolve_gamma_value(
            D, "D", best_gamma
        )
        Ka = self._get_kernel_at_fitted_gamma(
            A, fitted_gamma=self.gamma_a_
        )
        Kb = self._get_kernel_at_fitted_gamma(
            B, fitted_gamma=self.gamma_b_
        )
        Kc = self._get_kernel_at_fitted_gamma(
            C, fitted_gamma=self.gamma_c_
        )
        Kd = self._get_kernel_at_fitted_gamma(
            D, fitted_gamma=self.gamma_d_
        )
        Pc, Pd = self._build_range_projectors(
            Kc, Kd, n_scale=n, subsetted=subsetted,
            ind1_local=ind1, ind2_local=ind2
        )
        self.a, self.b = self._solve_l2_joint(
            Ka, Kb, Pc, Pd, multipliers, Y, self.best_alpha
        )
        self.A = A.copy()
        self.B = B.copy()
        return self


class ApproxRKHS2IVL2(_BaseRKHS2IV):
    """
    Approximate common-penalty Appendix L.1 / Algorithm 2 RKHS estimator using
    finite kernel features. It uses ``mu_prime = mu = alpha``.

    Instrument projections and the empirical-L2 normal equations are
    contracted directly through the finite feature matrices. The fitted block
    system is therefore feature-sized and does not require sample-sized Gram
    matrices.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int or float): Number of approximation components.
            Values in (0, 1] are sample fractions with a floor of 10 and are
            then capped at ``n_samples``; integer-like values greater than 1
            are fixed component counts.
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Kernel coefficient passed to scikit-learn; for RBF,
            the kernel is ``exp(-gamma * ||x - x'||^2)``.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        kernel_params (dict): Additional parameters for the kernel.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scale (str or float): Scale of the regularization parameter.
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

    @staticmethod
    def _instrument_action(
        features,
        subset_indices=None,
        scale_n=None,
    ):
        if subset_indices is None:
            selected_features = features
            selected_indices = None
            scale = 1.0
        else:
            selected_indices = np.asarray(subset_indices, dtype=int)
            if selected_indices.size == 0:
                raise ValueError(
                    "Instrument range requested with zero selected rows."
                )
            selected_features = features[selected_indices]
            scale = float(scale_n) / selected_indices.size

        feature_pinv = np.linalg.pinv(selected_features)
        return selected_features, feature_pinv, selected_indices, scale

    @staticmethod
    def _instrument_cross_gram(
        action,
        values,
    ):
        features, feature_pinv, subset_indices, scale = action
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if subset_indices is not None:
            values = values[subset_indices]
        cross_gram = scale * (
            (values.T @ features) @ (feature_pinv @ values)
        )
        return 0.5 * (cross_gram + cross_gram.T)

    @classmethod
    def _l2_joint_system_terms(
        cls,
        FA,
        FB,
        FC,
        FD,
        multipliers,
        Y,
        subsetted=False,
        ind1=None,
        ind2=None,
    ):
        weighted_FA = multipliers[:, None] * FA
        rank_a = FA.shape[1]
        rank_b = FB.shape[1]
        n = Y.shape[0]

        action_c = cls._instrument_action(
            FC,
            subset_indices=ind2 if subsetted else None,
            scale_n=n,
        )
        action_d = cls._instrument_action(
            FD,
            subset_indices=ind1 if subsetted else None,
            scale_n=n,
        )
        d_values = np.hstack([FA, Y])
        d_cross = cls._instrument_cross_gram(
            action_d,
            d_values,
        )
        c_values = np.hstack([weighted_FA, FB])
        c_cross = cls._instrument_cross_gram(
            action_c,
            c_values,
        )

        moment_aa = (
            d_cross[:rank_a, :rank_a]
            + c_cross[:rank_a, :rank_a]
        )
        moment_ab = -c_cross[:rank_a, rank_a:]
        moment_bb = c_cross[rank_a:, rank_a:]
        moment_system = np.block([
            [moment_aa, moment_ab],
            [moment_ab.T, moment_bb],
        ])
        penalty_system = np.block([
            [
                FA.T @ FA,
                np.zeros((rank_a, rank_b), dtype=float),
            ],
            [
                np.zeros((rank_b, rank_a), dtype=float),
                FB.T @ FB,
            ],
        ])
        rhs = np.vstack([
            d_cross[:rank_a, rank_a:],
            np.zeros((rank_b, Y.shape[1]), dtype=float),
        ])
        return moment_system, penalty_system, rhs, rank_a

    @staticmethod
    def _solve_l2_joint_feature_system(
        moment_system,
        penalty_system,
        rhs,
        rank_a,
        alpha,
    ):
        system = moment_system + alpha * penalty_system
        coefficients = np.linalg.pinv(system) @ rhs
        return coefficients[:rank_a], coefficients[rank_a:], system

    def _fit_l2_feature_maps(
        self,
        A,
        B,
        C,
        D,
        gamma,
        n_components,
        store=False,
    ):
        n = A.shape[0]
        fitted_gammas = (
            self._resolve_gamma_value(A, "A", gamma),
            self._resolve_gamma_value(B, "B", gamma),
            self._resolve_gamma_value(C, "C", gamma),
            self._resolve_gamma_value(D, "D", gamma),
        )
        feature_maps = tuple(
            self._get_new_approx_instance(
                n_samples=n,
                fitted_gamma=fitted_gamma,
                n_components=n_components,
            )
            for fitted_gamma in fitted_gammas
        )
        features = tuple(
            feature_map.fit_transform(values)
            for feature_map, values in zip(
                feature_maps, (A, B, C, D)
            )
        )

        if store:
            self.featA, self.featB, self.featC, self.featD = feature_maps
            self.FA, self.FB, self.FC, self.FD = features
            (
                self.gamma_a_,
                self.gamma_b_,
                self.gamma_c_,
                self.gamma_d_,
            ) = fitted_gammas
        return features, feature_maps, fitted_gammas

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        """
        Fit the nested RKHS IV estimator.

        Parameters:
            A (array-like or scipy.sparse matrix): First nested-stage treatment or endogenous block.
            B (array-like or scipy.sparse matrix): Second nested-stage treatment or endogenous block.
            C (array-like or scipy.sparse matrix): Second nested-stage instrument block.
            D (array-like or scipy.sparse matrix): First nested-stage instrument block.
            Y (array-like): Outcome values.
            W (array-like or None): Optional observation-level multipliers in
                the bridge residual ``h(B) - W*g(A)``; defaults to one.
            subsetted (bool): Whether to apply stage-specific subset restrictions.
            subset_ind1 (array-like or None): Indicator or mask selecting the
                first-stage ``D``-moment rows.
            subset_ind2 (array-like or None): Indicator or mask selecting the
                second-stage ``C``-moment rows; defaults to the complement of
                ``subset_ind1``.
        """
        Y = _to_column_vector(Y)
        A, B, C, D = map(_as_feature_input, (A, B, C, D))
        n = Y.shape[0]
        if any(values.ndim == 0 or values.shape[0] != n
               for values in (A, B, C, D)):
            raise ValueError(
                "A, B, C, D, and Y must have the same number of observations."
            )
        ind1, ind2 = self._validate_subset_inputs(
            n,
            subsetted=subsetted,
            subset_ind1=subset_ind1,
            subset_ind2=subset_ind2,
        )
        multipliers = self._as_multiplier_vector(W, n)

        delta = self._get_delta(n)
        alpha = (
            delta**4
            if _check_auto(self.alpha_scale)
            else self._get_alpha(delta, self.alpha_scale)
        )
        (FA, FB, FC, FD), _, _ = self._fit_l2_feature_maps(
            A,
            B,
            C,
            D,
            gamma=self.gamma,
            n_components=self.n_components,
            store=True,
        )
        system_terms = self._l2_joint_system_terms(
            FA,
            FB,
            FC,
            FD,
            multipliers,
            Y,
            subsetted=subsetted,
            ind1=ind1,
            ind2=ind2,
        )
        (
            self.theta_a,
            self.theta_b,
            _,
        ) = self._solve_l2_joint_feature_system(
            *system_terms,
            alpha,
        )

        self.A = A.copy()
        self.B = B.copy()
        self.fitted_alpha_ = alpha
        return self

    def predict(self, B_test, A_test=None):
        """
        Predict fitted bridge values for test data.

        Parameters:
            B_test (array-like): Test data for the second nested-stage block.
            A_test (array-like or None): Optional test data for the first nested-stage block.

        Returns:
            numpy.ndarray or tuple: ``h_hat(B_test)`` when ``A_test`` is
            omitted; otherwise ``(h_hat(B_test), g_hat(A_test))``.
        """
        pred_b = self.featB.transform(B_test) @ self.theta_b
        if A_test is None:
            return pred_b
        pred_a = self.featA.transform(A_test) @ self.theta_a
        return pred_b, pred_a


class ApproxRKHS2IVL2CV(ApproxRKHS2IVL2):
    """
    Cross-validated approximate common-penalty Appendix L.1 / Algorithm 2 RKHS
    estimator, with ``mu_prime = mu = alpha``.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int, float, or array-like): Component count, sample
            fraction, or candidate grid. Values in (0, 1] are sample
            fractions with a floor of 10 and are then capped at ``n_samples``;
            integer-like values greater than 1 are fixed component counts.
        kernel (str or callable): Kernel function or string identifier.
        gamma (str, float, or array-like): Automatic RBF coefficient, fixed
            coefficient, or candidate coefficient grid; the RBF kernel is
            ``exp(-gamma * ||x - x'||^2)``.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        kernel_params (dict): Additional parameters for the kernel.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        n_alphas (int): Number of alpha scales to try.
        cv (int): Number of folds for cross-validation.
        expand_alpha_grid (bool): Whether to expand the alpha grid when the CV optimum lies on a boundary.
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

    def _run_l2_feature_cv(
        self,
        A,
        B,
        C,
        D,
        Y,
        multipliers,
        alpha_scales,
        gamma_candidate,
        n_components_candidate,
        validation_n_components,
        subsetted,
        ind1,
        ind2,
    ):
        alpha_scales = self._normalize_positive_grid(
            alpha_scales, "alpha_scales"
        )
        fold_scores_all = []

        for train, test in KFold(n_splits=self.cv).split(Y):
            n_train = len(train)
            n_test = len(test)
            A_train, B_train, C_train, D_train = (
                values[train] for values in (A, B, C, D)
            )
            A_test, B_test, C_test, D_test = (
                values[test] for values in (A, B, C, D)
            )
            (
                (FA_train, FB_train, FC_train, FD_train),
                feature_maps,
                fitted_gammas,
            ) = self._fit_l2_feature_maps(
                A_train,
                B_train,
                C_train,
                D_train,
                gamma=gamma_candidate,
                n_components=n_components_candidate,
            )
            FA_test, FB_test, FC_test, FD_test = tuple(
                feature_map.transform(values)
                for feature_map, values in zip(
                    feature_maps, (A_test, B_test, C_test, D_test)
                )
            )

            if (
                self._resolve_n_components(
                    n_samples=n_train,
                    n_components=validation_n_components,
                )
                != FC_train.shape[1]
            ):
                validation_c = self._get_new_approx_instance(
                    n_samples=n_train,
                    fitted_gamma=fitted_gammas[2],
                    n_components=validation_n_components,
                )
                validation_d = self._get_new_approx_instance(
                    n_samples=n_train,
                    fitted_gamma=fitted_gammas[3],
                    n_components=validation_n_components,
                )
                validation_c.fit(C_train)
                validation_d.fit(D_train)
                FC_score = validation_c.transform(C_test)
                FD_score = validation_d.transform(D_test)
            else:
                FC_score = FC_test
                FD_score = FD_test

            if subsetted:
                train_ind1 = self._local_subset_indices(train, ind1)
                train_ind2 = self._local_subset_indices(train, ind2)
                test_ind1 = self._local_subset_indices(test, ind1)
                test_ind2 = self._local_subset_indices(test, ind2)

                if (train_ind1.size == 0 or train_ind2.size == 0
                        or test_ind1.size == 0 or test_ind2.size == 0):
                    continue
            else:
                train_ind1 = train_ind2 = None
                test_ind1 = test_ind2 = None

            system_terms = self._l2_joint_system_terms(
                FA_train,
                FB_train,
                FC_train,
                FD_train,
                multipliers[train],
                Y[train],
                subsetted=subsetted,
                ind1=train_ind1,
                ind2=train_ind2,
            )
            action_c_test = self._instrument_action(
                FC_score,
                subset_indices=test_ind2 if subsetted else None,
                scale_n=n_test,
            )
            action_d_test = self._instrument_action(
                FD_score,
                subset_indices=test_ind1 if subsetted else None,
                scale_n=n_test,
            )
            delta_train = self._get_delta(n_train)
            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = float(alpha_scale) * (delta_train**4)
                theta_a, theta_b, _ = (
                    self._solve_l2_joint_feature_system(
                        *system_terms,
                        alpha,
                    )
                )
                prediction_a = FA_test @ theta_a
                prediction_b = FB_test @ theta_b
                residual_y = Y[test] - prediction_a
                residual_bridge = (
                    prediction_b
                    - multipliers[test, None] * prediction_a
                )
                d_score = self._instrument_cross_gram(
                    action_d_test,
                    residual_y,
                )
                c_score = self._instrument_cross_gram(
                    action_c_test,
                    residual_bridge,
                )
                fold_scores.append(
                    (_to_scalar(d_score) + _to_scalar(c_score))
                    / (n_test ** 2)
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
        """
        Fit the nested RKHS IV estimator.

        Parameters:
            A (array-like or scipy.sparse matrix): First nested-stage treatment or endogenous block.
            B (array-like or scipy.sparse matrix): Second nested-stage treatment or endogenous block.
            C (array-like or scipy.sparse matrix): Second nested-stage instrument block.
            D (array-like or scipy.sparse matrix): First nested-stage instrument block.
            Y (array-like): Outcome values.
            W (array-like or None): Optional observation-level multipliers in
                the bridge residual ``h(B) - W*g(A)``; defaults to one.
            subsetted (bool): Whether to apply stage-specific subset restrictions.
            subset_ind1 (array-like or None): Indicator or mask selecting the
                first-stage ``D``-moment rows.
            subset_ind2 (array-like or None): Indicator or mask selecting the
                second-stage ``C``-moment rows; defaults to the complement of
                ``subset_ind1``.
        """
        Y = _to_column_vector(Y)
        A, B, C, D = map(_as_feature_input, (A, B, C, D))
        n = Y.shape[0]
        if any(values.ndim == 0 or values.shape[0] != n
               for values in (A, B, C, D)):
            raise ValueError(
                "A, B, C, D, and Y must have the same number of observations."
            )
        ind1, ind2 = self._validate_subset_inputs(
            n,
            subsetted=subsetted,
            subset_ind1=subset_ind1,
            subset_ind2=subset_ind2,
        )
        multipliers = self._as_multiplier_vector(W, n)
        alpha_scales = self._get_alpha_scales()
        delta = self._get_delta(n)
        gamma_candidates = self._as_candidate_values(
            self.gamma, "gamma", allow_auto=True, positive=True
        )
        ncomp_candidates = self._as_candidate_values(
            self.n_components,
            "n_components",
            allow_auto=False,
            positive=True,
        )
        validation_n_components = max(
            ncomp_candidates,
            key=lambda candidate: self._resolve_n_components(
                n_samples=n, n_components=candidate
            ),
        )
        candidate_summaries = []
        candidate_states = []

        for gamma_candidate in gamma_candidates:
            for n_components_candidate in ncomp_candidates:
                cv_result = self._run_alpha_cv_with_optional_expansion(
                    lambda alpha_grid, gamma_value=gamma_candidate,
                    component_value=n_components_candidate:
                        self._run_l2_feature_cv(
                            A,
                            B,
                            C,
                            D,
                            Y,
                            multipliers,
                            alpha_grid,
                            gamma_value,
                            component_value,
                            validation_n_components,
                            subsetted,
                            ind1,
                            ind2,
                        ),
                    alpha_scales,
                )
                cv_result["best_alpha"] = (
                    cv_result["best_alpha_scale"] * (delta**4)
                )
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

        best_candidate_idx = int(np.argmin([
            state[-1]["best_score"] for state in candidate_states
        ]))
        best_gamma, best_n_components, cv_result = (
            candidate_states[best_candidate_idx]
        )
        self.best_gamma_ = best_gamma
        self.best_n_components_ = best_n_components
        self.cv_gamma_grid_ = list(gamma_candidates)
        self.cv_n_components_grid_ = list(ncomp_candidates)
        self.cv_candidate_summaries_ = candidate_summaries

        self.alpha_scales_ = np.asarray(
            cv_result["alpha_scales_used"], dtype=float
        )
        self.avg_scores = cv_result["avg_scores"]
        self.best_alpha_scale = cv_result["best_alpha_scale"]
        self.best_alpha = cv_result["best_alpha"]
        self._set_cv_diagnostics(cv_result)

        (FA, FB, FC, FD), _, _ = self._fit_l2_feature_maps(
            A,
            B,
            C,
            D,
            gamma=best_gamma,
            n_components=best_n_components,
            store=True,
        )
        system_terms = self._l2_joint_system_terms(
            FA,
            FB,
            FC,
            FD,
            multipliers,
            Y,
            subsetted=subsetted,
            ind1=ind1,
            ind2=ind2,
        )
        (
            self.theta_a,
            self.theta_b,
            _,
        ) = self._solve_l2_joint_feature_system(
            *system_terms,
            self.best_alpha,
        )
        self.A = A.copy()
        self.B = B.copy()
        self.fitted_alpha_ = self.best_alpha
        return self


class ApproxRKHS2IV(ApproxRKHS2IVL2):
    """
    Approximate simultaneous RKHS-norm estimator using finite kernel features.

    The learner and instrument equations are contracted directly through the
    finite feature matrices. The fitted system therefore has at most
    ``n_components(A) + n_components(B)`` rows and does not require
    sample-sized Gram matrices.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or
            'rbfsampler').
        n_components (int or float): Number of approximation components.
            Values in (0, 1] are sample fractions with a floor of 10 and are
            then capped at ``n_samples``; integer-like values greater than 1
            are fixed component counts.
        kernel (str or callable): Kernel function or string identifier.
        gamma (str or float): Kernel coefficient passed to scikit-learn; for
            RBF, the kernel is ``exp(-gamma * ||x - x'||^2)``.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        kernel_params (dict): Additional parameters for the kernel.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scale (str or float): Scale of the learner regularization.
    """

    def __init__(
        self,
        kernel_approx='nystrom',
        n_components=10,
        kernel='rbf',
        gamma=2,
        degree=3,
        coef0=1,
        kernel_params=None,
        delta_scale='auto',
        delta_exp='auto',
        alpha_scale='auto',
    ):
        super().__init__(
            kernel_approx=kernel_approx,
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            delta_scale=delta_scale,
            delta_exp=delta_exp,
            alpha_scale=alpha_scale,
        )

    @staticmethod
    def _ridge_feature_cross_gram(
        instrument_features,
        values,
        ridge,
        subset_indices=None,
        scale_n=None,
    ):
        instrument_features = np.asarray(
            instrument_features, dtype=float
        )
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if subset_indices is None:
            selected_features = instrument_features
            selected_values = values
            scale = 1.0
        else:
            subset_indices = np.asarray(subset_indices, dtype=int)
            if subset_indices.size == 0:
                raise ValueError(
                    "Instrument contraction requested with zero selected rows."
                )
            selected_features = instrument_features[subset_indices]
            selected_values = values[subset_indices]
            scale = float(scale_n) / subset_indices.size

        feature_gram = (
            selected_features.T @ selected_features
            + ridge * np.eye(selected_features.shape[1])
        )
        coordinates = selected_features.T @ selected_values
        solved_coordinates = np.linalg.solve(feature_gram, coordinates)
        cross_gram = scale * coordinates.T @ solved_coordinates
        return 0.5 * (cross_gram + cross_gram.T)

    def _joint_feature_system(
        self,
        FA,
        FB,
        FC,
        FD,
        multipliers,
        Y,
        alpha,
        subsetted=False,
        ind1=None,
        ind2=None,
    ):
        n = Y.shape[0]
        rank_a = FA.shape[1]
        rank_b = FB.shape[1]
        weighted_FA = multipliers[:, None] * FA

        d_values = np.hstack([FA, Y])
        d_cross = self._ridge_feature_cross_gram(
            FD,
            d_values,
            ridge=1.0,
            subset_indices=ind1 if subsetted else None,
            scale_n=n,
        )
        c_values = np.hstack([weighted_FA, FB])
        c_cross = self._ridge_feature_cross_gram(
            FC,
            c_values,
            ridge=1.0,
            subset_indices=ind2 if subsetted else None,
            scale_n=n,
        )

        system_aa = (
            d_cross[:rank_a, :rank_a]
            + c_cross[:rank_a, :rank_a]
            + alpha * np.eye(rank_a)
        )
        system_ab = -c_cross[:rank_a, rank_a:]
        system_bb = (
            c_cross[rank_a:, rank_a:]
            + alpha * np.eye(rank_b)
        )
        system = np.block([
            [system_aa, system_ab],
            [system_ab.T, system_bb],
        ])
        rhs = np.vstack([
            d_cross[:rank_a, rank_a:],
            np.zeros((rank_b, Y.shape[1]), dtype=float),
        ])
        return system, rhs

    def _solve_joint_feature_system(
        self,
        FA,
        FB,
        FC,
        FD,
        multipliers,
        Y,
        alpha,
        subsetted=False,
        ind1=None,
        ind2=None,
    ):
        system, rhs = self._joint_feature_system(
            FA,
            FB,
            FC,
            FD,
            multipliers,
            Y,
            alpha,
            subsetted=subsetted,
            ind1=ind1,
            ind2=ind2,
        )
        coefficients = np.linalg.pinv(system) @ rhs
        rank_a = FA.shape[1]
        return (
            coefficients[:rank_a],
            coefficients[rank_a:],
            system,
            rhs,
        )

    def _fit_joint_feature_maps(
        self,
        A,
        B,
        C,
        D,
        gamma,
        n_components,
        store=False,
    ):
        n = A.shape[0]
        gamma_a = self._resolve_gamma_value(A, "A", gamma)
        gamma_b = self._resolve_gamma_value(B, "B", gamma)
        gamma_c = self._resolve_gamma_value(C, "C", gamma)
        gamma_d = self._resolve_gamma_value(D, "D", gamma)
        feature_maps = (
            self._get_new_approx_instance(
                n_samples=n,
                fitted_gamma=gamma_a,
                n_components=n_components,
            ),
            self._get_new_approx_instance(
                n_samples=n,
                fitted_gamma=gamma_b,
                n_components=n_components,
            ),
            self._get_new_approx_instance(
                n_samples=n,
                fitted_gamma=gamma_c,
                n_components=n_components,
            ),
            self._get_new_approx_instance(
                n_samples=n,
                fitted_gamma=gamma_d,
                n_components=n_components,
            ),
        )
        features = tuple(
            feature_map.fit_transform(values)
            for feature_map, values in zip(
                feature_maps, (A, B, C, D)
            )
        )
        fitted_gammas = (gamma_a, gamma_b, gamma_c, gamma_d)

        if store:
            self.featA, self.featB, self.featC, self.featD = feature_maps
            self.FA, self.FB, self.FC, self.FD = features
            (
                self.gamma_a_,
                self.gamma_b_,
                self.gamma_c_,
                self.gamma_d_,
            ) = fitted_gammas
        return features, feature_maps, fitted_gammas

    def _score_joint_residuals(
        self,
        FC,
        FD,
        residual_y,
        residual_bridge,
        subsetted=False,
        ind1=None,
        ind2=None,
    ):
        n = residual_y.shape[0]
        d_score = self._ridge_feature_cross_gram(
            FD,
            residual_y,
            ridge=1.0,
            subset_indices=ind1 if subsetted else None,
            scale_n=n,
        )
        c_score = self._ridge_feature_cross_gram(
            FC,
            residual_bridge,
            ridge=1.0,
            subset_indices=ind2 if subsetted else None,
            scale_n=n,
        )
        return (_to_scalar(d_score) + _to_scalar(c_score)) / (n ** 2)

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        """
        Fit the nested RKHS IV estimator.

        Parameters:
            A (array-like or scipy.sparse matrix): First nested-stage treatment or endogenous block.
            B (array-like or scipy.sparse matrix): Second nested-stage treatment or endogenous block.
            C (array-like or scipy.sparse matrix): Second nested-stage instrument block.
            D (array-like or scipy.sparse matrix): First nested-stage instrument block.
            Y (array-like): Outcome values.
            W (array-like or None): Optional observation-level multipliers in
                the bridge residual ``h(B) - W*g(A)``; defaults to one.
            subsetted (bool): Whether to apply stage-specific subset restrictions.
            subset_ind1 (array-like or None): Indicator or mask selecting the
                first-stage ``D``-moment rows.
            subset_ind2 (array-like or None): Indicator or mask selecting the
                second-stage ``C``-moment rows; defaults to the complement of
                ``subset_ind1``.
        """
        Y = _to_column_vector(Y)
        A, B, C, D = map(_as_feature_input, (A, B, C, D))
        n = Y.shape[0]
        if any(values.ndim == 0 or values.shape[0] != n
               for values in (A, B, C, D)):
            raise ValueError(
                "A, B, C, D, and Y must have the same number of observations."
            )
        ind1, ind2 = self._validate_subset_inputs(
            n,
            subsetted=subsetted,
            subset_ind1=subset_ind1,
            subset_ind2=subset_ind2,
        )
        multipliers = self._as_multiplier_vector(W, n)

        delta = self._get_delta(n)
        alpha = (
            delta**4
            if _check_auto(self.alpha_scale)
            else self._get_alpha(delta, self.alpha_scale)
        )
        (FA, FB, FC, FD), _, _ = self._fit_joint_feature_maps(
            A,
            B,
            C,
            D,
            gamma=self.gamma,
            n_components=self.n_components,
            store=True,
        )
        (
            self.theta_a,
            self.theta_b,
            _,
            _,
        ) = self._solve_joint_feature_system(
            FA,
            FB,
            FC,
            FD,
            multipliers,
            Y,
            alpha,
            subsetted=subsetted,
            ind1=ind1,
            ind2=ind2,
        )

        self.A = A.copy()
        self.B = B.copy()
        self.fitted_alpha_ = alpha
        return self


class ApproxRKHS2IVCV(ApproxRKHS2IV):
    """
    Cross-validated approximate simultaneous RKHS-norm estimator.

    Parameters:
        kernel_approx (str): Kernel approximation method ('nystrom' or 'rbfsampler').
        n_components (int, float, or array-like): Component count, sample
            fraction, or candidate grid. Values in (0, 1] are sample
            fractions with a floor of 10 and are then capped at ``n_samples``;
            integer-like values greater than 1 are fixed component counts.
        kernel (str or callable): Kernel function or string identifier.
        gamma (str, float, or array-like): Automatic RBF coefficient, fixed
            coefficient, or candidate coefficient grid; the RBF kernel is
            ``exp(-gamma * ||x - x'||^2)``.
        degree (int): Degree for polynomial kernels.
        coef0 (float): Zero coefficient for polynomial kernels.
        kernel_params (dict): Additional parameters for the kernel.
        delta_scale (str or float): Scale of the critical radius.
        delta_exp (str or float): Exponent of the critical radius.
        alpha_scales (str or array-like): Scale of the regularization parameter.
        n_alphas (int): Number of alpha scales to try.
        cv (int): Number of folds for cross-validation.
        expand_alpha_grid (bool): Whether to expand the alpha grid when the CV optimum lies on a boundary.
    """

    def __init__(
        self,
        kernel_approx='nystrom',
        n_components=10,
        kernel='rbf',
        gamma=2,
        degree=3,
        coef0=1,
        kernel_params=None,
        delta_scale='auto',
        delta_exp='auto',
        alpha_scales='auto',
        n_alphas=30,
        cv=6,
        expand_alpha_grid=True,
    ):
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

    def _run_joint_feature_cv(
        self,
        A,
        B,
        C,
        D,
        Y,
        multipliers,
        alpha_scales,
        gamma_candidate,
        n_components_candidate,
        validation_n_components,
        subsetted,
        ind1,
        ind2,
    ):
        alpha_scales = self._normalize_positive_grid(
            alpha_scales, "alpha_scales"
        )
        fold_scores_all = []

        for train, test in KFold(n_splits=self.cv).split(Y):
            n_train = len(train)
            A_train, B_train, C_train, D_train = (
                values[train] for values in (A, B, C, D)
            )
            A_test, B_test, C_test, D_test = (
                values[test] for values in (A, B, C, D)
            )
            (
                (FA_train, FB_train, FC_train, FD_train),
                feature_maps,
                fitted_gammas,
            ) = self._fit_joint_feature_maps(
                A_train,
                B_train,
                C_train,
                D_train,
                gamma=gamma_candidate,
                n_components=n_components_candidate,
            )
            FA_test, FB_test, FC_test, FD_test = tuple(
                feature_map.transform(values)
                for feature_map, values in zip(
                    feature_maps, (A_test, B_test, C_test, D_test)
                )
            )

            if (
                self._resolve_n_components(
                    n_samples=n_train,
                    n_components=validation_n_components,
                )
                != FC_train.shape[1]
            ):
                validation_c = self._get_new_approx_instance(
                    n_samples=n_train,
                    fitted_gamma=fitted_gammas[2],
                    n_components=validation_n_components,
                )
                validation_d = self._get_new_approx_instance(
                    n_samples=n_train,
                    fitted_gamma=fitted_gammas[3],
                    n_components=validation_n_components,
                )
                validation_c.fit(C_train)
                validation_d.fit(D_train)
                FC_score = validation_c.transform(C_test)
                FD_score = validation_d.transform(D_test)
            else:
                FC_score = FC_test
                FD_score = FD_test

            if subsetted:
                train_ind1 = self._local_subset_indices(train, ind1)
                train_ind2 = self._local_subset_indices(train, ind2)
                test_ind1 = self._local_subset_indices(test, ind1)
                test_ind2 = self._local_subset_indices(test, ind2)

                if (train_ind1.size == 0 or train_ind2.size == 0
                        or test_ind1.size == 0 or test_ind2.size == 0):
                    continue
            else:
                train_ind1 = train_ind2 = None
                test_ind1 = test_ind2 = None

            delta_train = self._get_delta(n_train)
            fold_scores = []
            for alpha_scale in alpha_scales:
                alpha = float(alpha_scale) * (delta_train**4)
                theta_a, theta_b, _, _ = (
                    self._solve_joint_feature_system(
                        FA_train,
                        FB_train,
                        FC_train,
                        FD_train,
                        multipliers[train],
                        Y[train],
                        alpha,
                        subsetted=subsetted,
                        ind1=train_ind1,
                        ind2=train_ind2,
                    )
                )
                prediction_a = FA_test @ theta_a
                prediction_b = FB_test @ theta_b
                residual_y = Y[test] - prediction_a
                residual_bridge = (
                    prediction_b
                    - multipliers[test, None] * prediction_a
                )
                fold_scores.append(
                    self._score_joint_residuals(
                        FC_score,
                        FD_score,
                        residual_y,
                        residual_bridge,
                        subsetted=subsetted,
                        ind1=test_ind1,
                        ind2=test_ind2,
                    )
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
        """
        Fit the nested RKHS IV estimator.

        Parameters:
            A (array-like or scipy.sparse matrix): First nested-stage treatment or endogenous block.
            B (array-like or scipy.sparse matrix): Second nested-stage treatment or endogenous block.
            C (array-like or scipy.sparse matrix): Second nested-stage instrument block.
            D (array-like or scipy.sparse matrix): First nested-stage instrument block.
            Y (array-like): Outcome values.
            W (array-like or None): Optional observation-level multipliers in
                the bridge residual ``h(B) - W*g(A)``; defaults to one.
            subsetted (bool): Whether to apply stage-specific subset restrictions.
            subset_ind1 (array-like or None): Indicator or mask selecting the
                first-stage ``D``-moment rows.
            subset_ind2 (array-like or None): Indicator or mask selecting the
                second-stage ``C``-moment rows; defaults to the complement of
                ``subset_ind1``.
        """
        Y = _to_column_vector(Y)
        A, B, C, D = map(_as_feature_input, (A, B, C, D))
        n = Y.shape[0]
        if any(values.ndim == 0 or values.shape[0] != n
               for values in (A, B, C, D)):
            raise ValueError(
                "A, B, C, D, and Y must have the same number of observations."
            )
        ind1, ind2 = self._validate_subset_inputs(
            n,
            subsetted=subsetted,
            subset_ind1=subset_ind1,
            subset_ind2=subset_ind2,
        )
        multipliers = self._as_multiplier_vector(W, n)
        alpha_scales = self._get_alpha_scales()
        delta = self._get_delta(n)
        gamma_candidates = self._as_candidate_values(
            self.gamma, "gamma", allow_auto=True, positive=True
        )
        ncomp_candidates = self._as_candidate_values(
            self.n_components,
            "n_components",
            allow_auto=False,
            positive=True,
        )
        validation_n_components = max(
            ncomp_candidates,
            key=lambda candidate: self._resolve_n_components(
                n_samples=n, n_components=candidate
            ),
        )
        candidate_summaries = []
        candidate_states = []

        for gamma_candidate in gamma_candidates:
            for n_components_candidate in ncomp_candidates:
                cv_result = self._run_alpha_cv_with_optional_expansion(
                    lambda alpha_grid, gamma_value=gamma_candidate,
                    component_value=n_components_candidate:
                        self._run_joint_feature_cv(
                            A,
                            B,
                            C,
                            D,
                            Y,
                            multipliers,
                            alpha_grid,
                            gamma_value,
                            component_value,
                            validation_n_components,
                            subsetted,
                            ind1,
                            ind2,
                        ),
                    alpha_scales,
                )
                cv_result["best_alpha"] = (
                    cv_result["best_alpha_scale"] * (delta**4)
                )
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

        best_candidate_idx = int(np.argmin([
            state[-1]["best_score"] for state in candidate_states
        ]))
        best_gamma, best_n_components, cv_result = (
            candidate_states[best_candidate_idx]
        )
        self.best_gamma_ = best_gamma
        self.best_n_components_ = best_n_components
        self.cv_gamma_grid_ = list(gamma_candidates)
        self.cv_n_components_grid_ = list(ncomp_candidates)
        self.cv_candidate_summaries_ = candidate_summaries

        self.alpha_scales_ = np.asarray(
            cv_result["alpha_scales_used"], dtype=float
        )
        self.avg_scores = cv_result["avg_scores"]
        self.best_alpha_scale = cv_result["best_alpha_scale"]
        self.best_alpha = cv_result["best_alpha"]
        self._set_cv_diagnostics(cv_result)

        (FA, FB, FC, FD), _, _ = self._fit_joint_feature_maps(
            A,
            B,
            C,
            D,
            gamma=best_gamma,
            n_components=best_n_components,
            store=True,
        )
        (
            self.theta_a,
            self.theta_b,
            _,
            _,
        ) = self._solve_joint_feature_system(
            FA,
            FB,
            FC,
            FD,
            multipliers,
            Y,
            self.best_alpha,
            subsetted=subsetted,
            ind1=ind1,
            ind2=ind2,
        )
        self.A = A.copy()
        self.B = B.copy()
        self.fitted_alpha_ = self.best_alpha
        return self
