"""Shared numerical utilities for localization and ratio moments.

The estimators in this package can be written as solutions to
``E[H - a * theta] = 0``, where ``H`` is an uncentered score contribution
and ``a`` is the loading that defines the target. Ordinary averages use
``a=1``; localized and subgroup targets use nonconstant loadings.
"""

import numpy as np


def as_2d(values, name):
    """Return an observation-level array with an explicit feature axis.

    One-dimensional inputs are interpreted as a single feature.  Rejecting
    higher-dimensional inputs here prevents accidental NumPy broadcasting in
    downstream score calculations.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    return values


def as_column(values, name):
    """Return a scalar observation-level variable as an ``(n, 1)`` array."""
    values = as_2d(values, name)
    if values.shape[1] != 1:
        raise ValueError(f"{name} must contain exactly one column.")
    return values


def _as_localization_matrix(values, name):
    """Convert localization covariates to a finite two-dimensional array."""
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")

    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one column.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _format_localization_values(v_values, n_covariates, default_rows):
    """Format localization evaluation values as one row per target."""
    if v_values is None:
        return np.mean(default_rows, axis=0, keepdims=True)

    try:
        values = np.asarray(v_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("v_values must be a numeric array.") from exc

    if values.ndim == 0:
        if n_covariates != 1:
            raise ValueError(
                "A scalar v_values is valid only with one localization covariate."
            )
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        if n_covariates == 1:
            values = values.reshape(-1, 1)
        elif values.size == n_covariates:
            values = values.reshape(1, -1)
        else:
            raise ValueError(
                "A one-dimensional v_values must either be a grid for one "
                "localization covariate or contain one value per covariate."
            )
    elif values.ndim == 2:
        if values.shape[1] != n_covariates:
            raise ValueError(
                "v_values must have one column per localization covariate."
            )
    else:
        raise ValueError("v_values must be scalar, one-dimensional, or two-dimensional.")

    if values.shape[0] == 0:
        raise ValueError("v_values must contain at least one evaluation point.")
    if not np.all(np.isfinite(values)):
        raise ValueError("v_values must contain only finite values.")
    return values


def canonicalize_localization_inputs(V, v_values=None):
    """Canonicalize localization covariates and evaluation values.

    Parameters
    ----------
    V : array-like
        Localization covariates. A one-dimensional input is interpreted as a
        single covariate.
    v_values : array-like, optional
        Evaluation values. For one localization covariate, a one-dimensional
        input is a grid. With multiple covariates, a one-dimensional input is
        one multivariate evaluation point. If omitted, the column means of
        ``V`` form one evaluation point.

    Returns
    -------
    V_array : ndarray of shape (n_samples, n_covariates)
    values_array : ndarray of shape (n_targets, n_covariates)
    """
    V_array = _as_localization_matrix(V, "V")
    values_array = _format_localization_values(
        v_values, V_array.shape[1], V_array
    )
    return V_array, values_array


def _canonicalize_bandwidth(bandwidth, n_covariates):
    """Return a finite, positive bandwidth vector."""
    try:
        bandwidth = np.asarray(bandwidth, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("The localization bandwidth must be numeric.") from exc

    if bandwidth.ndim == 0:
        bandwidth = np.repeat(float(bandwidth), n_covariates)
    elif bandwidth.ndim == 1:
        if bandwidth.size == 1:
            bandwidth = np.repeat(float(bandwidth[0]), n_covariates)
        elif bandwidth.size != n_covariates:
            raise ValueError(
                "The localization bandwidth must be scalar or have one value "
                "per localization covariate."
            )
    else:
        raise ValueError("The localization bandwidth must be scalar or one-dimensional.")

    if not np.all(np.isfinite(bandwidth)) or np.any(bandwidth <= 0):
        raise ValueError("All localization bandwidths must be finite and positive.")
    return bandwidth


def resolve_localization_bandwidth(target_V, bw_loc):
    """Resolve one run-level localization bandwidth.

    ``target_V`` contains exactly the population rows that define an automatic
    bandwidth. The existing componentwise Silverman and Scott rules are
    retained for backward compatibility.
    """
    target_V = _as_localization_matrix(target_V, "target_V")
    n_covariates = target_V.shape[1]

    if isinstance(bw_loc, str):
        if bw_loc == "silverman":
            iqr = (
                np.percentile(target_V, 75, axis=0)
                - np.percentile(target_V, 25, axis=0)
            )
            scale = np.minimum(np.std(target_V, axis=0), iqr / 1.349)
            bandwidth = 0.9 * scale * target_V.shape[0] ** (-0.2)
        elif bw_loc == "scott":
            scale = np.std(target_V, axis=0)
            bandwidth = 1.059 * scale * target_V.shape[0] ** (-0.2)
        else:
            raise ValueError(
                "bw_loc must be 'silverman', 'scott', or a numeric bandwidth."
            )
    else:
        bandwidth = bw_loc

    try:
        return _canonicalize_bandwidth(bandwidth, n_covariates)
    except ValueError as exc:
        if isinstance(bw_loc, str):
            raise ValueError(
                f"The {bw_loc} bandwidth is zero or nonfinite for at least one "
                "localization covariate. Remove constant covariates or provide "
                "a finite positive bandwidth."
            ) from exc
        raise


def localization_kernel_matrix(V, v_values, bandwidth, kernel):
    """Compute raw product-kernel weights for every observation and target.

    Parameters
    ----------
    V : array-like
        Localization covariates.
    v_values : array-like
        Evaluation values, with one row per target.
    bandwidth : array-like
        One positive bandwidth per localization covariate.
    kernel : callable
        An instantiated statsmodels kernel, for example
        ``kernel_switch['gau']()``.

    Returns
    -------
    ndarray of shape (n_samples, n_targets)
        Raw product-kernel weights.
    """
    V = _as_localization_matrix(V, "V")
    v_values = _format_localization_values(
        v_values, V.shape[1], V
    )
    bandwidth = _canonicalize_bandwidth(bandwidth, V.shape[1])
    if not callable(kernel):
        raise TypeError("kernel must be an instantiated callable kernel.")

    weights = np.ones((V.shape[0], v_values.shape[0]), dtype=float)
    domain = getattr(kernel, "domain", None)
    for column in range(V.shape[1]):
        scaled = (
            V[:, column, np.newaxis] - v_values[np.newaxis, :, column]
        ) / bandwidth[column]
        component = np.asarray(kernel(scaled), dtype=float)
        if component.shape != scaled.shape:
            try:
                component = np.broadcast_to(component, scaled.shape)
            except ValueError as exc:
                raise ValueError(
                    "The localization kernel returned values with an invalid shape."
                ) from exc
        if domain is not None:
            component = component * (
                (domain[0] <= scaled) & (scaled <= domain[1])
            )
        weights *= component

    if not np.all(np.isfinite(weights)):
        raise ValueError("The localization kernel produced nonfinite weights.")
    return weights


def localization_normalizers(target_V, v_values, bandwidth, kernel):
    """Compute fixed kernel normalizers from the supplied target rows."""
    kernel_matrix = localization_kernel_matrix(
        target_V, v_values, bandwidth, kernel
    )
    normalizers = np.mean(kernel_matrix, axis=0)
    if np.any(~np.isfinite(normalizers)) or np.any(
        np.abs(normalizers) <= np.finfo(float).eps
    ):
        raise ValueError(
            "The localization kernel has a zero or nonfinite target-population "
            "normalizer. Choose supported evaluation values or increase the bandwidth."
        )
    return normalizers


def localization_loadings(V, v_values, bandwidth, kernel, normalizers):
    """Compute kernel loadings using fixed run-level normalizers."""
    kernel_matrix = localization_kernel_matrix(V, v_values, bandwidth, kernel)
    try:
        normalizers = np.asarray(normalizers, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalizers must be a numeric array.") from exc
    if normalizers.ndim == 0:
        normalizers = normalizers.reshape(1)
    if normalizers.ndim != 1 or normalizers.size != kernel_matrix.shape[1]:
        raise ValueError("normalizers must contain one value per evaluation point.")
    if np.any(~np.isfinite(normalizers)) or np.any(
        np.abs(normalizers) <= np.finfo(float).eps
    ):
        raise ValueError("All localization normalizers must be finite and nonzero.")
    return kernel_matrix / normalizers.reshape(1, -1)


def prepare_localization(V, v_values, bw_loc, kernel, target_V=None):
    """Prepare one fixed localization specification for an estimator run.

    Returns the canonical full localization array, canonical evaluation rows,
    one bandwidth vector, and fixed target-population normalizers. Use
    :func:`localization_loadings` with these stored quantities in every fold.
    """
    V, v_values = canonicalize_localization_inputs(V, v_values)
    if target_V is None:
        target_V = V
    else:
        target_V = _as_localization_matrix(target_V, "target_V")
        if target_V.shape[1] != V.shape[1]:
            raise ValueError(
                "target_V and V must have the same number of localization covariates."
            )

    bandwidth = resolve_localization_bandwidth(target_V, bw_loc)
    normalizers = localization_normalizers(
        target_V, v_values, bandwidth, kernel
    )
    return V, v_values, bandwidth, normalizers


def _ratio_estimate_and_influence(score, loading):
    """Return a ratio estimate and its observation-level influence values."""
    score = np.asarray(score, dtype=float)
    loading = np.asarray(loading, dtype=float)
    if score.ndim == 1:
        score = score.reshape(-1, 1)
    if loading.ndim == 1:
        loading = loading.reshape(-1, 1)
    if score.shape[0] != loading.shape[0]:
        raise ValueError(
            "score and loading must contain the same number of observations."
        )

    try:
        loading = np.broadcast_to(loading, score.shape)
    except ValueError as exc:
        raise ValueError(
            "loading must be broadcastable to the shape of score."
        ) from exc

    loading_mean = np.mean(loading, axis=0)
    threshold = np.finfo(float).eps
    if np.any(~np.isfinite(loading_mean)) or np.any(
        np.abs(loading_mean) <= threshold
    ):
        raise ValueError(
            "The target loading must have a finite, nonzero empirical mean."
        )

    theta = np.mean(score, axis=0) / loading_mean
    influence = (score - loading * theta) / loading_mean
    return theta, influence


def _summarize_influence(influence):
    """Return the marginal variances and covariance of influence values."""
    variance = np.var(influence, axis=0, ddof=1)
    covariance = np.atleast_2d(np.cov(influence, rowvar=False))
    return variance, covariance


def align_crossfit_results(fold_results, test_indices, n_obs):
    """Arrange cross-fitted fold scores in original observation order."""
    if len(fold_results) != len(test_indices):
        raise ValueError("fold_results and test_indices must have equal length.")

    score_parts = []
    loading_parts = []
    index_parts = []
    for result, test_index in zip(fold_results, test_indices):
        score = np.asarray(result[0], dtype=float)
        loading = np.asarray(result[1], dtype=float)
        test_index = np.asarray(test_index, dtype=int).reshape(-1)
        if score.ndim == 0 or loading.ndim == 0:
            raise ValueError("Fold scores and loadings must include a sample axis.")
        if score.shape[0] != test_index.size or loading.shape[0] != test_index.size:
            raise ValueError(
                "Each fold result must contain one score and loading per test row."
            )
        score_parts.append(score)
        loading_parts.append(loading)
        index_parts.append(test_index)

    indices = np.concatenate(index_parts)
    if indices.size != n_obs or not np.array_equal(
        np.sort(indices), np.arange(n_obs)
    ):
        raise ValueError("Test indices must partition the observations exactly once.")

    order = np.argsort(indices)
    score = np.concatenate(score_parts, axis=0)[order]
    loading = np.concatenate(loading_parts, axis=0)[order]
    return score, loading


def summarize_ratio_scores(score, loading):
    """Estimate a ratio target and summarize its influence values.

    The function solves ``mean(H - a * theta) = 0`` and therefore computes
    ``theta = mean(H) / mean(a)``.  Its estimated influence values are
    ``(H - a * theta) / mean(a)``; the denominator is required whenever the
    loading is not normalized to have empirical mean one.

    Parameters
    ----------
    score : array-like of shape (n_samples,) or (n_samples, n_targets)
        Uncentered score contributions ``H``.
    loading : array-like
        Parameter loadings ``a``, broadcastable to the shape of ``score``.
        Examples are one for an ordinary average, a normalized kernel
        loading for a localized target, a normalized group indicator for a
        subgroup target, or their product.

    Returns
    -------
    theta : ndarray of shape (n_targets,)
        Ratio estimates.
    variance : ndarray of shape (n_targets,)
        Sample variances of the estimated influence values.  These are not
        divided by the sample size.
    covariance : ndarray of shape (n_targets, n_targets)
        Sample covariance of the estimated influence values across targets.
        This is not divided by the sample size.
    """
    theta, influence = _ratio_estimate_and_influence(score, loading)
    variance, covariance = _summarize_influence(influence)
    return theta, variance, covariance


def summarize_repeated_ratio_scores(score_reps, loading_reps):
    """Summarize ratio scores averaged over repeated sample splits.

    Scores and loadings for every repetition must be arranged in the same
    observation order.  Each repetition is centered at its own ratio estimate;
    inference is then based on the observation-level average of those
    influence values.
    """
    if len(score_reps) == 0 or len(score_reps) != len(loading_reps):
        raise ValueError(
            "score_reps and loading_reps must have the same positive length."
        )

    theta_reps = []
    influence_reps = []
    for score, loading in zip(score_reps, loading_reps):
        theta, influence = _ratio_estimate_and_influence(score, loading)
        theta_reps.append(theta)
        influence_reps.append(influence)

    try:
        theta = np.mean(np.stack(theta_reps, axis=0), axis=0)
        influence = np.mean(np.stack(influence_reps, axis=0), axis=0)
    except ValueError as exc:
        raise ValueError(
            "All repetitions must have matching sample and target dimensions."
        ) from exc

    variance, covariance = _summarize_influence(influence)
    return theta, variance, covariance
