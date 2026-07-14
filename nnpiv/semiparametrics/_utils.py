"""Shared numerical utilities for semiparametric ratio moments.

The estimators in this package can be written as solutions to
``E[H - a * theta] = 0``, where ``H`` is an uncentered orthogonal-score
contribution and ``a`` is the loading that defines the target.  Ordinary
averages use ``a=1``; localized and subgroup targets use nonconstant
loadings.
"""

import numpy as np


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
    covariance : ndarray
        Sample covariance of the estimated influence values across targets.
        This is not divided by the sample size.
    """
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
    variance = np.var(influence, axis=0, ddof=1)
    covariance = np.cov(influence, rowvar=False)
    return theta, variance, covariance
