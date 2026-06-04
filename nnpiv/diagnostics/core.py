# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import PolynomialFeatures


def _as_2d_float(array, name):
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"`{name}` must be 1D or 2D, got shape={arr.shape!r}.")


def _validate_same_n(A, C, C_prime):
    n = A.shape[0]
    if C.shape[0] != n or C_prime.shape[0] != n:
        raise ValueError(
            "A, C, and C_prime must have the same number of rows. "
            f"Got n_A={n}, n_C={C.shape[0]}, n_C_prime={C_prime.shape[0]}."
        )
    return n


def _resolve_indices(mask, n, name):
    if mask is None:
        return np.arange(n, dtype=int)

    arr = np.asarray(mask).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"`{name}` cannot be empty.")

    if arr.dtype == bool:
        if arr.shape[0] != n:
            raise ValueError(f"`{name}` boolean mask must have length {n}.")
        idx = np.flatnonzero(arr)
    elif arr.shape[0] == n and np.all(np.isin(arr, [0, 1])):
        idx = np.flatnonzero(arr == 1)
    else:
        try:
            idx = arr.astype(int, copy=False)
        except Exception as exc:
            raise ValueError(f"`{name}` must be a mask or integer indices.") from exc
        if np.any(idx < 0) or np.any(idx >= n):
            raise ValueError(f"`{name}` index values must be in [0, {n-1}].")
        idx = np.unique(idx)

    if idx.size == 0:
        raise ValueError(f"`{name}` selects zero observations.")
    return idx


def _resolve_gamma(X, gamma):
    if isinstance(gamma, str):
        if gamma != "auto":
            raise ValueError("`gamma` must be numeric or 'auto'.")
        dists = euclidean_distances(X, X)
        positive = dists[dists > 0]
        median_dist = float(np.median(positive)) if positive.size > 0 else 1.0
        return 1.0 / (2.0 * median_dist)
    gamma_val = float(gamma)
    if not np.isfinite(gamma_val) or gamma_val <= 0:
        raise ValueError("`gamma` must be finite and > 0.")
    return gamma_val


def _build_feature_matrix(
    A,
    feature_map="rff",
    n_features=300,
    gamma="auto",
    poly_degree=3,
    poly_include_bias=False,
    random_state=123,
    feature_builder=None,
    feature_matrix=None,
):
    n = A.shape[0]
    if feature_matrix is not None:
        BA = _as_2d_float(feature_matrix, "feature_matrix")
        if BA.shape[0] != n:
            raise ValueError(
                "`feature_matrix` must have the same number of rows as A. "
                f"Got n_A={n}, n_feature_rows={BA.shape[0]}."
            )
        return BA, {"feature_map": "precomputed", "gamma": None, "n_features": BA.shape[1]}

    if feature_builder is not None:
        BA = _as_2d_float(feature_builder(A), "feature_builder(A)")
        if BA.shape[0] != n:
            raise ValueError(
                "`feature_builder(A)` must return one row per sample in A."
            )
        return BA, {"feature_map": "callable", "gamma": None, "n_features": BA.shape[1]}

    if callable(feature_map):
        BA = _as_2d_float(feature_map(A), "feature_map(A)")
        if BA.shape[0] != n:
            raise ValueError("Callable `feature_map` must return one row per sample in A.")
        return BA, {"feature_map": "callable", "gamma": None, "n_features": BA.shape[1]}

    if feature_map == "rff":
        n_features_i = int(n_features)
        if n_features_i < 1:
            raise ValueError("`n_features` must be >= 1 for feature_map='rff'.")
        gamma_resolved = _resolve_gamma(A, gamma)
        sampler = RBFSampler(
            gamma=gamma_resolved,
            n_components=n_features_i,
            random_state=random_state,
        )
        BA = sampler.fit_transform(A)
        return BA, {
            "feature_map": "rff",
            "gamma": gamma_resolved,
            "n_features": BA.shape[1],
        }

    if feature_map == "polynomial":
        poly = PolynomialFeatures(
            degree=int(poly_degree),
            include_bias=bool(poly_include_bias),
        )
        BA = poly.fit_transform(A)
        return BA, {
            "feature_map": "polynomial",
            "gamma": None,
            "n_features": BA.shape[1],
        }

    raise ValueError(
        "Unsupported `feature_map`. Use 'rff', 'polynomial', a callable, "
        "or provide `feature_matrix`/`feature_builder`."
    )


def _fit_conditional_mean_regression(X, Y, alpha):
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, Y)
    pred = np.asarray(model.predict(X))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred


def _symmetrize(M):
    return 0.5 * (M + M.T)


def _stabilizer_matrix(Sigma_i, p, eta_mode):
    mode = str(eta_mode).lower()
    if mode == "identity":
        return np.eye(p), "identity"
    if mode == "sigma_i":
        return _symmetrize(Sigma_i), "sigma_i"
    raise ValueError("`eta_mode` must be one of {'sigma_i', 'identity'}.")


def _safe_whitened_max_eigenvalue(Sigma_s, Sigma_t, Sigma_i, eta, eta_mode="sigma_i"):
    p = Sigma_s.shape[0]
    eta_val = float(eta)
    if eta_val <= 0:
        raise ValueError("`eta` must be > 0.")

    stabilizer, resolved_mode = _stabilizer_matrix(Sigma_i, p, eta_mode)
    Sigma_stab = _symmetrize(Sigma_s + eta_val * stabilizer)
    evals_stab, evecs_stab = np.linalg.eigh(Sigma_stab)
    evals_stab = np.clip(evals_stab, 1e-15, None)
    inv_sqrt = evecs_stab @ np.diag(1.0 / np.sqrt(evals_stab)) @ evecs_stab.T

    whitened = _symmetrize(inv_sqrt @ Sigma_t @ inv_sqrt)
    evals_w, evecs_w = np.linalg.eigh(whitened)
    kappa2 = float(np.max(evals_w))

    top_idx = int(np.argmax(evals_w))
    top_w = evecs_w[:, top_idx]
    theta_max = inv_sqrt @ top_w
    norm_stab = float(theta_max.T @ Sigma_stab @ theta_max)
    if norm_stab > 1e-15:
        theta_max = theta_max / np.sqrt(norm_stab)

    return kappa2, evals_stab, evals_w, theta_max, resolved_mode


def _compute_operator_grams(BA, C, C_prime, idx_s, idx_t, ridge_alpha):
    BA_s = BA[idx_s]
    BA_t = BA[idx_t]
    Cps = C_prime[idx_s]
    Cs = C[idx_t]

    m_s_hat = _fit_conditional_mean_regression(Cps, BA_s, alpha=ridge_alpha)
    m_t_hat = _fit_conditional_mean_regression(Cs, BA_t, alpha=ridge_alpha)

    Sigma_s = _symmetrize((m_s_hat.T @ m_s_hat) / float(m_s_hat.shape[0]))
    Sigma_t = _symmetrize((m_t_hat.T @ m_t_hat) / float(m_t_hat.shape[0]))
    Sigma_i = _symmetrize((BA.T @ BA) / float(BA.shape[0]))
    return m_s_hat, m_t_hat, Sigma_s, Sigma_t, Sigma_i


def _resolve_eta_grid(eta_grid, fallback_eta):
    if eta_grid is None:
        vals = [float(fallback_eta)]
    elif np.isscalar(eta_grid):
        vals = [float(eta_grid)]
    else:
        vals = [float(x) for x in eta_grid]

    if len(vals) == 0:
        raise ValueError("`eta_grid` must contain at least one value.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("`eta_grid` must contain finite values.")
    if not np.all(np.asarray(vals) > 0):
        raise ValueError("`eta_grid` values must all be > 0.")
    return vals


def _annotate_monotone_path(rows, value_key="kappa", tol=1e-12):
    if len(rows) == 0:
        return rows, 0

    out = list(rows)
    prev = -np.inf
    violations = 0
    for row in out:
        val = float(row[value_key])
        if np.isfinite(val) and (val + tol) < prev:
            violations += 1
        prev = max(prev, val) if np.isfinite(val) else prev

    cum = -np.inf
    cum_key = f"{value_key}_cummax"
    for row in out:
        val = float(row[value_key])
        cum = max(cum, val) if np.isfinite(val) else cum
        row[cum_key] = cum
    return out, violations


def _sieve_can_use_nested_rff(feature_map, sieve_grid, kwargs):
    if feature_map != "rff":
        return False
    if kwargs.get("feature_matrix", None) is not None:
        return False
    if kwargs.get("feature_builder", None) is not None:
        return False
    for value in sieve_grid:
        if isinstance(value, dict):
            return False
        try:
            ivalue = int(value)
        except Exception:
            return False
        if ivalue < 1:
            return False
    return True


def _project_error_to_feature_span(BA, e_g, projection_ridge):
    ridge = float(projection_ridge)
    if ridge <= 0:
        raise ValueError("`projection_ridge` must be > 0.")

    e = np.asarray(e_g, dtype=float)
    if e.ndim == 1:
        e = e.reshape(-1, 1)
    elif e.ndim == 2 and e.shape[1] == 1:
        pass
    else:
        raise ValueError("`e_g` must be a 1D vector or a 2D column vector.")

    if e.shape[0] != BA.shape[0]:
        raise ValueError(
            "`e_g` must have the same number of rows as A. "
            f"Got n_e={e.shape[0]}, n_A={BA.shape[0]}."
        )

    n = float(BA.shape[0])
    G = _symmetrize((BA.T @ BA) / n)
    rhs = (BA.T @ e) / n
    theta = np.linalg.solve(G + ridge * np.eye(G.shape[0]), rhs).reshape(-1)
    fitted_error = BA @ theta.reshape(-1, 1)
    return theta, fitted_error.reshape(-1)


def _is_pandas_dataframe(data):
    cls = data.__class__
    return (cls.__name__ == "DataFrame") and str(getattr(cls, "__module__", "")).startswith("pandas")


def _is_column_selector_list(selector):
    return (
        isinstance(selector, (list, tuple))
        and len(selector) > 0
        and all(isinstance(x, (str, int, np.integer)) for x in selector)
    )


def _extract_from_mapping(mapping, selector, name):
    if isinstance(selector, str):
        return _as_2d_float(mapping[selector], name)
    if _is_column_selector_list(selector):
        if any(not isinstance(col, str) for col in selector):
            raise ValueError(f"`{name}` selector must be str or list[str] for mapping data.")
        return np.column_stack([_as_2d_float(mapping[col], f"{name}[{col!r}]") for col in selector])
    raise ValueError(
        f"Unsupported `{name}` selector for mapping data. "
        "Use a column name, list of column names, callable, or pre-built array."
    )


def _extract_from_ndarray(array, selector, name):
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError("When `data` is array-like, it must be 2D.")

    if isinstance(selector, (int, np.integer)):
        return _as_2d_float(arr[:, [int(selector)]], name)
    if isinstance(selector, slice):
        return _as_2d_float(arr[:, selector], name)
    if _is_column_selector_list(selector):
        idx = [int(i) for i in selector]
        return _as_2d_float(arr[:, idx], name)

    raise ValueError(
        f"Unsupported `{name}` selector for array-like data. "
        "Use an int, slice, list[int], callable, or pre-built array."
    )


def _extract_from_dataframe(df, selector, name):
    if isinstance(selector, str):
        return _as_2d_float(df.loc[:, [selector]].to_numpy(), name)

    if _is_column_selector_list(selector):
        if all(isinstance(col, (str,)) for col in selector):
            return _as_2d_float(df.loc[:, list(selector)].to_numpy(), name)
        if all(isinstance(col, (int, np.integer)) for col in selector):
            return _as_2d_float(df.iloc[:, list(selector)].to_numpy(), name)
        raise ValueError(
            f"`{name}` selector list for DataFrame must be all strings or all integers."
        )

    if isinstance(selector, (int, np.integer)):
        return _as_2d_float(df.iloc[:, [int(selector)]].to_numpy(), name)
    if isinstance(selector, slice):
        return _as_2d_float(df.iloc[:, selector].to_numpy(), name)

    raise ValueError(
        f"Unsupported `{name}` selector for DataFrame data. "
        "Use str, list[str], int, list[int], slice, callable, or pre-built array."
    )


def _extract_block_matrix(data, selector, name):
    if callable(selector):
        return _as_2d_float(selector(data), name)

    # Pre-built matrix/vector path.
    if isinstance(selector, np.ndarray):
        return _as_2d_float(selector, name)
    if isinstance(selector, (list, tuple)) and not _is_column_selector_list(selector):
        return _as_2d_float(selector, name)

    if _is_pandas_dataframe(data):
        return _extract_from_dataframe(data, selector, name)

    if hasattr(data, "keys") and hasattr(data, "__getitem__"):
        return _extract_from_mapping(data, selector, name)

    return _extract_from_ndarray(data, selector, name)


def _extract_mask(data, selector, name):
    if selector is None:
        return None
    if callable(selector):
        return np.asarray(selector(data)).reshape(-1)
    if isinstance(selector, np.ndarray):
        return np.asarray(selector).reshape(-1)
    if isinstance(selector, (list, tuple)):
        arr = np.asarray(selector)
        if arr.dtype == bool or np.all(np.isin(arr, [0, 1])):
            return arr.reshape(-1)
        # Treat list[int] as direct index vector.
        if np.issubdtype(arr.dtype, np.integer):
            return arr.reshape(-1)

    if _is_pandas_dataframe(data):
        if isinstance(selector, str):
            return np.asarray(data.loc[:, selector]).reshape(-1)
        raise ValueError(f"`{name}` for DataFrame data must be str, array-like, or callable.")

    if hasattr(data, "keys") and hasattr(data, "__getitem__"):
        if isinstance(selector, str):
            return np.asarray(data[selector]).reshape(-1)
        raise ValueError(f"`{name}` for mapping data must be str, array-like, or callable.")

    # For ndarray-style data, selector must already be mask/index-like.
    if isinstance(selector, (str,)):
        raise ValueError(f"`{name}` cannot be a string when `data` is array-like.")
    return np.asarray(selector).reshape(-1)


def relative_wellposedness_diagnostic(
    A,
    C,
    C_prime,
    *,
    feature_map="rff",
    n_features=300,
    gamma="auto",
    poly_degree=3,
    poly_include_bias=False,
    ridge_alpha=1.0,
    eta=1e-6,
    eta_mode="sigma_i",
    null_eig_atol=1e-10,
    null_eig_rtol=1e-8,
    null_leakage_tol=1e-10,
    random_state=123,
    feature_builder=None,
    feature_matrix=None,
    mask_s=None,
    mask_t=None,
    return_top_direction=False,
    return_details=False,
):
    r"""
    Compute the finite-dimensional relative well-posedness diagnostic ``kappa``.

    Parameters:
    A : array-like of shape (n,) or (n, d_a)
        First-stage endogenous block used to build featureized functions
        :math:`b(A)`.
    C : array-like of shape (n,) or (n, d_c)
        Second-stage instrument block for estimating
        :math:`m_T(C)=E[b(A)\\mid C]`.
    C_prime : array-like of shape (n,) or (n, d_cp)
        First-stage instrument block for estimating
        :math:`m_S(C')=E[b(A)\\mid C']`.
    feature_map : {'rff', 'polynomial'} or callable, default='rff'
        Feature construction for :math:`b(A)`. A callable must return one row
        per sample in ``A``.
    n_features : int, default=300
        Number of random Fourier features when ``feature_map='rff'``.
    gamma : float or 'auto', default='auto'
        RBF bandwidth parameter used by random Fourier features.
    poly_degree : int, default=3
        Polynomial degree when ``feature_map='polynomial'``.
    poly_include_bias : bool, default=False
        Whether to include the bias term in polynomial features.
    ridge_alpha : float, default=1.0
        Ridge penalty used in the conditional-mean regressions.
    eta : float, default=1e-6
        Stabilization magnitude in ``Sigma_S + eta * R``. Must be positive.
    eta_mode : {'sigma_i', 'identity'}, default='sigma_i'
        Stabilizer choice. ``'sigma_i'`` uses the empirical feature second
        moment ``Sigma_I``; ``'identity'`` uses ``I``.
    null_eig_atol : float, default=1e-10
        Absolute tolerance for identifying near-null eigendirections of
        ``Sigma_s``.
    null_eig_rtol : float, default=1e-8
        Relative tolerance (scaled by ``max_eig_sigma_s``) for near-null
        eigendirections.
    null_leakage_tol : float, default=1e-10
        Threshold used to flag nullspace leakage from ``Sigma_t`` into the
        near-null eigenspace of ``Sigma_s``.
    random_state : int, default=123
        Random seed used by random Fourier feature generation.
    feature_builder : callable, optional
        Legacy alias for callable feature construction. If provided, called as
        ``feature_builder(A)``.
    feature_matrix : array-like of shape (n, p), optional
        Precomputed feature matrix for ``A``. If provided, bypasses
        ``feature_map`` construction.
    mask_s : array-like, optional
        Optional subset selector for the ``S`` side. Can be a boolean mask,
        0/1 mask, or integer index vector.
    mask_t : array-like, optional
        Optional subset selector for the ``T`` side. Same accepted formats as
        ``mask_s``.
    return_top_direction : bool, default=False
        If ``True``, include the maximizing generalized-eigenvector direction
        and its induced norms in the output.
    return_details : bool, default=False
        If ``True``, include intermediate matrices and fitted conditional means
        in the output.

    Returns
    -------
    dict
        Dictionary with diagnostic scalars including ``kappa2``, ``kappa``,
        stability flags, and metadata. Optional keys are added when
        ``return_top_direction`` and/or ``return_details`` are enabled.

    Notes
    -----
    The core target is

    .. math::

       \kappa_{J,\eta}^2
       =
       \lambda_{\max}\!\left[
       (\Sigma_S + \eta R)^{-1/2}\Sigma_T(\Sigma_S + \eta R)^{-1/2}
       \right],

    where :math:`R` is either ``I`` or :math:`\Sigma_I`.
    """
    A = _as_2d_float(A, "A")
    C = _as_2d_float(C, "C")
    C_prime = _as_2d_float(C_prime, "C_prime")
    n = _validate_same_n(A, C, C_prime)

    idx_s = _resolve_indices(mask_s, n, "mask_s")
    idx_t = _resolve_indices(mask_t, n, "mask_t")

    BA, feature_meta = _build_feature_matrix(
        A,
        feature_map=feature_map,
        n_features=n_features,
        gamma=gamma,
        poly_degree=poly_degree,
        poly_include_bias=poly_include_bias,
        random_state=random_state,
        feature_builder=feature_builder,
        feature_matrix=feature_matrix,
    )

    m_s_hat, m_t_hat, Sigma_s, Sigma_t, Sigma_i = _compute_operator_grams(
        BA, C, C_prime, idx_s, idx_t, ridge_alpha=ridge_alpha
    )

    kappa2, evals_sigma_s_eta, evals_whitened, theta_max, eta_mode_resolved = _safe_whitened_max_eigenvalue(
        Sigma_s,
        Sigma_t,
        Sigma_i,
        eta,
        eta_mode=eta_mode,
    )
    kappa = float(np.sqrt(max(kappa2, 0.0)))

    eig_s = np.linalg.eigvalsh(Sigma_s)
    eig_t = np.linalg.eigvalsh(Sigma_t)
    max_eig_s = float(np.max(eig_s))
    min_eig_s = float(np.min(eig_s))
    max_eig_t = float(np.max(eig_t))

    if max_eig_s <= 0:
        null_thresh = float(max(null_eig_atol, 0.0))
    else:
        null_thresh = float(max(null_eig_atol, null_eig_rtol * max_eig_s))

    null_mask = eig_s <= null_thresh
    null_like_dim = int(np.sum(null_mask))
    nullspace_leakage = 0.0
    nullspace_violation_flag = False
    if null_like_dim > 0:
        _, U = np.linalg.eigh(Sigma_s)
        U_null = U[:, null_mask]
        T_on_null = _symmetrize(U_null.T @ Sigma_t @ U_null)
        leak_eigs = np.linalg.eigvalsh(T_on_null)
        nullspace_leakage = float(np.max(leak_eigs))
        nullspace_violation_flag = bool(nullspace_leakage > float(null_leakage_tol))

    stabilizer, _ = _stabilizer_matrix(Sigma_i, Sigma_s.shape[0], eta_mode_resolved)
    stab_eigs = np.linalg.eigvalsh(_symmetrize(float(eta) * stabilizer))
    max_stab_eig = float(np.max(stab_eigs))
    if max_eig_s > 0:
        stabilization_dominance_ratio = float(max_stab_eig / max_eig_s)
    else:
        stabilization_dominance_ratio = np.inf if max_stab_eig > 0 else 0.0

    diag_ratio = np.diag(Sigma_t) / np.maximum(np.diag(Sigma_s), 1e-15)
    max_diag_ratio = float(np.max(diag_ratio))
    unstable = bool(
        (kappa2 > 1e6)
        or (max_diag_ratio > 1e6)
        or nullspace_violation_flag
    )

    out = {
        "kappa2": kappa2,
        "kappa": kappa,
        "eta": float(eta),
        "eta_mode": eta_mode_resolved,
        "ridge_alpha": float(ridge_alpha),
        "n_total": int(n),
        "n_s": int(idx_s.shape[0]),
        "n_t": int(idx_t.shape[0]),
        "n_features": int(BA.shape[1]),
        "feature_meta": feature_meta,
        "null_like_dim_sigma_s": null_like_dim,
        "null_eig_threshold_sigma_s": null_thresh,
        "min_eig_sigma_s": min_eig_s,
        "max_eig_sigma_s": max_eig_s,
        "max_eig_sigma_t": max_eig_t,
        "nullspace_leakage_sigma_t_on_null_sigma_s": nullspace_leakage,
        "nullspace_violation_flag": nullspace_violation_flag,
        "null_leakage_tol": float(null_leakage_tol),
        "max_diag_ratio_sigma_t_over_sigma_s": max_diag_ratio,
        "stabilization_dominance_ratio": stabilization_dominance_ratio,
        "unstable_flag": unstable,
        "min_eig_sigma_s_eta": float(np.min(evals_sigma_s_eta)),
        "max_eig_whitened": float(np.max(evals_whitened)),
        "trace_sigma_i": float(np.trace(Sigma_i)),
    }

    if return_top_direction:
        theta = np.asarray(theta_max).reshape(-1)
        s_sq = float(theta.T @ Sigma_s @ theta)
        t_sq = float(theta.T @ Sigma_t @ theta)
        i_sq = float(theta.T @ Sigma_i @ theta)
        out["theta_max"] = theta
        out["u_max_norm_s"] = float(np.sqrt(max(s_sq, 0.0)))
        out["u_max_norm_t"] = float(np.sqrt(max(t_sq, 0.0)))
        out["u_max_norm_i"] = float(np.sqrt(max(i_sq, 0.0)))

    if return_details:
        out["Sigma_s"] = Sigma_s
        out["Sigma_t"] = Sigma_t
        out["Sigma_i"] = Sigma_i
        out["m_s_hat"] = m_s_hat
        out["m_t_hat"] = m_t_hat
        out["feature_matrix"] = BA
        out["indices_s"] = idx_s
        out["indices_t"] = idx_t

    return out


def _default_sieve_grid(feature_map):
    if feature_map == "polynomial":
        return [1, 2, 3, 4]
    if feature_map == "rff":
        return [50, 100, 200, 400]
    raise ValueError(
        "`sieve_grid` must be provided when feature_map is callable, "
        "precomputed, or not one of {'rff', 'polynomial'}."
    )


def _sieve_params_from_value(value, feature_map):
    if isinstance(value, dict):
        return dict(value), value.get("sieve_value", None)
    if feature_map == "polynomial":
        return {"poly_degree": int(value)}, int(value)
    return {"n_features": int(value)}, int(value)


def _summarize_sieve_rows(rows, stability_growth_tol):
    kappas = np.asarray([row["kappa"] for row in rows], dtype=float)
    finite = kappas[np.isfinite(kappas)]
    nullspace_violations = int(sum(bool(row.get("nullspace_violation_flag", False)) for row in rows))
    if finite.size == 0:
        return {
            "n_points": int(len(rows)),
            "max_kappa": np.nan,
            "kappa_growth_ratio": np.nan,
            "stable_flag": False,
            "any_unstable_flag": True,
            "nullspace_violation_count": nullspace_violations,
        }

    first = finite[0]
    last = finite[-1]
    if first <= 0:
        growth = np.inf if last > 0 else 1.0
    else:
        growth = float(last / first)

    any_unstable = bool(any(row.get("unstable_flag", False) for row in rows))
    stable = bool(np.all(np.isfinite(kappas)) and (growth <= stability_growth_tol) and not any_unstable)
    return {
        "n_points": int(len(rows)),
        "max_kappa": float(np.max(finite)),
        "kappa_growth_ratio": growth,
        "stable_flag": stable,
        "any_unstable_flag": any_unstable,
        "nullspace_violation_count": nullspace_violations,
    }


def relative_wellposedness_sieve_diagnostic(
    A,
    C,
    C_prime,
    *,
    sieve_grid=None,
    eta_grid=None,
    enforce_nested_rff=True,
    stability_growth_tol=5.0,
    **kwargs,
):
    r"""
    Run the pre-estimation diagnostic over a sieve and eta path.

    Parameters:
    A, C, C_prime : array-like
        Inputs passed to :func:`relative_wellposedness_diagnostic`. Arrays must
        be row-aligned with common sample size ``n``.
    sieve_grid : iterable, optional
        Sieve values. For ``feature_map='rff'``, entries map to ``n_features``.
        For ``feature_map='polynomial'``, entries map to ``poly_degree``.
        Entries may also be dictionaries of per-call keyword overrides.
        If ``None``, defaults are chosen by feature map.
    eta_grid : float or iterable of float, optional
        Stabilization values. If ``None``, uses ``eta`` from ``kwargs`` (or
        ``1e-6``).
    enforce_nested_rff : bool, default=True
        If ``True`` and applicable, compute one large RFF map and reuse nested
        prefixes across sieve values for speed and consistency.
    stability_growth_tol : float, default=5.0
        Growth threshold used in the summary ``stable_flag`` calculation.
    **kwargs
        Additional arguments forwarded to
        :func:`relative_wellposedness_diagnostic`.

    Returns
    -------
    dict
        Dictionary with two keys:

        - ``rows``: list of row-level diagnostics for each ``(eta, sieve)``
          pair.
        - ``summary``: aggregate stability metrics and path metadata.

    Notes
    -----
    This function is a path wrapper around
    :func:`relative_wellposedness_diagnostic`; it does not alter the underlying
    diagnostic definition.
    """
    feature_map = kwargs.get("feature_map", "rff")
    grid = _default_sieve_grid(feature_map) if sieve_grid is None else list(sieve_grid)
    if len(grid) == 0:
        raise ValueError("`sieve_grid` must contain at least one entry.")
    eta_vals = _resolve_eta_grid(eta_grid, kwargs.get("eta", 1e-6))

    rows = []
    base_kwargs = dict(kwargs)
    base_kwargs.pop("eta", None)
    nested_rff_used = False

    can_nested_rff = bool(enforce_nested_rff) and _sieve_can_use_nested_rff(feature_map, grid, base_kwargs)
    precomputed_nested = None
    precomputed_meta = None
    if can_nested_rff:
        max_features = max(int(v) for v in grid)
        A2 = _as_2d_float(A, "A")
        precomputed_nested, precomputed_meta = _build_feature_matrix(
            A2,
            feature_map="rff",
            n_features=max_features,
            gamma=base_kwargs.get("gamma", "auto"),
            random_state=base_kwargs.get("random_state", 123),
        )
        nested_rff_used = True

    for eta_i, eta_val in enumerate(eta_vals):
        for i, value in enumerate(grid):
            params, sieve_value = _sieve_params_from_value(value, feature_map)
            call_kwargs = dict(base_kwargs)
            call_kwargs["eta"] = float(eta_val)

            if precomputed_nested is not None:
                n_feat = int(value)
                call_kwargs["feature_matrix"] = precomputed_nested[:, :n_feat]
                call_kwargs.pop("feature_builder", None)
                call_kwargs["feature_map"] = feature_map
                call_kwargs["n_features"] = n_feat
            else:
                call_kwargs.update(params)

            diag = relative_wellposedness_diagnostic(A, C, C_prime, **call_kwargs)
            row = dict(diag)
            feature_meta = row.pop("feature_meta", {})
            row_feature_map = feature_meta.get("feature_map", feature_map)
            row_feature_gamma = feature_meta.get("gamma", None)
            if precomputed_nested is not None and precomputed_meta is not None:
                row_feature_map = feature_map
                row_feature_gamma = precomputed_meta.get("gamma", None)
            row.update({
                "eta_index": int(eta_i),
                "eta": float(eta_val),
                "sieve_index": int(i),
                "sieve_value": sieve_value if sieve_value is not None else i,
                "feature_map": row_feature_map,
                "feature_gamma": row_feature_gamma,
            })
            rows.append(row)

    per_eta = []
    rows_annotated = []
    total_violations = 0
    for eta_i, eta_val in enumerate(eta_vals):
        eta_rows = [r for r in rows if r["eta_index"] == eta_i]
        eta_rows = sorted(eta_rows, key=lambda r: r["sieve_index"])
        eta_rows, violations = _annotate_monotone_path(eta_rows, value_key="kappa")
        total_violations += int(violations)
        eta_summary = _summarize_sieve_rows(eta_rows, stability_growth_tol=float(stability_growth_tol))
        eta_summary["eta"] = float(eta_val)
        eta_summary["monotone_violations"] = int(violations)
        per_eta.append(eta_summary)
        rows_annotated.extend(eta_rows)

    rows = sorted(rows_annotated, key=lambda r: (r["eta_index"], r["sieve_index"]))
    if len(eta_vals) == 1:
        summary = dict(per_eta[0])
        summary["n_eta"] = 1
    else:
        finite = np.asarray([r["kappa"] for r in rows], dtype=float)
        finite = finite[np.isfinite(finite)]
        summary = {
            "n_points": int(len(rows)),
            "n_sieve": int(len(grid)),
            "n_eta": int(len(eta_vals)),
            "max_kappa": float(np.max(finite)) if finite.size else np.nan,
            "kappa_growth_ratio": np.nan,
            "stable_flag": bool(all(x["stable_flag"] for x in per_eta)),
            "any_unstable_flag": bool(any(x["any_unstable_flag"] for x in per_eta)),
            "nullspace_violation_count": int(sum(int(x.get("nullspace_violation_count", 0)) for x in per_eta)),
            "per_eta": per_eta,
            "monotone_violations": int(total_violations),
        }
    summary["stability_growth_tol"] = float(stability_growth_tol)
    summary["eta_grid"] = [float(x) for x in eta_vals]
    summary["nested_rff_used"] = bool(nested_rff_used)
    return {
        "rows": rows,
        "summary": summary,
    }


def relative_wellposedness_effective_diagnostic(
    A,
    C,
    C_prime,
    e_g,
    *,
    feature_map="rff",
    n_features=300,
    gamma="auto",
    poly_degree=3,
    poly_include_bias=False,
    ridge_alpha=1.0,
    projection_ridge=1e-8,
    eta=1e-6,
    eta_mode="sigma_i",
    random_state=123,
    feature_builder=None,
    feature_matrix=None,
    mask_s=None,
    mask_t=None,
    return_details=False,
):
    r"""
    Compute the post-estimation effective-direction diagnostic ``kappa_eff``.

    Parameters:
    A : array-like of shape (n,) or (n, d_a)
        First-stage endogenous block used to construct feature functions.
    C : array-like of shape (n,) or (n, d_c)
        Second-stage instrument block.
    C_prime : array-like of shape (n,) or (n, d_cp)
        First-stage instrument block.
    e_g : array-like of shape (n,) or (n, 1)
        Estimated first-stage error direction, typically
        :math:`\widehat g - g_0`.
    feature_map, n_features, gamma, poly_degree, poly_include_bias, ridge_alpha, eta, eta_mode, random_state, feature_builder, feature_matrix
        Same interpretation as in :func:`relative_wellposedness_diagnostic`.
    projection_ridge : float, default=1e-8
        Ridge used when projecting ``e_g`` into the feature span.
    mask_s, mask_t : array-like, optional
        Optional stage-specific subset selectors, using the same mask/index
        formats accepted by :func:`relative_wellposedness_diagnostic`.
    return_details : bool, default=False
        If ``True``, include projected coefficients and intermediate matrices.

    Returns
    -------
    dict
        Dictionary containing ``kappa_eff``, regularized counterpart
        ``kappa_eff_reg``, associated norms, and metadata.

    Notes
    -----
    The core quantity is

    .. math::

       \kappa_{\mathrm{eff}}
       =
       \frac{\|T_g e_g\|_2}{\|S e_g\|_2},

    computed after projecting ``e_g`` onto the selected finite feature span.
    """
    A = _as_2d_float(A, "A")
    C = _as_2d_float(C, "C")
    C_prime = _as_2d_float(C_prime, "C_prime")
    n = _validate_same_n(A, C, C_prime)

    idx_s = _resolve_indices(mask_s, n, "mask_s")
    idx_t = _resolve_indices(mask_t, n, "mask_t")

    BA, feature_meta = _build_feature_matrix(
        A,
        feature_map=feature_map,
        n_features=n_features,
        gamma=gamma,
        poly_degree=poly_degree,
        poly_include_bias=poly_include_bias,
        random_state=random_state,
        feature_builder=feature_builder,
        feature_matrix=feature_matrix,
    )
    m_s_hat, m_t_hat, Sigma_s, Sigma_t, Sigma_i = _compute_operator_grams(
        BA, C, C_prime, idx_s, idx_t, ridge_alpha=ridge_alpha
    )
    theta_err, fitted_error = _project_error_to_feature_span(BA, e_g, projection_ridge)

    num_sq = float(theta_err.T @ Sigma_t @ theta_err)
    den_sq = float(theta_err.T @ Sigma_s @ theta_err)
    p = Sigma_s.shape[0]
    stabilizer, eta_mode_resolved = _stabilizer_matrix(Sigma_i, p, eta_mode)
    den_reg_sq = float(theta_err.T @ (Sigma_s + float(eta) * stabilizer) @ theta_err)

    num_sq = max(num_sq, 0.0)
    den_sq = max(den_sq, 0.0)
    den_reg_sq = max(den_reg_sq, 0.0)

    if den_sq <= 1e-15:
        kappa_eff = np.inf if num_sq > 1e-15 else 0.0
        near_null = True
    else:
        kappa_eff = float(np.sqrt(num_sq / den_sq))
        near_null = False

    if den_reg_sq <= 1e-15:
        kappa_eff_reg = np.inf if num_sq > 1e-15 else 0.0
    else:
        kappa_eff_reg = float(np.sqrt(num_sq / den_reg_sq))

    out = {
        "kappa_eff": kappa_eff,
        "kappa_eff_reg": kappa_eff_reg,
        "t_norm": float(np.sqrt(num_sq)),
        "s_norm": float(np.sqrt(den_sq)),
        "s_norm_reg": float(np.sqrt(den_reg_sq)),
        "eta": float(eta),
        "eta_mode": eta_mode_resolved,
        "projection_ridge": float(projection_ridge),
        "ridge_alpha": float(ridge_alpha),
        "near_null_s_direction": bool(near_null),
        "n_total": int(n),
        "n_s": int(idx_s.shape[0]),
        "n_t": int(idx_t.shape[0]),
        "n_features": int(BA.shape[1]),
        "feature_meta": feature_meta,
    }

    if return_details:
        out["theta_error"] = theta_err
        out["fitted_error"] = fitted_error
        out["feature_matrix"] = BA
        out["Sigma_s"] = Sigma_s
        out["Sigma_t"] = Sigma_t
        out["Sigma_i"] = Sigma_i
        out["m_s_hat"] = m_s_hat
        out["m_t_hat"] = m_t_hat
        out["indices_s"] = idx_s
        out["indices_t"] = idx_t
    return out


def relative_wellposedness_effective_sieve_diagnostic(
    A,
    C,
    C_prime,
    e_g,
    *,
    sieve_grid=None,
    eta_grid=None,
    enforce_nested_rff=True,
    **kwargs,
):
    """
    Run ``kappa_eff`` over a sieve and eta path.

    Parameters:
    A, C, C_prime : array-like
        Inputs passed to :func:`relative_wellposedness_effective_diagnostic`.
        Arrays must share the same number of rows.
    e_g : array-like of shape (n,) or (n, 1)
        Estimated first-stage error direction.
    sieve_grid : iterable, optional
        Sieve values interpreted as in
        :func:`relative_wellposedness_sieve_diagnostic`.
    eta_grid : float or iterable of float, optional
        Stabilization values for path evaluation.
    enforce_nested_rff : bool, default=True
        Reuse nested RFF prefixes when possible.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`relative_wellposedness_effective_diagnostic`.

    Returns
    -------
    dict
        Dictionary with ``rows`` (path-level outputs) and ``summary``
        (aggregated metadata and maxima).

    Notes
    -----
    This is the post-estimation analogue of
    :func:`relative_wellposedness_sieve_diagnostic`.
    """
    feature_map = kwargs.get("feature_map", "rff")
    grid = _default_sieve_grid(feature_map) if sieve_grid is None else list(sieve_grid)
    if len(grid) == 0:
        raise ValueError("`sieve_grid` must contain at least one entry.")
    eta_vals = _resolve_eta_grid(eta_grid, kwargs.get("eta", 1e-6))

    rows = []
    base_kwargs = dict(kwargs)
    base_kwargs.pop("eta", None)
    nested_rff_used = False

    can_nested_rff = bool(enforce_nested_rff) and _sieve_can_use_nested_rff(feature_map, grid, base_kwargs)
    precomputed_nested = None
    precomputed_meta = None
    if can_nested_rff:
        max_features = max(int(v) for v in grid)
        A2 = _as_2d_float(A, "A")
        precomputed_nested, precomputed_meta = _build_feature_matrix(
            A2,
            feature_map="rff",
            n_features=max_features,
            gamma=base_kwargs.get("gamma", "auto"),
            random_state=base_kwargs.get("random_state", 123),
        )
        nested_rff_used = True

    for eta_i, eta_val in enumerate(eta_vals):
        for i, value in enumerate(grid):
            params, sieve_value = _sieve_params_from_value(value, feature_map)
            call_kwargs = dict(base_kwargs)
            call_kwargs["eta"] = float(eta_val)

            if precomputed_nested is not None:
                n_feat = int(value)
                call_kwargs["feature_matrix"] = precomputed_nested[:, :n_feat]
                call_kwargs.pop("feature_builder", None)
                call_kwargs["feature_map"] = feature_map
                call_kwargs["n_features"] = n_feat
            else:
                call_kwargs.update(params)

            diag = relative_wellposedness_effective_diagnostic(
                A, C, C_prime, e_g, **call_kwargs
            )
            row = dict(diag)
            feature_meta = row.pop("feature_meta", {})
            row_feature_map = feature_meta.get("feature_map", feature_map)
            row_feature_gamma = feature_meta.get("gamma", None)
            if precomputed_nested is not None and precomputed_meta is not None:
                row_feature_map = feature_map
                row_feature_gamma = precomputed_meta.get("gamma", None)
            row.update({
                "eta_index": int(eta_i),
                "eta": float(eta_val),
                "sieve_index": int(i),
                "sieve_value": sieve_value if sieve_value is not None else i,
                "feature_map": row_feature_map,
                "feature_gamma": row_feature_gamma,
            })
            rows.append(row)

    rows_annotated = []
    per_eta = []
    for eta_i, eta_val in enumerate(eta_vals):
        eta_rows = [r for r in rows if r["eta_index"] == eta_i]
        eta_rows = sorted(eta_rows, key=lambda r: r["sieve_index"])
        eta_rows, violations = _annotate_monotone_path(eta_rows, value_key="kappa_eff")
        eta_summary = {
            "eta": float(eta_val),
            "max_kappa_eff": float(
                np.nanmax(np.asarray([r["kappa_eff"] for r in eta_rows], dtype=float))
            ),
            "monotone_violations": int(violations),
        }
        per_eta.append(eta_summary)
        rows_annotated.extend(eta_rows)

    rows = sorted(rows_annotated, key=lambda r: (r["eta_index"], r["sieve_index"]))
    summary = {
        "n_points": int(len(rows)),
        "n_sieve": int(len(grid)),
        "n_eta": int(len(eta_vals)),
        "eta_grid": [float(x) for x in eta_vals],
        "nested_rff_used": bool(nested_rff_used),
        "per_eta": per_eta,
    }
    if len(eta_vals) == 1:
        summary.update(per_eta[0])
    return {
        "rows": rows,
        "summary": summary,
    }


def relative_wellposedness_from_data(
    data,
    *,
    A,
    B=None,
    C,
    C_prime,
    mask_s=None,
    mask_t=None,
    **kwargs,
):
    """
    Run the pre-estimation diagnostic from dataset-level selectors.

    Parameters:
    data : DataFrame, mapping, or 2D array-like
        Source container holding all blocks.
    A, C, C_prime
        Block selectors for each argument required by
        :func:`relative_wellposedness_diagnostic`.
    B
        Optional selector accepted only for interface consistency with
        ``(A, B, C, C')`` notation. It is intentionally ignored by this
        diagnostic.

        Supported selector forms:
        - callable ``selector(data) -> array-like``
        - direct array-like matrix/vector
        - DataFrame/mapping columns (string or list of strings)
        - DataFrame/array-style integer selectors (int, list[int], slice)
    mask_s, mask_t : array-like, str, or callable, optional
        Optional subset selectors for stage-specific sample restrictions.
    **kwargs
        Forwarded to :func:`relative_wellposedness_diagnostic`.

    Returns
    -------
    dict
        Output dictionary returned by
        :func:`relative_wellposedness_diagnostic`.

    Notes
    -----
    ``A``, ``C``, and ``C_prime`` must resolve to arrays with the same row
    count.
    """
    _ = B  # accepted for block-interface consistency; not used by Diagnostic A

    A_mat = _extract_block_matrix(data, A, "A")
    C_mat = _extract_block_matrix(data, C, "C")
    Cp_mat = _extract_block_matrix(data, C_prime, "C_prime")
    mask_s_arr = _extract_mask(data, mask_s, "mask_s")
    mask_t_arr = _extract_mask(data, mask_t, "mask_t")

    return relative_wellposedness_diagnostic(
        A=A_mat,
        C=C_mat,
        C_prime=Cp_mat,
        mask_s=mask_s_arr,
        mask_t=mask_t_arr,
        **kwargs,
    )


def relative_wellposedness_effective_from_data(
    data,
    *,
    e_g,
    A,
    B=None,
    C,
    C_prime,
    mask_s=None,
    mask_t=None,
    **kwargs,
):
    """
    Run the post-estimation ``kappa_eff`` diagnostic from dataset selectors.

    Parameters:
    data : DataFrame, mapping, or 2D array-like
        Source container holding all blocks.
    e_g
        Selector for the first-stage error direction block.
    A, C, C_prime
        Selectors resolved and passed to
        :func:`relative_wellposedness_effective_diagnostic`.
    B
        Optional selector accepted for interface consistency with
        ``(A, B, C, C')`` notation; ignored by this diagnostic.
    mask_s, mask_t : array-like, str, or callable, optional
        Optional subset selectors for stage-specific sample restrictions.
    **kwargs
        Forwarded to :func:`relative_wellposedness_effective_diagnostic`.

    Returns
    -------
    dict
        Output dictionary returned by
        :func:`relative_wellposedness_effective_diagnostic`.

    Notes
    -----
    Resolved ``A``, ``C``, ``C_prime``, and ``e_g`` must be row-aligned.
    """
    _ = B
    A_mat = _extract_block_matrix(data, A, "A")
    C_mat = _extract_block_matrix(data, C, "C")
    Cp_mat = _extract_block_matrix(data, C_prime, "C_prime")
    e_vec = _extract_block_matrix(data, e_g, "e_g")
    mask_s_arr = _extract_mask(data, mask_s, "mask_s")
    mask_t_arr = _extract_mask(data, mask_t, "mask_t")
    return relative_wellposedness_effective_diagnostic(
        A=A_mat,
        C=C_mat,
        C_prime=Cp_mat,
        e_g=e_vec,
        mask_s=mask_s_arr,
        mask_t=mask_t_arr,
        **kwargs,
    )


def relative_wellposedness_sieve_from_data(
    data,
    *,
    A,
    B=None,
    C,
    C_prime,
    mask_s=None,
    mask_t=None,
    **kwargs,
):
    """
    Run the sieve-path pre-estimation diagnostic from dataset selectors.

    Parameters:
    data : DataFrame, mapping, or 2D array-like
        Source container holding all blocks.
    A, C, C_prime
        Selectors resolved and passed to
        :func:`relative_wellposedness_sieve_diagnostic`.
    B
        Optional selector accepted for interface consistency with
        ``(A, B, C, C')`` notation; ignored by this diagnostic.
    mask_s, mask_t : array-like, str, or callable, optional
        Optional subset selectors for stage-specific sample restrictions.
    **kwargs
        Forwarded to :func:`relative_wellposedness_sieve_diagnostic`.

    Returns
    -------
    dict
        Output dictionary returned by
        :func:`relative_wellposedness_sieve_diagnostic`.

    Notes
    -----
    Selector semantics match :func:`relative_wellposedness_from_data`.
    """
    _ = B

    A_mat = _extract_block_matrix(data, A, "A")
    C_mat = _extract_block_matrix(data, C, "C")
    Cp_mat = _extract_block_matrix(data, C_prime, "C_prime")
    mask_s_arr = _extract_mask(data, mask_s, "mask_s")
    mask_t_arr = _extract_mask(data, mask_t, "mask_t")

    return relative_wellposedness_sieve_diagnostic(
        A=A_mat,
        C=C_mat,
        C_prime=Cp_mat,
        mask_s=mask_s_arr,
        mask_t=mask_t_arr,
        **kwargs,
    )


def relative_wellposedness_effective_sieve_from_data(
    data,
    *,
    e_g,
    A,
    B=None,
    C,
    C_prime,
    mask_s=None,
    mask_t=None,
    **kwargs,
):
    """
    Run the sieve-path post-estimation ``kappa_eff`` diagnostic from selectors.

    Parameters:
    data : DataFrame, mapping, or 2D array-like
        Source container holding all blocks.
    e_g
        Selector for the first-stage error direction block.
    A, C, C_prime
        Selectors resolved and passed to
        :func:`relative_wellposedness_effective_sieve_diagnostic`.
    B
        Optional selector accepted for interface consistency with
        ``(A, B, C, C')`` notation; ignored by this diagnostic.
    mask_s, mask_t : array-like, str, or callable, optional
        Optional subset selectors for stage-specific sample restrictions.
    **kwargs
        Forwarded to :func:`relative_wellposedness_effective_sieve_diagnostic`.

    Returns
    -------
    dict
        Output dictionary returned by
        :func:`relative_wellposedness_effective_sieve_diagnostic`.

    Notes
    -----
    Selector semantics match :func:`relative_wellposedness_from_data`.
    """
    _ = B
    A_mat = _extract_block_matrix(data, A, "A")
    C_mat = _extract_block_matrix(data, C, "C")
    Cp_mat = _extract_block_matrix(data, C_prime, "C_prime")
    e_vec = _extract_block_matrix(data, e_g, "e_g")
    mask_s_arr = _extract_mask(data, mask_s, "mask_s")
    mask_t_arr = _extract_mask(data, mask_t, "mask_t")
    return relative_wellposedness_effective_sieve_diagnostic(
        A=A_mat,
        C=C_mat,
        C_prime=Cp_mat,
        e_g=e_vec,
        mask_s=mask_s_arr,
        mask_t=mask_t_arr,
        **kwargs,
    )


def relative_wellposedness_from_nested_npiv(
    A,
    D,
    B,
    C,
    *,
    mask_s=None,
    mask_t=None,
    **kwargs,
):
    """
    Run the pre-estimation diagnostic for canonical nested NPIV blocks.

    Parameters:
    A, D, B, C
        Blocks from nested NPIV data generation where ``A`` is first-stage
        endogenous treatment, ``D`` first-stage instrument, ``B`` second-stage
        endogenous treatment, and ``C`` second-stage instrument.
        Diagnostic A uses ``A``, ``C`` and ``D`` (as :math:`C'`); ``B`` is
        accepted for interface consistency but not used directly.
    mask_s, mask_t : array-like, optional
        Optional subset masks/indices for stage-specific diagnostics.
    **kwargs
        Forwarded to :func:`relative_wellposedness_diagnostic`.

    Returns
    -------
    dict
        Output dictionary returned by
        :func:`relative_wellposedness_diagnostic`.

    Notes
    -----
    This wrapper maps ``D`` to ``C_prime`` and ignores ``B``.
    """
    _ = B  # accepted for block-interface consistency; not used by Diagnostic A
    return relative_wellposedness_diagnostic(
        A=A,
        C=C,
        C_prime=D,
        mask_s=mask_s,
        mask_t=mask_t,
        **kwargs,
    )


def relative_wellposedness_effective_from_nested_npiv(
    A,
    D,
    B,
    C,
    e_g,
    *,
    mask_s=None,
    mask_t=None,
    **kwargs,
):
    """
    Run post-estimation ``kappa_eff`` for canonical nested NPIV blocks.

    Parameters:
    A, D, B, C
        Canonical nested NPIV blocks. ``D`` is mapped to ``C_prime``.
    e_g : array-like of shape (n,) or (n, 1)
        First-stage error direction used by
        :func:`relative_wellposedness_effective_diagnostic`.
    mask_s, mask_t : array-like, optional
        Optional subset masks/indices for stage-specific diagnostics.
    **kwargs
        Forwarded to :func:`relative_wellposedness_effective_diagnostic`.

    Returns
    -------
    dict
        Output dictionary returned by
        :func:`relative_wellposedness_effective_diagnostic`.

    Notes
    -----
    ``B`` is accepted for interface consistency and ignored.
    """
    _ = B
    return relative_wellposedness_effective_diagnostic(
        A=A,
        C=C,
        C_prime=D,
        e_g=e_g,
        mask_s=mask_s,
        mask_t=mask_t,
        **kwargs,
    )


def relative_wellposedness_sieve_from_nested_npiv(
    A,
    D,
    B,
    C,
    *,
    mask_s=None,
    mask_t=None,
    **kwargs,
):
    """
    Run the pre-estimation sieve-path diagnostic for canonical nested NPIV.

    Parameters:
    A, D, B, C
        Canonical nested NPIV blocks. ``D`` is mapped to ``C_prime``.
    mask_s, mask_t : array-like, optional
        Optional subset masks/indices for stage-specific diagnostics.
    **kwargs
        Forwarded to :func:`relative_wellposedness_sieve_diagnostic`.

    Returns
    -------
    dict
        Output dictionary returned by
        :func:`relative_wellposedness_sieve_diagnostic`.

    Notes
    -----
    ``B`` is accepted for interface consistency and ignored.
    """
    _ = B
    return relative_wellposedness_sieve_diagnostic(
        A=A,
        C=C,
        C_prime=D,
        mask_s=mask_s,
        mask_t=mask_t,
        **kwargs,
    )


def relative_wellposedness_effective_sieve_from_nested_npiv(
    A,
    D,
    B,
    C,
    e_g,
    *,
    mask_s=None,
    mask_t=None,
    **kwargs,
):
    """
    Run the post-estimation sieve-path ``kappa_eff`` diagnostic for nested NPIV.

    Parameters:
    A, D, B, C
        Canonical nested NPIV blocks. ``D`` is mapped to ``C_prime``.
    e_g : array-like of shape (n,) or (n, 1)
        First-stage error direction used by
        :func:`relative_wellposedness_effective_sieve_diagnostic`.
    mask_s, mask_t : array-like, optional
        Optional subset masks/indices for stage-specific diagnostics.
    **kwargs
        Forwarded to :func:`relative_wellposedness_effective_sieve_diagnostic`.

    Returns
    -------
    dict
        Output dictionary returned by
        :func:`relative_wellposedness_effective_sieve_diagnostic`.

    Notes
    -----
    ``B`` is accepted for interface consistency and ignored.
    """
    _ = B
    return relative_wellposedness_effective_sieve_diagnostic(
        A=A,
        C=C,
        C_prime=D,
        e_g=e_g,
        mask_s=mask_s,
        mask_t=mask_t,
        **kwargs,
    )
