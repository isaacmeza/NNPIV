# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import cvxopt
from functools import reduce


def cross_product(*XS):
    """
    Compute the cross product of features.

    Parameters:
        XS (array-like): Feature matrices with the same number of rows.
            One-dimensional inputs are treated as column vectors.

    Returns
    -------
    A : n x (d1*d2*...) matrix
        Matrix of n samples of d1*d2*... cross product features,
        arranged in form such that each row t of X12 contains:
        [X1[t,0]*X2[t,0]*..., ..., X1[t,d1-1]*X2[t,0]*..., X1[t,0]*X2[t,1]*..., ..., X1[t,d1-1]*X2[t,1]*..., ...]
    """
    for X in XS:
        assert 2 >= np.ndim(X) >= 1
    n = np.shape(XS[0])[0]
    for X in XS:
        assert n == np.shape(X)[0]

    def cross(XS):
        k = len(XS)
        XS = [np.reshape(XS[i], (n,) + (1,) * (k - i - 1) + (-1,) + (1,) * i)
              for i in range(k)]
        return np.reshape(reduce(np.multiply, XS), (n, -1))
    return cross(XS)


def quadratic_min_l1(linear, quadratic, regularization, radius,
                     quadratic_pinv=None):
    r"""Return the minimum of a convex quadratic over an :math:`\ell_1` ball.

    This evaluates

    .. math::

        \min_{\|x\|_1\leq R}
        c^\top x + \frac{\lambda}{2}x^\top Qx.

    A feasible unconstrained solution is used when supplied through
    ``quadratic_pinv``; otherwise the constrained problem is solved in the
    positive-negative lifting of ``x``.

    Parameters:
        linear (array-like): Linear coefficient :math:`c`.
        quadratic (array-like): Positive-semidefinite matrix :math:`Q`.
        regularization (float): Nonnegative multiplier :math:`\lambda`.
        radius (float): Nonnegative radius :math:`R`.
        quadratic_pinv (array-like, optional): Precomputed pseudoinverse of
            :math:`Q`.

    Returns:
        float: Minimum objective value.
    """
    linear = np.asarray(linear, dtype=float).reshape(-1)
    quadratic = np.asarray(quadratic, dtype=float)
    radius = float(radius)

    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")
    if radius == 0 or linear.size == 0:
        return 0.0
    if np.linalg.norm(linear, ord=np.inf) == 0:
        return 0.0
    if regularization == 0:
        return -radius * np.linalg.norm(linear, ord=np.inf)

    d = linear.size
    quadratic = .5 * (quadratic + quadratic.T)
    if quadratic_pinv is not None:
        candidate = -np.asarray(quadratic_pinv) @ linear / regularization
        residual = linear + regularization * quadratic @ candidate
        tolerance = 1e-10 * max(1.0, np.linalg.norm(linear))
        if (np.linalg.norm(residual) <= tolerance
                and np.linalg.norm(candidate, ord=1) <= radius + tolerance):
            return float(linear @ candidate
                         + .5 * regularization
                         * candidate @ quadratic @ candidate)

    lifted_quadratic = regularization * np.block([
        [quadratic, -quadratic],
        [-quadratic, quadratic]
    ])
    lifted_linear = np.concatenate([linear, -linear])
    constraints = np.vstack([-np.eye(2 * d), np.ones((1, 2 * d))])
    bounds = np.concatenate([np.zeros(2 * d), [radius]])

    previous_progress = cvxopt.solvers.options.get('show_progress', True)
    cvxopt.solvers.options['show_progress'] = False
    try:
        solution = cvxopt.solvers.qp(
            cvxopt.matrix(lifted_quadratic),
            cvxopt.matrix(lifted_linear),
            cvxopt.matrix(constraints),
            cvxopt.matrix(bounds)
        )
    finally:
        cvxopt.solvers.options['show_progress'] = previous_progress

    if solution['status'] != 'optimal':
        raise RuntimeError("Unable to compute the l1-ball best response")
    lifted = np.asarray(solution['x']).reshape(-1)
    point = lifted[:d] - lifted[d:]
    return float(linear @ point
                 + .5 * regularization * point @ quadratic @ point)


def quadratic_min_l2(linear, quadratic, regularization, radius,
                     quadratic_eigh=None):
    r"""Return the minimum of a convex quadratic over an :math:`\ell_2` ball.

    The minimized objective is
    :math:`c^\top x+(\lambda/2)x^\top Qx` subject to
    :math:`\|x\|_2\leq R`.  The trust-region multiplier is found by bisection
    when the unconstrained minimizer is infeasible.

    Parameters:
        linear (array-like): Linear coefficient :math:`c`.
        quadratic (array-like): Positive-semidefinite matrix :math:`Q`.
        regularization (float): Nonnegative multiplier :math:`\lambda`.
        radius (float): Nonnegative radius :math:`R`.
        quadratic_eigh (tuple, optional): Precomputed eigenvalues and
            eigenvectors of :math:`Q`.

    Returns:
        float: Minimum objective value.
    """
    linear = np.asarray(linear, dtype=float).reshape(-1)
    quadratic = np.asarray(quadratic, dtype=float)
    radius = float(radius)

    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")
    if radius == 0 or linear.size == 0:
        return 0.0
    linear_norm = np.linalg.norm(linear)
    if linear_norm == 0:
        return 0.0
    if regularization == 0:
        return -radius * linear_norm

    if quadratic_eigh is None:
        eigenvalues, eigenvectors = np.linalg.eigh(
            .5 * (quadratic + quadratic.T))
    else:
        eigenvalues, eigenvectors = quadratic_eigh
    eigenvalues = np.maximum(eigenvalues, 0)
    coordinates = eigenvectors.T @ linear
    denominators = regularization * eigenvalues
    tolerance = np.finfo(float).eps * max(1.0, np.max(denominators)) * linear.size
    in_range = np.all(np.abs(coordinates[denominators <= tolerance]) <= tolerance)

    if in_range:
        transformed = np.zeros_like(coordinates)
        nonzero = denominators > tolerance
        transformed[nonzero] = -coordinates[nonzero] / denominators[nonzero]
        if np.linalg.norm(transformed) <= radius:
            point = eigenvectors @ transformed
            return float(linear @ point
                         + .5 * regularization * point @ quadratic @ point)

    def squared_norm(multiplier):
        return np.sum((coordinates / (denominators + multiplier))**2)

    lower = 0.0
    upper = max(1.0, linear_norm / radius)
    while squared_norm(upper) > radius**2:
        upper *= 2
    for _ in range(80):
        midpoint = .5 * (lower + upper)
        if squared_norm(midpoint) > radius**2:
            lower = midpoint
        else:
            upper = midpoint

    transformed = -coordinates / (denominators + upper)
    point = eigenvectors @ transformed
    return float(linear @ point
                 + .5 * regularization * point @ quadratic @ point)


def quadratic_min_l2_identity(linear, regularization, radius):
    r"""Return an isotropic quadratic minimum over an :math:`\ell_2` ball.

    Evaluates :math:`\min_{\|x\|_2\leq R}
    c^\top x+(\lambda/2)\|x\|_2^2` in closed form.

    Parameters:
        linear (array-like): Linear coefficient :math:`c`.
        regularization (float): Nonnegative multiplier :math:`\lambda`.
        radius (float): Nonnegative radius :math:`R`.

    Returns:
        float: Minimum objective value.
    """
    linear = np.asarray(linear, dtype=float).reshape(-1)
    radius = float(radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")

    linear_norm = np.linalg.norm(linear, ord=2)
    if radius == 0 or linear_norm == 0:
        return 0.0
    response_norm = (radius if regularization == 0
                     else min(radius, linear_norm / regularization))
    return float(-linear_norm * response_norm
                 + .5 * regularization * response_norm**2)
