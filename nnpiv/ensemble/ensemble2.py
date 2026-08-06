"""
Random-forest ensemble estimators for simultaneous nested NPIV problems.

Classes:
    Ensemble2IV: Bounded-learner simultaneous adversarial IV estimator.
    Ensemble2IVL2: Empirical-L2-regularized simultaneous adversarial IV
        estimator, optionally with conditional-moment cross-validation.

Functions:
    _mysign: Return 1 for nonnegative inputs and -1 otherwise.
"""

# Licensed under the MIT License.

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import KFold

def _mysign(x):
    return 2 * (x >= 0) - 1


class Ensemble2IV:
    """
    Bounded-learner simultaneous ensemble IV estimator.

    The two critics fit the residuals ``g(A) - Y`` and
    ``h(B) - W * g(A)``. The corresponding learner directions are
    ``-f1 + W * f2`` for ``g`` and ``-f2`` for ``h``. Weighted classifiers
    approximate bounded best responses and predictions average the learners
    retained after burn-in.
    
    Parameters:
        adversary (str or estimator): Regression critic. ``'auto'`` uses a
            shallow ``RandomForestRegressor``.
        learnerg (str or estimator): Weighted-classification learner for
            ``g``. ``'auto'`` uses a shallow ``RandomForestClassifier``.
        learnerh (str or estimator): Weighted-classification learner for
            ``h``. ``'auto'`` uses a shallow ``RandomForestClassifier``.
        max_abs_value (float): Absolute value of each signed learner output.
        n_iter (int): Number of post-burn-in learners to average; must be
            positive.
        n_burn_in (int): Number of preliminary updates excluded from the
            returned ensemble.
    """
    
    def __init__(self, adversary='auto', learnerg='auto', learnerh='auto',
                 max_abs_value=4, n_iter=100, n_burn_in=10):
        self.adversary = adversary
        self.learnerg = learnerg
        self.learnerh = learnerh
        self.max_abs_value = max_abs_value
        self.n_iter = n_iter
        self.n_burn_in = n_burn_in
        return

    def _check_input(self, A, B, C, D, Y, W):
        if len(A.shape) == 1:
            A = A.reshape(-1, 1)
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if len(C.shape) == 1:
            C = C.reshape(-1, 1)
        if len(D.shape) == 1:
            D = D.reshape(-1, 1)
        return A, B, C, D, Y.flatten(), W.flatten()

    def _validate_subset_inputs(self, n, subsetted, subset_ind1,
                                subset_ind2):
        if not subsetted:
            return None, None
        if subset_ind1 is None:
            raise ValueError(
                "subset_ind1 must be provided when subsetted is True"
            )
        subset_ind1 = np.asarray(subset_ind1).reshape(-1)
        if subset_ind1.shape[0] != n:
            raise ValueError("subset_ind1 must have the same length as Y")
        if not np.all(np.isin(subset_ind1, (0, 1))):
            raise ValueError("subset_ind1 must be a binary indicator")
        if subset_ind2 is None:
            subset_ind2 = 1 - subset_ind1
        else:
            subset_ind2 = np.asarray(subset_ind2).reshape(-1)
            if subset_ind2.shape[0] != n:
                raise ValueError("subset_ind2 must have the same length as Y")
            if not np.all(np.isin(subset_ind2, (0, 1))):
                raise ValueError("subset_ind2 must be a binary indicator")
        ind1 = np.flatnonzero(subset_ind1 == 1)
        ind2 = np.flatnonzero(subset_ind2 == 1)
        if ind1.size == 0:
            raise ValueError("subset_ind1 selects zero observations")
        if ind2.size == 0:
            raise ValueError(
                "subset_ind2/subset_ind1 complement selects zero observations"
            )
        return ind1, ind2

    @staticmethod
    def _scaled_action(action, n, indices):
        if indices is None:
            return np.asarray(action).reshape(-1)
        result = np.zeros(n)
        result[indices] = (
            n / indices.size
        ) * np.asarray(action).reshape(-1)[indices]
        return result

    @staticmethod
    def _fit_adversary(adversary, X, target, indices=None):
        if indices is None:
            return adversary.fit(X, np.asarray(target).reshape(-1))
        return adversary.fit(
            X[indices], np.asarray(target).reshape(-1)[indices]
        )

    def _get_new_adversary(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2,
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learnerg(self):
        return RandomForestClassifier(n_estimators=5, max_depth=2, criterion='gini',
                                      bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learnerg == 'auto' else clone(self.learnerg)
    
    def _get_new_learnerh(self):
        return RandomForestClassifier(n_estimators=5, max_depth=2, criterion='gini',
                                      bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learnerh == 'auto' else clone(self.learnerh)

    def fit(self, A, B, C, D, Y, W=None, subsetted=False, subset_ind1=None, subset_ind2=None):
        """
        Fit the bounded-learner simultaneous ensemble IV model.
        
        Parameters:
            A (array-like): Covariates for the first learner ``g``.
            B (array-like): Covariates for the second learner ``h``.
            C (array-like): Instruments for the residual
                ``h(B) - W * g(A)``.
            D (array-like): Instruments for the residual ``g(A) - Y``.
            Y (array-like): Scalar outcome.
            W (array-like, optional): Observation-level multiplier of
                ``g(A)`` in the second bridge equation. Defaults to one.
            subsetted (bool): Whether to estimate the two moment equations on
                selected rows.
            subset_ind1 (array-like, optional): Nonempty binary mask for the
                ``g(A) - Y`` moment. Required when ``subsetted=True``.
            subset_ind2 (array-like, optional): Nonempty binary mask for the
                ``h(B) - W * g(A)`` moment. Defaults to the complement of
                ``subset_ind1``; explicit masks may overlap and need not cover
                every observation.
        
        Returns:
            self: Fitted nested ensemble IV model.
        """
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if self.n_burn_in < 0:
            raise ValueError("n_burn_in must be nonnegative")
        Y = np.asarray(Y)
        W = np.ones(Y.shape) if W is None else W
        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W)
        ind1, ind2 = self._validate_subset_inputs(
            len(Y), subsetted, subset_ind1, subset_ind2
        )

        max_value = self.max_abs_value
        adversary1 = self._fit_adversary(
            self._get_new_adversary(), D, -Y, ind1
        )
        adversary2 = self._fit_adversary(
            self._get_new_adversary(), C, -Y, ind2
        )
        learnersg = []
        learnersh = []
        h = np.zeros_like(Y, dtype=float)
        g = np.zeros_like(Y, dtype=float)

        for it in range(self.n_iter + self.n_burn_in):
            phase_it = it if it < self.n_burn_in else it - self.n_burn_in
            if it == self.n_burn_in:
                g.fill(0)
                h.fill(0)

            action1 = self._scaled_action(
                adversary1.predict(D), len(Y), ind1
            )
            action2 = self._scaled_action(
                adversary2.predict(C), len(Y), ind2
            )
            v = -action2
            v_ = -action1 + W * action2
            aug_A = np.vstack([np.zeros((2, A.shape[1])), A])
            aug_B = np.vstack([np.zeros((2, B.shape[1])), B])
            lbl_v = np.concatenate(([-1, 1], _mysign(v)))
            lbl_v_ = np.concatenate(([-1, 1], _mysign(v_)))

            wt_v = np.concatenate(([0, 0], np.abs(v)))
            wt_v_ = np.concatenate(([0, 0], np.abs(v_)))

            learnersg.append(self._get_new_learnerg().fit(
                aug_A, lbl_v_, sample_weight=wt_v_))
            learnersh.append(self._get_new_learnerh().fit(
                aug_B, lbl_v, sample_weight=wt_v))
            g = g * phase_it / (phase_it + 1)
            h = h * phase_it / (phase_it + 1)

            g += max_value * _mysign(learnersg[it].predict_proba(A)[
                :, -1] * learnersg[it].classes_[-1] - 1 / 2) / (phase_it + 1)
            h += max_value * _mysign(learnersh[it].predict_proba(B)[
                :, -1] * learnersh[it].classes_[-1] - 1 / 2) / (phase_it + 1)
            self._fit_adversary(adversary2, C, h - g * W, ind2)
            self._fit_adversary(adversary1, D, g - Y, ind1)

        self.learnersg = learnersg[self.n_burn_in:]
        self.learnersh = learnersh[self.n_burn_in:]
        return self

    def predict(self, B, *args):
        """
        Predict the second bridge, and optionally both bridge functions.
        
        Parameters:
            B (array-like): Covariates at which to evaluate ``h``.
            args (tuple): Optional single array of covariates at which to
                evaluate ``g``.
        
        Returns:
            array or tuple: ``h(B)`` if only ``B`` is supplied; otherwise the
            tuple ``(h(B), g(A))``.
        """
        if len(args) == 0:
            # Only B_test provided, return h prediction
            return np.mean([self.max_abs_value * _mysign(l.predict_proba(B)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learnersh], axis=0)
        elif len(args) == 1:
            # Two arguments provided, assume the second is A_test
            A = args[0]
            pred_h = np.mean([self.max_abs_value * _mysign(l.predict_proba(B)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learnersh], axis=0)
            pred_g = np.mean([self.max_abs_value * _mysign(l.predict_proba(A)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learnersg], axis=0)
            return pred_h, pred_g

        else:
            # More than one additional argument provided, raise an error
            raise ValueError("predict expects at most two arguments, B_test and optionally A_test")


class Ensemble2IVL2:
    """
    Empirical-L2-regularized simultaneous ensemble IV estimator.

    Both bridge functions use the effective penalty
    ``mu = alpha * delta_n**2``, with
    ``delta_n = delta_scale / n**delta_exp``. The two learner targets average
    ``(-f1 + W * f2) / mu`` and ``-f2 / mu``. Cross-validation sums held-out
    conditional-moment payoffs for the two residual equations.
    
    Parameters:
        adversary (str or estimator): Regression critic. ``'auto'`` uses a
            shallow ``RandomForestRegressor``.
        learnerg (str or estimator): Regression learner for ``g``. ``'auto'``
            uses a shallow ``RandomForestRegressor``.
        learnerh (str or estimator): Regression learner for ``h``. ``'auto'``
            uses a shallow ``RandomForestRegressor``.
        n_iter (int): Number of post-burn-in learners to average; must be
            positive.
        n_burn_in (int): Number of preliminary updates excluded from the
            returned ensemble.
        delta_scale (str or float): Numerator of ``delta_n``. ``'auto'`` uses
            5.
        delta_exp (str or float): Sample-size exponent in ``delta_n``.
            ``'auto'`` uses 0.4.
        CV (bool): Whether to select ``alpha`` by cross-validation.
        alpha_scales (str or iterable): Candidate ``alpha`` scales.
            ``'auto'`` uses a geometric grid.
        n_alphas (int): Size of the automatic candidate grid.
        n_folds (int): Number of cross-validation folds.

    Attributes:
        best_alpha_ (float): Selected scale when ``CV=True``.
    """
    
    def __init__(self, adversary='auto', learnerg='auto', learnerh='auto',
                 n_iter=100, n_burn_in=10, delta_scale='auto', delta_exp='auto', CV=False, 
                 alpha_scales='auto', n_alphas=30, n_folds=5):
        self.adversary = adversary
        self.learnerg = learnerg
        self.learnerh = learnerh
        self.n_iter = n_iter
        self.n_burn_in = n_burn_in
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.CV = CV
        self.alpha_scales = alpha_scales
        self.n_alphas = n_alphas
        self.n_folds = n_folds
        return

    def _get_delta(self, n):
        """
        Compute ``delta_scale / n**delta_exp``.
        
        Parameters:
            n (int): Sample size.
        
        Returns:
            float: Critical radius delta.
        """
        delta_scale = 5 if self.delta_scale == 'auto' else self.delta_scale
        delta_exp = .4 if self.delta_exp == 'auto' else self.delta_exp
        return delta_scale / (n**(delta_exp))
    
    def _get_alpha_scales(self):
        return ([c for c in np.geomspace(0.1, 1e4, self.n_alphas)]
                if self.alpha_scales == 'auto' else self.alpha_scales)
        
    def _check_input(self, A, B, C, D, Y, W):
        if len(A.shape) == 1:
            A = A.reshape(-1, 1)
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if len(C.shape) == 1:
            C = C.reshape(-1, 1)
        if len(D.shape) == 1:
            D = D.reshape(-1, 1)
        return A, B, C, D, Y.flatten(), W.flatten()

    def _validate_subset_inputs(self, n, subsetted, subset_ind1,
                                subset_ind2):
        return Ensemble2IV._validate_subset_inputs(
            self, n, subsetted, subset_ind1, subset_ind2
        )

    @staticmethod
    def _scaled_action(action, n, indices):
        return Ensemble2IV._scaled_action(action, n, indices)

    @staticmethod
    def _fit_adversary(adversary, X, target, indices=None):
        return Ensemble2IV._fit_adversary(adversary, X, target, indices)

    def _conditional_moment_score(self, X_train, residual_train,
                                  X_test, residual_test):
        """Evaluate a training-critic payoff on held-out residuals."""
        critic = self._get_new_adversary().fit(
            X_train, np.asarray(residual_train).reshape(-1)
        )
        test_action = critic.predict(X_test).reshape(-1)
        residual_test = np.asarray(residual_test).reshape(-1)
        return np.mean(
            2 * residual_test * test_action - test_action ** 2
        )

    def _get_new_adversary(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2,
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learnerg(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2, 
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learnerg == 'auto' else clone(self.learnerg)

    def _get_new_learnerh(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2, 
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learnerh == 'auto' else clone(self.learnerh)

    def _cross_validate_alpha(self, A, B, C, D, Y, W,
                              subsetted=False, subset_ind1=None,
                              subset_ind2=None):
        """
        Select the ``alpha`` scale by the two conditional-moment payoffs.
        
        Parameters:
            A (array-like): Covariates for ``g``.
            B (array-like): Covariates for ``h``.
            C (array-like): Instruments for ``h(B) - W * g(A)``.
            D (array-like): Instruments for ``g(A) - Y``.
            Y (array-like): Scalar outcome.
            W (array-like): Multiplier in the second bridge residual.
        
        Returns:
            float: Best alpha value.
        """
        alpha_scales = self._get_alpha_scales()
        best_alpha = None
        best_score = float('inf')
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for alpha in alpha_scales:
            scores = []
            for train_index, test_index in kf.split(Y):
                A_train, A_test = A[train_index], A[test_index]
                B_train, B_test = B[train_index], B[test_index]
                C_train, C_test = C[train_index], C[test_index]
                D_train, D_test = D[train_index], D[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                W_train, W_test = W[train_index], W[test_index]
                
                train_ind1 = (
                    np.asarray(subset_ind1).reshape(-1)[train_index]
                    if subsetted else None
                )
                test_ind1 = (
                    np.asarray(subset_ind1).reshape(-1)[test_index]
                    if subsetted else None
                )
                if subsetted and subset_ind2 is not None:
                    subset_ind2_array = np.asarray(subset_ind2).reshape(-1)
                    train_ind2 = subset_ind2_array[train_index]
                    test_ind2 = subset_ind2_array[test_index]
                elif subsetted:
                    train_ind2 = 1 - train_ind1
                    test_ind2 = 1 - test_ind1
                else:
                    train_ind2 = test_ind2 = None

                try:
                    train_rows1, train_rows2 = self._validate_subset_inputs(
                        len(train_index), subsetted, train_ind1, train_ind2
                    )
                    test_rows1, test_rows2 = self._validate_subset_inputs(
                        len(test_index), subsetted, test_ind1, test_ind2
                    )
                except ValueError:
                    continue

                self.fit(
                    A_train, B_train, C_train, D_train, Y_train,
                    W=W_train, alpha=alpha, cross_validating=True,
                    subsetted=subsetted, subset_ind1=train_ind1,
                    subset_ind2=train_ind2
                )
                train_h, train_g = self.predict(B_train, A_train)
                test_h, test_g = self.predict(B_test, A_test)
                train_residual1 = train_g - Y_train
                test_residual1 = test_g - Y_test
                train_residual2 = train_h - W_train * train_g
                test_residual2 = test_h - W_test * test_g
                score = self._conditional_moment_score(
                    D_train if train_rows1 is None else D_train[train_rows1],
                    train_residual1 if train_rows1 is None else train_residual1[train_rows1],
                    D_test if test_rows1 is None else D_test[test_rows1],
                    test_residual1 if test_rows1 is None else test_residual1[test_rows1],
                )
                score += self._conditional_moment_score(
                    C_train if train_rows2 is None else C_train[train_rows2],
                    train_residual2 if train_rows2 is None else train_residual2[train_rows2],
                    C_test if test_rows2 is None else C_test[test_rows2],
                    test_residual2 if test_rows2 is None else test_residual2[test_rows2],
                )
                scores.append(score)
            
            if not scores:
                continue
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
        
        if best_alpha is None:
            raise ValueError("Cross-validation produced no valid alpha score")
        return best_alpha
 
    def fit(self, A, B, C, D, Y, W=None, alpha=1.0, cross_validating=False, subsetted=False, subset_ind1=None, subset_ind2=None): 
        """
        Fit the empirical-L2-regularized simultaneous ensemble IV model.
        
        Parameters:
            A (array-like): Covariates for the first learner ``g``.
            B (array-like): Covariates for the second learner ``h``.
            C (array-like): Instruments for the residual
                ``h(B) - W * g(A)``.
            D (array-like): Instruments for the residual ``g(A) - Y``.
            Y (array-like): Scalar outcome.
            W (array-like, optional): Observation-level multiplier of
                ``g(A)`` in the second bridge equation. Defaults to one.
            alpha (float): Positive scale in
                ``mu = alpha * delta_n**2``. Ignored when ``CV=True`` on the
                outer fit.
            cross_validating (bool): Internal guard that prevents recursive
                cross-validation while fitting a fold.
            subsetted (bool): Whether to estimate the two moment equations on
                selected rows.
            subset_ind1 (array-like, optional): Nonempty binary mask for the
                ``g(A) - Y`` moment. Required when ``subsetted=True``.
            subset_ind2 (array-like, optional): Nonempty binary mask for the
                ``h(B) - W * g(A)`` moment. Defaults to the complement of
                ``subset_ind1``; explicit masks may overlap and need not cover
                every observation.
        
        Returns:
            self: Fitted nested ensemble IV model.
        """
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if self.n_burn_in < 0:
            raise ValueError("n_burn_in must be nonnegative")
        Y = np.asarray(Y)
        W = np.ones(Y.shape) if W is None else W
        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W)
        ind1, ind2 = self._validate_subset_inputs(
            len(Y), subsetted, subset_ind1, subset_ind2
        )
        if self.CV and not cross_validating:
            alpha = self._cross_validate_alpha(
                A, B, C, D, Y, W, subsetted=subsetted,
                subset_ind1=subset_ind1, subset_ind2=subset_ind2
            )
            self.best_alpha_ = alpha

        n = Y.shape[0] 
        delta = self._get_delta(n)
        mu = alpha * delta ** 2
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("alpha * delta**2 must be positive and finite")
        adversary1 = self._fit_adversary(
            self._get_new_adversary(), D, -Y, ind1
        )
        adversary2 = self._fit_adversary(
            self._get_new_adversary(), C, np.zeros_like(Y), ind2
        )
        learnersg = []
        learnersh = []
        h = np.zeros_like(Y, dtype=float)
        g = np.zeros_like(Y, dtype=float)
        v = np.zeros_like(Y, dtype=float)
        v_ = np.zeros_like(Y, dtype=float)
        for it in range(self.n_iter + self.n_burn_in):
            phase_it = it if it < self.n_burn_in else it - self.n_burn_in
            if it == self.n_burn_in:
                g.fill(0)
                h.fill(0)
                v.fill(0)
                v_.fill(0)

            action1 = self._scaled_action(
                adversary1.predict(D), n, ind1
            )
            action2 = self._scaled_action(
                adversary2.predict(C), n, ind2
            )
            current_v = -action2 / mu
            current_v_ = (-action1 + W * action2) / mu
            v = (phase_it * v + current_v) / (phase_it + 1)
            v_ = (phase_it * v_ + current_v_) / (phase_it + 1)
            learnersg.append(self._get_new_learnerg().fit(A, v_))
            learnersh.append(self._get_new_learnerh().fit(B, v))
            g = g * phase_it / (phase_it + 1)
            h = h * phase_it / (phase_it + 1)
            g += learnersg[it].predict(A).flatten() / (phase_it + 1)
            h += learnersh[it].predict(B).flatten() / (phase_it + 1)
            self._fit_adversary(adversary2, C, h - g * W, ind2)
            self._fit_adversary(adversary1, D, g - Y, ind1)

        self.learnersg = learnersg[self.n_burn_in:]
        self.learnersh = learnersh[self.n_burn_in:]
        return self

    def predict(self, B, *args):
        """
        Predict the second bridge, and optionally both bridge functions.
        
        Parameters:
            B (array-like): Covariates at which to evaluate ``h``.
            args (tuple): Optional single array of covariates at which to
                evaluate ``g``.
        
        Returns:
            array or tuple: ``h(B)`` if only ``B`` is supplied; otherwise the
            tuple ``(h(B), g(A))``.
        """
        if len(args) == 0:
            # Only B_test provided, return h prediction
            return np.mean([l.predict(B) for l in self.learnersh], axis=0)
        elif len(args) == 1:
            # Two arguments provided, assume the second is A_test
            A = args[0]
            return np.mean([l.predict(B) for l in self.learnersh], axis=0), np.mean([l.predict(A) for l in self.learnersg], axis=0)
        else:
            # More than one additional argument provided, raise an error
            raise ValueError("predict expects at most two arguments, B_test and optionally A_test")
