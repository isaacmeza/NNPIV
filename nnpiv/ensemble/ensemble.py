"""
Random-forest ensemble estimators for one-stage NPIV problems.

Classes:
    EnsembleIV: Bounded-learner adversarial IV estimator.
    EnsembleIVStar: Heuristic bounded-learner ensemble with an adaptive critic.
    EnsembleIVL2: Empirical-L2-regularized adversarial IV estimator, optionally
        with conditional-moment cross-validation.

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


class EnsembleIV:
    """
    Bounded-learner ensemble IV estimator.

    At each iteration a regression critic fits the current conditional-moment
    residual and a weighted classifier approximates the bounded learner best
    response. Predictions average the resulting learner ensemble.
    
    Parameters:
        adversary (str or estimator): Regression critic. ``'auto'`` uses a
            shallow ``RandomForestRegressor``.
        learner (str or estimator): Weighted-classification learner.
            ``'auto'`` uses a shallow ``RandomForestClassifier``.
        max_abs_value (float): Absolute value of each signed learner output.
        n_iter (int): Number of learners to average; must be positive.
    """
    
    def __init__(self, adversary='auto', learner='auto',
                 max_abs_value=4, n_iter=100):
        self.adversary = adversary
        self.learner = learner
        self.max_abs_value = max_abs_value
        self.n_iter = n_iter
        return

    def _check_input(self, Z, T, Y):
        if len(T.shape) == 1:
            T = T.reshape(-1, 1)
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        return Z, T, Y.flatten()

    def _get_new_adversary(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2,
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learner(self):
        return RandomForestClassifier(n_estimators=5, max_depth=2, criterion='gini',
                                      bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learner == 'auto' else clone(self.learner)

    def fit(self, Z, T, Y):
        """
        Fit the bounded-learner ensemble IV model.
        
        Parameters:
            Z (array-like): Instrument covariates used by the critic.
            T (array-like): Covariates used by the learner function.
            Y (array-like): Scalar outcome.
        
        Returns:
            self: Fitted ensemble IV model.
        """
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        Z, T, Y = self._check_input(Z, T, Y)
        max_value = self.max_abs_value
        adversary = self._get_new_adversary().fit(Z, Y.flatten())
        learners = []
        h = 0
        for it in range(self.n_iter):
            test = adversary.predict(Z).flatten()
            aug_T = np.vstack([np.zeros((2, T.shape[1])), T])
            aug_label = np.concatenate(([-1, 1], _mysign(test)))
            aug_weights = np.concatenate(([0, 0], np.abs(test)))
            learners.append(self._get_new_learner().fit(
                aug_T, aug_label, sample_weight=aug_weights))
            h = h * it / (it + 1)
            h += max_value * _mysign(learners[it].predict_proba(T)[
                :, -1] * learners[it].classes_[-1] - 1 / 2) / (it + 1)
            adversary.fit(Z, Y - h)

        self.learners = learners
        return self

    def predict(self, T):
        """
        Average the fitted bounded learners at ``T``.
        
        Parameters:
            T (array-like): Treatment variables.
        
        Returns:
            array: Predicted outcomes.
        """
        return np.mean([self.max_abs_value * _mysign(l.predict_proba(T)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learners], axis=0)


class EnsembleIVStar:
    """
    Heuristic bounded-learner ensemble with an adaptive critic update.

    This variant selects a linear combination of the previous critic and a
    newly fitted residual critic before computing each learner response. It is
    distinct from the exact best-response update used by ``EnsembleIV``.
    
    Parameters:
        adversary (str or estimator): Regression critic. ``'auto'`` uses a
            shallow ``RandomForestRegressor``.
        learner (str or estimator): Weighted-classification learner.
            ``'auto'`` uses a shallow ``RandomForestClassifier``.
        max_abs_value (float): Absolute value of each signed learner output.
        n_iter (int): Number of learners to average; must be positive.
    """
    
    def __init__(self, adversary='auto', learner='auto',
                 max_abs_value=4, n_iter=100):
        self.adversary = adversary
        self.learner = learner
        self.max_abs_value = max_abs_value
        self.n_iter = n_iter
        return

    def _check_input(self, Z, T, Y):
        if len(T.shape) == 1:
            T = T.reshape(-1, 1)
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        return Z, T, Y.flatten()

    def _get_new_adversary(self):
        return RandomForestRegressor(n_estimators=5, max_depth=2,
                                     bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.0001) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learner(self):
        return RandomForestClassifier(n_estimators=5, max_depth=2, criterion='gini',
                                      bootstrap=False, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learner == 'auto' else clone(self.learner)

    def _update_test(self, Z, Y, pred_old, adv):
        best_loss = np.mean((Y - pred_old)**2)
        pred_new = pred_old.copy()
        for gamma in np.linspace(.1, .9, 5):
            adv.fit(Z, Y - gamma * pred_old)
            pred = adv.predict(Z).flatten()
            loss = np.mean(
                (Y - gamma * pred_old - pred)**2)
            if loss <= best_loss:
                pred_new = gamma * pred_old + pred
                best_loss = loss
        return pred_new

    def fit(self, Z, T, Y):
        """
        Fit the adaptive-critic ensemble IV model.
        
        Parameters:
            Z (array-like): Instrument covariates used by the critic.
            T (array-like): Covariates used by the learner function.
            Y (array-like): Scalar outcome.
        
        Returns:
            self: Fitted ensemble IV model.
        """
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        Z, T, Y = self._check_input(Z, T, Y)
        max_value = self.max_abs_value
        adversary = self._get_new_adversary()
        test = np.zeros(Z.shape[0])
        h = 0
        learners = []
        for it in range(self.n_iter):
            test = self._update_test(Z, Y - h, test, adversary)
            aug_T = np.vstack([np.zeros((2, T.shape[1])), T])
            aug_label = np.concatenate(([-1, 1], _mysign(test)))
            aug_weights = np.concatenate(([0, 0], np.abs(test)))
            learners.append(self._get_new_learner().fit(
                aug_T, aug_label, sample_weight=aug_weights))
            h = h * it / (it + 1)
            h += max_value * _mysign(learners[it].predict_proba(T)[
                :, -1] * learners[it].classes_[-1] - 1 / 2) / (it + 1)

        self.learners = learners
        return self

    def predict(self, T):
        """
        Average the fitted bounded learners at ``T``.
        
        Parameters:
            T (array-like): Treatment variables.
        
        Returns:
            array: Predicted outcomes.
        """
        return np.mean([self.max_abs_value * _mysign(l.predict_proba(T)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learners], axis=0)


class EnsembleIVL2:
    """
    Empirical-L2-regularized ensemble IV estimator.

    The effective learner penalty is ``mu = alpha * delta_n**2``, where
    ``delta_n = delta_scale / n**delta_exp``. Cross-validation evaluates a
    critic trained on each training-fold residual through the held-out payoff
    ``mean(2 * residual * critic - critic**2)``.
    
    Parameters:
        adversary (str or estimator): Regression critic. ``'auto'`` uses a
            shallow ``RandomForestRegressor``.
        learner (str or estimator): Regression learner. ``'auto'`` uses a
            shallow ``RandomForestRegressor``.
        n_iter (int): Number of learners to average; must be positive.
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
    
    def __init__(self, adversary='auto', learner='auto',
                 n_iter=100, delta_scale='auto', delta_exp='auto', CV=False, 
                 alpha_scales='auto', n_alphas=30, n_folds=5):
        self.adversary = adversary
        self.learner = learner
        self.n_iter = n_iter
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
        
    def _check_input(self, Z, T, Y):
        if len(T.shape) == 1:
            T = T.reshape(-1, 1)
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        return Z, T, Y.flatten()

    def _get_new_adversary(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2,
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learner(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2, 
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learner == 'auto' else clone(self.learner)

    def _conditional_moment_score(self, Z_train, residual_train,
                                  Z_test, residual_test):
        """Evaluate a training-critic payoff on held-out residuals."""
        critic = self._get_new_adversary().fit(
            Z_train, np.asarray(residual_train).reshape(-1)
        )
        test_action = critic.predict(Z_test).reshape(-1)
        residual_test = np.asarray(residual_test).reshape(-1)
        return np.mean(
            2 * residual_test * test_action - test_action ** 2
        )

    def _cross_validate_alpha(self, Z, T, Y):
        """
        Select the ``alpha`` scale by held-out conditional-moment payoff.
        
        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatment variables.
            Y (array-like): Outcome variables.
        
        Returns:
            float: Best alpha value.
        """
        alpha_scales = self._get_alpha_scales()
        best_alpha = None
        best_score = float('inf')
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for alpha in alpha_scales:
            scores = []
            for train_index, test_index in kf.split(Z):
                Z_train, Z_test = Z[train_index], Z[test_index]
                T_train, T_test = T[train_index], T[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                
                self.fit(
                    Z_train, T_train, Y_train, alpha=alpha,
                    cross_validating=True
                )
                train_residual = self.predict(T_train) - Y_train
                test_residual = self.predict(T_test) - Y_test
                score = self._conditional_moment_score(
                    Z_train, train_residual, Z_test, test_residual
                )
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
        
        if best_alpha is None:
            raise ValueError("Cross-validation produced no valid alpha score")
        return best_alpha
 
    def fit(self, Z, T, Y, alpha=1.0, cross_validating=False):
        """
        Fit the empirical-L2-regularized ensemble IV model.
        
        Parameters:
            Z (array-like): Instrument covariates used by the critic.
            T (array-like): Covariates used by the learner function.
            Y (array-like): Scalar outcome.
            alpha (float): Positive scale in
                ``mu = alpha * delta_n**2``. Ignored when ``CV=True`` on the
                outer fit.
            cross_validating (bool): Internal guard that prevents recursive
                cross-validation while fitting a fold.
        
        Returns:
            self: Fitted ensemble IV model.
        """
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        Z, T, Y = self._check_input(Z, T, Y)
        if self.CV and not cross_validating:
            alpha = self._cross_validate_alpha(Z, T, Y)
            self.best_alpha_ = alpha

        n = Y.shape[0] 
        delta = self._get_delta(n)
        mu = alpha * delta ** 2
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("alpha * delta**2 must be positive and finite")
        adversary = []
        adversary.append(self._get_new_adversary().fit(Z, Y.flatten()))
        f = 0
        learners = []
        h = 0
        for it in range(self.n_iter):
            f = f * it / (it + 1)
            f += adversary[it].predict(Z).flatten() / (mu * (it + 1))
            learners.append(self._get_new_learner().fit(T, f))
            h = h * it / (it + 1)
            h += learners[it].predict(T).flatten() / (it + 1)
            adversary.append(self._get_new_adversary().fit(Z, Y - h))

        self.learners = learners
        return self

    def predict(self, T):
        """
        Average the fitted regression learners at ``T``.
        
        Parameters:
            T (array-like): Treatment variables.
        
        Returns:
            array: Predicted outcomes.
        """
        return np.mean([l.predict(T) for l in self.learners], axis=0)
