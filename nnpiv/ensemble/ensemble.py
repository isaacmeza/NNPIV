"""
This module provides implementations of ensemble instrumental variable (IV) estimators using RandomForest models.

Classes:
    EnsembleIV: Implements an ensemble learning IV method with adversarial and learner components.
    EnsembleIVStar: Similar to EnsembleIV but with a different method for updating the test predictions.
    EnsembleIVL2: An extension of EnsembleIV with L2 regularization and optional cross-validation for regularization parameter selection.

Functions:
    _mysign: A helper function that returns 2 if the input is non-negative and -1 otherwise.
"""

# Licensed under the MIT License.

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def _mysign(x):
    return 2 * (x >= 0) - 1


class EnsembleIV:
    """
    Implements an ensemble learning IV method with adversarial and learner components.
    
    Parameters:
        adversary (str or estimator): Adversary model. If 'auto', a default RandomForestRegressor is used.
        learner (str or estimator): Learner model. If 'auto', a default RandomForestClassifier is used.
        max_abs_value (float): Maximum absolute value for the predictions.
        n_iter (int): Number of iterations for the ensemble.
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
        Fits the ensemble IV model to the provided data.
        
        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatment variables.
            Y (array-like): Outcome variables.
        
        Returns:
            self: Fitted ensemble IV model.
        """
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
        Predicts outcomes for new data using the fitted ensemble IV model.
        
        Parameters:
            T (array-like): Treatment variables.
        
        Returns:
            array: Predicted outcomes.
        """
        return np.mean([self.max_abs_value * _mysign(l.predict_proba(T)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learners], axis=0)


class EnsembleIVStar:
    """
    Similar to EnsembleIV but with a different method for updating the test predictions using a linear combination approach.
    
    Parameters:
        adversary (str or estimator): Adversary model. If 'auto', a default RandomForestRegressor is used.
        learner (str or estimator): Learner model. If 'auto', a default RandomForestClassifier is used.
        max_abs_value (float): Maximum absolute value for the predictions.
        n_iter (int): Number of iterations for the ensemble.
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
        Fits the ensemble IV model to the provided data.
        
        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatment variables.
            Y (array-like): Outcome variables.
        
        Returns:
            self: Fitted ensemble IV model.
        """
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
        Predicts outcomes for new data using the fitted ensemble IV model.
        
        Parameters:
            T (array-like): Treatment variables.
        
        Returns:
            array: Predicted outcomes.
        """
        return np.mean([self.max_abs_value * _mysign(l.predict_proba(T)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learners], axis=0)


class EnsembleIVL2:
    """
    An extension of EnsembleIV with L2 regularization and optional cross-validation to select the best regularization parameter.
    
    Parameters:
        adversary (str or estimator): Adversary model. If 'auto', a default RandomForestRegressor is used.
        learner (str or estimator): Learner model. If 'auto', a default RandomForestRegressor is used.
        n_iter (int): Number of iterations for the ensemble.
        delta_scale (str or float): Scale factor for the critical radius delta. Default is 'auto'.
        delta_exp (str or float): Exponent for the critical radius delta. Default is 'auto'.
        CV (bool): Whether to perform cross-validation to select the best alpha value.
        alpha_scales (str or list): Scales for alpha in cross-validation. Default is 'auto'.
        n_alphas (int): Number of alpha values to test in cross-validation.
        n_folds (int): Number of folds for cross-validation.
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
        '''
        Computes the critical radius delta based on the sample size.
        
        Parameters:
            n (int): Sample size.
        
        Returns:
            float: Critical radius delta.
        '''
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

    def _cross_validate_alpha(self, Z, T, Y):
        """
        Performs cross-validation to select the best alpha value.
        
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
                
                self.fit(Z_train, T_train, Y_train, alpha=alpha)
                predictions = self.predict(T_test)
                score = mean_squared_error(Y_test, predictions)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
        
        return best_alpha
 
    def fit(self, Z, T, Y, alpha=1.0, cross_validating=False):
        """
        Fits the ensemble IV model with L2 regularization to the provided data.
        
        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatment variables.
            Y (array-like): Outcome variables.
            alpha (float): Regularization parameter.
            cross_validating (bool): Whether the function is called during cross-validation.
        
        Returns:
            self: Fitted ensemble IV model.
        """
        if self.CV and not cross_validating:
            alpha = self._cross_validate_alpha(Z, T, Y)

        Z, T, Y = self._check_input(Z, T, Y)
        n = Y.shape[0] 
        delta = self._get_delta(n)
        adversary = []
        adversary.append(self._get_new_adversary().fit(Z, Y.flatten()))
        f = 0
        learners = []
        h = 0
        for it in range(self.n_iter):
            f = f * it / (it + 1)
            f += adversary[it].predict(Z).flatten() / ((alpha * delta ** 2) * (it + 1))
            learners.append(self._get_new_learner().fit(T, f))
            h = h * it / (it + 1)
            h += learners[it].predict(T).flatten() / (it + 1)
            adversary.append(self._get_new_adversary().fit(Z, Y - h))

        self.learners = learners
        return self

    def predict(self, T):
        """
        Predicts outcomes for new data using the fitted ensemble IV model with L2 regularization.
        
        Parameters:
            T (array-like): Treatment variables.
        
        Returns:
            array: Predicted outcomes.
        """
        return np.mean([l.predict(T) for l in self.learners], axis=0)
