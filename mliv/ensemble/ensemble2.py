# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def _mysign(x):
    return 2 * (x >= 0) - 1


class Ensemble2IV:

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
        W = np.ones(Y.shape) if W is None else W
        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W)
        if subsetted:
            if subset_ind1 is None:
                raise ValueError("subset_ind1 must be provided when subsetted is True")
            if len(subset_ind1) != len(Y):
                raise ValueError("subset_ind1 must have the same length as Y")
            subset_ind1 = subset_ind1.flatten()
            subset_ind2 = subset_ind2.flatten() if subset_ind2 is not None else 1 - subset_ind1 

        max_value = self.max_abs_value
        adversary1 = self._get_new_adversary().fit(D, -Y.flatten())
        adversary2 = self._get_new_adversary().fit(C, -Y.flatten())
        learnersg = []
        learnersh = []
        h = 0
        g = 0

        for it in range(self.n_iter + self.n_burn_in):
            v = -adversary2.predict(C).flatten()* W if not subsetted else -adversary2.predict(C).flatten()* W * subset_ind2
            v_ = -adversary1.predict(D).flatten() - v if not subsetted else -adversary1.predict(D).flatten() * subset_ind1 - v
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
            g = g * it / (it + 1)
            h = h * it / (it + 1)

            g += max_value * _mysign(learnersg[it].predict_proba(A)[
                :, -1] * learnersg[it].classes_[-1] - 1 / 2) / (it + 1)
            h += max_value * _mysign(learnersh[it].predict_proba(B)[
                :, -1] * learnersh[it].classes_[-1] - 1 / 2) / (it + 1)
            adversary2.fit(C, h - g*W)
            adversary1.fit(D, g - Y)

        self.learnersg = learnersg[self.n_burn_in:]
        self.learnersh = learnersh[self.n_burn_in:]
        return self

    def predict(self, B, *args):
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

    def __init__(self, adversary='auto', learnerg='auto', learnerh='auto',
                 n_iter=100, n_burn_in=10, delta_scale='auto', delta_exp='auto', CV = False, 
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
        '''
        delta -> Critical radius
        '''
        delta_scale = 5 if self.delta_scale == 'auto' else self.delta_scale
        delta_exp = .4 if self.delta_exp == 'auto' else self.delta_exp
        return delta_scale / (n**(delta_exp))
    
    def _get_alpha_scales(self):
        return ([c for c in np.geomspace(0.1, 1e4, self.n_alphas)]
                if self.alpha_scales == True else self.alpha_scales)
        
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

    def _get_new_adversary(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2,
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learnerg(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2, 
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learnerg == 'auto' else clone(self.learnerg)

    def _get_new_learnerh(self):
        return RandomForestRegressor(n_estimators=40, max_depth=2, 
                                     bootstrap=True, min_samples_leaf=40, min_impurity_decrease=0.001) if self.learnerh == 'auto' else clone(self.learnerh)

    def _cross_validate_alpha(self, A, B, C, D, Y, W):

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
                
                self.fit(A_train, B_train, C_train, D_train, Y_train, W=W_train, alpha=alpha)
                predictionB, predictionA = self.predict(B_test, A_test)
                score = mean_squared_error(Y_test, predictionA) + mean_squared_error(predictionA*W_test, predictionB)
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha
        
        return best_alpha
 
    def fit(self, A, B, C, D, Y, W=None, alpha=1.0, cross_validating=False, subsetted=False, subset_ind1=None, subset_ind2=None): 
        W = np.ones(Y.shape) if W is None else W
        if self.CV and not cross_validating:
            alpha = self._cross_validate_alpha(A, B, C, D, Y, W)

        A, B, C, D, Y, W = self._check_input(A, B, C, D, Y, W)
        if subsetted:
            if subset_ind1 is None:
                raise ValueError("subset_ind1 must be provided when subsetted is True")
            if len(subset_ind1) != len(Y):
                raise ValueError("subset_ind1 must have the same length as Y")
            subset_ind1 = subset_ind1.flatten()
            subset_ind2 = subset_ind2.flatten() if subset_ind2 is not None else 1 - subset_ind1

        n = Y.shape[0] 
        delta = self._get_delta(n)
        adversary1 = self._get_new_adversary().fit(D, -Y.flatten())
        adversary2 = self._get_new_adversary().fit(C, np.zeros(Y.shape))
        learnersg = []
        learnersh = []
        h = 0
        g = 0
        for it in range(self.n_iter + self.n_burn_in):
            if it < self.n_burn_in:
                v = -adversary2.predict(C).flatten()* W if not subsetted else -adversary2.predict(C).flatten()*W * subset_ind2
                v_ = -adversary1.predict(D).flatten() - v if not subsetted else -adversary1.predict(D).flatten() * subset_ind1 - v
            else:
                iter = it - self.n_burn_in
                v = v * iter / (iter + 1)
                v_ = v_ * iter / (iter + 1)
                v += -adversary2.predict(C).flatten()* W / (alpha * delta ** 2) if not subsetted else -adversary2.predict(C).flatten()*W * subset_ind2 / (alpha * delta ** 2)
                v_ += -adversary1.predict(D).flatten()/ (alpha * delta ** 2) - v if not subsetted else -adversary1.predict(D).flatten() * subset_ind1 / (alpha * delta ** 2) - v
            learnersg.append(self._get_new_learnerg().fit(A, v_))
            learnersh.append(self._get_new_learnerh().fit(B, v))
            g = g * it / (it + 1)
            h = h * it / (it + 1)
            g += learnersg[it].predict(A).flatten() / (it + 1)
            h += learnersh[it].predict(B).flatten() / (it + 1)
            adversary2.fit(C, h - g*W)
            adversary1.fit(D, g - Y)

        self.learnersg = learnersg[self.n_burn_in:]
        self.learnersh = learnersh[self.n_burn_in:]
        return self

    def predict(self, B, *args):
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
