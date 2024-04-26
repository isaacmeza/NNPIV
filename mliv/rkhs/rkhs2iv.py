# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold
from sklearn.kernel_approximation import Nystroem, RBFSampler
import numpy as np
import scipy


def _check_auto(param):
    return (isinstance(param, str) and (param == 'auto'))


class _BaseRKHS2IV:

    def __init__(self, *args, **kwargs):
        return

    def _get_delta(self, n):
        '''
        delta -> Critical radius
        '''
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
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)


class RKHS2IV(_BaseRKHS2IV):

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1,
                 delta_scale='auto', delta_exp='auto', kernel_params=None):
        """
        Parameters:
            kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma : the gamma parameter for the kernel
            degree : the degree of a polynomial kernel
            coef0 : the zero coef for a polynomial kernel
            kernel_params : other kernel params passed to the kernel
        """
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp

    def fit(self, A, B, C, D, Y):
        n = Y.shape[0]  # number of samples
        delta = self._get_delta(n)
        alpha = delta**4

        Ka = self._get_kernel(A)
        Kb = self._get_kernel(B)
        Kc = self._get_kernel(C)
        Kd = self._get_kernel(D)

        Id = np.eye(n)
        Pc = np.linalg.pinv(Kc / n + Id) @ Kc
        Pd = np.linalg.pinv(Kd / n + Id) @ Kd

        KbPcKa_inv = np.linalg.pinv(Kb @ Pc @ Ka)

        M = Ka @ (Pc + (Pd @ Ka + Pc @ Ka + alpha * Id) @ KbPcKa_inv @ (Kb @ Pc + alpha * Id)) @ Kb
        
        self.b = np.linalg.pinv(M) @ Ka @ Pd @ Y
        self.a = KbPcKa_inv @ (Kb @ Pc + alpha * Id) @ Kb @ self.b

        self.A = A.copy()
        self.B = B.copy()
        return self

    def predict(self, B_test, *args):
        if len(args) == 0:
            # Only B_test provided, return h prediction
            return self._get_kernel(B_test, Y=self.B) @ self.b
        elif len(args) == 1:
            # Two arguments provided, assume the second is A_test
            A_test = args[0]
            return (self._get_kernel(B_test, Y=self.B) @ self.b, self._get_kernel(A_test, Y=self.A) @ self.a)
        else:
            # More than one additional argument provided, raise an error
            raise ValueError("predict expects at most two arguments, B_test and optionally A_test")


class RKHS2IVCV(RKHS2IV):

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Parameters:
            kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma : the gamma parameter for the kernel
            degree : the degree of a polynomial kernel
            coef0 : the zero coef for a polynomia kernel
            kernel_params : other kernel params passed to the kernel
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**2)
            n_alphas : how mny alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale
        """
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  
        self.n_alphas = n_alphas
        self.cv = cv


    def fit(self, A, B, C, D, Y):
        n = Y.shape[0]

        Ka = self._get_kernel(A)
        Kb = self._get_kernel(B)
        Kc = self._get_kernel(C)
        Kd = self._get_kernel(D)

        Id = np.eye(n)
        Pc = np.linalg.pinv(Kc / n + Id) @ Kc
        Pd = np.linalg.pinv(Kd / n + Id) @ Kd

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta = self._get_delta(n)
        scores = []
        for it, (train, test) in enumerate(KFold(n_splits=self.cv).split(Y)):
            Ka_train = Ka[np.ix_(train, train)]
            Kb_train = Kb[np.ix_(train, train)]
            Id_train = np.eye(len(train))
            Id_test = np.eye(len(test))
            Pc_train = np.linalg.pinv(Kc[np.ix_(train, train)] / n_train + Id_train) @ Kc[np.ix_(train, train)]
            Pd_train = np.linalg.pinv(Kd[np.ix_(train, train)] / n_train + Id_train) @ Kd[np.ix_(train, train)]
            Pc_test = np.linalg.pinv(Kc[np.ix_(test, test)] / n_test + Id_test) @ Kc[np.ix_(test, test)]
            Pd_test = np.linalg.pinv(Kd[np.ix_(test, test)] / n_test + Id_test) @ Kd[np.ix_(test, test)]
            
            KbPcKa_inv = np.linalg.pinv(Kb_train @ Pc_train @ Ka_train)
            B_train = Ka_train @ Pd_train @ Y[train]
            C_train = KbPcKa_inv @ Kb_train 

            scores.append([])
            for alpha_scale in alpha_scales:
                alpha = alpha_scale * delta_train**4
                M = Ka_train @ (Pc_train + (Pd_train @ Ka_train + Pc_train @ Ka_train + alpha * Id_train) @ KbPcKa_inv @ (Kb_train @ Pc_train + alpha * Id_train)) @ Kb_train
                b = np.linalg.pinv(M) @ B_train
                a = KbPcKa_inv @ (Kb_train @ Pc_train + alpha * Id_train) @ Kb_train @ b
                res1 = Y[test] - Ka[np.ix_(test, train)] @ a
                res2 = Ka[np.ix_(test, train)] @ a - Kb[np.ix_(test, train)] @ b
                scores[it].append((res1.T @ Pd_test @ res1)[
                                  0, 0] / (res1.shape[0]**2) 
                                  + (res2.T @ Pc_test @ res2)[
                                  0, 0] / (res2.shape[0]**2) )

        self.alpha_scales = alpha_scales
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = alpha_scales[np.argmin(self.avg_scores)]

        delta = self._get_delta(n)
        self.best_alpha = self.best_alpha_scale * delta**4

        KbPcKa_inv = np.linalg.pinv(Kb @ Pc @ Ka)
        M = Ka @ (Pc + (Pd @ Ka + Pc @ Ka + self.best_alpha * Id) @ KbPcKa_inv @ (Kb @ Pc + self.best_alpha * Id)) @ Kb
        
        self.b = np.linalg.pinv(M) @ Ka @ Pd @ Y
        self.a = KbPcKa_inv @ (Kb @ Pc + self.best_alpha * Id) @ Kb @ self.b

        self.A = A.copy()
        self.B = B.copy()

        return self
    

class RKHS2IVL2(_BaseRKHS2IV):

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1,
                 delta_scale='auto', delta_exp='auto', kernel_params=None):
        """
        Parameters:
            kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma : the gamma parameter for the kernel
            degree : the degree of a polynomial kernel
            coef0 : the zero coef for a polynomial kernel
            kernel_params : other kernel params passed to the kernel
        """
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp

    def fit(self, A, B, C, D, Y):
        n = Y.shape[0]  # number of samples
        delta = self._get_delta(n)
        alpha = delta**4

        Ka = self._get_kernel(A)
        Kb = self._get_kernel(B)
        Kc = self._get_kernel(C)
        Kd = self._get_kernel(D)

        Id = np.eye(n)
        Pc = np.linalg.pinv(Kc / n) @ Kc
        Pd = np.linalg.pinv(Kd / n) @ Kd

        KbPcKa_inv = np.linalg.pinv(Kb @ Pc @ Ka)

        M = Ka @ (Pc + (Pd + Pc + alpha * Id) @ Ka @ KbPcKa_inv @ Kb @ (Pc + alpha * Id)) @ Kb
        
        self.b = np.linalg.pinv(M) @ Ka @ Pd @ Y
        self.a = KbPcKa_inv @ Kb @ (Pc + alpha * Id) @ Kb @ self.b

        self.A = A.copy()
        self.B = B.copy()
        return self

    def predict(self, B_test, *args):
        if len(args) == 0:
            # Only B_test provided, return h prediction
            return self._get_kernel(B_test, Y=self.B) @ self.b
        elif len(args) == 1:
            # Two arguments provided, assume the second is A_test
            A_test = args[0]
            return (self._get_kernel(B_test, Y=self.B) @ self.b, self._get_kernel(A_test, Y=self.A) @ self.a)
        else:
            # More than one additional argument provided, raise an error
            raise ValueError("predict expects at most two arguments, B_test and optionally A_test")


class RKHS2IVL2CV(RKHS2IVL2):

    def __init__(self, kernel='rbf', gamma=2, degree=3, coef0=1, kernel_params=None,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Parameters:
            kernel : a pairwise kernel function or a string; similar interface with KernelRidge in sklearn
            gamma : the gamma parameter for the kernel
            degree : the degree of a polynomial kernel
            coef0 : the zero coef for a polynomia kernel
            kernel_params : other kernel params passed to the kernel
            delta_scale : the scale of the critical radius; delta_n = delta_scal / n**(delta_exp)
            delta_exp : the exponent of the cirical radius; delta_n = delta_scal / n**(delta_exp)
            alpha_scales : a list of scale of the regularization to choose from; alpha = alpha_scale * (delta**2)
            n_alphas : how mny alpha_scales to try
            cv : how many folds to use in cross-validation for alpha_scale
        """
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.kernel_params = kernel_params
        self.delta_scale = delta_scale  # worst-case critical value of RKHS spaces
        self.delta_exp = delta_exp  # worst-case critical value of RKHS spaces
        self.alpha_scales = alpha_scales  
        self.n_alphas = n_alphas
        self.cv = cv


    def fit(self, A, B, C, D, Y):
        n = Y.shape[0]

        Ka = self._get_kernel(A)
        Kb = self._get_kernel(B)
        Kc = self._get_kernel(C)
        Kd = self._get_kernel(D)

        Id = np.eye(n)
        Pc = np.linalg.pinv(Kc / n) @ Kc
        Pd = np.linalg.pinv(Kd / n) @ Kd

        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta = self._get_delta(n)
        scores = []
        for it, (train, test) in enumerate(KFold(n_splits=self.cv).split(Y)):
            Ka_train = Ka[np.ix_(train, train)]
            Kb_train = Kb[np.ix_(train, train)]
            Id_train = np.eye(len(train))
            Pc_train = np.linalg.pinv(Kc[np.ix_(train, train)] / n_train) @ Kc[np.ix_(train, train)]
            Pd_train = np.linalg.pinv(Kd[np.ix_(train, train)] / n_train) @ Kd[np.ix_(train, train)]
            Pc_test = np.linalg.pinv(Kc[np.ix_(test, test)] / n_test) @ Kc[np.ix_(test, test)]
            Pd_test = np.linalg.pinv(Kd[np.ix_(test, test)] / n_test) @ Kd[np.ix_(test, test)]
            
            KbPcKa_inv = np.linalg.pinv(Kb_train @ Pc_train @ Ka_train)
            W = Ka_train @ KbPcKa_inv @ Kb_train
            B_train = Ka_train @ Pd_train @ Y[train]
            C_train = KbPcKa_inv @ Kb_train 

            scores.append([])
            for alpha_scale in alpha_scales:
                alpha = alpha_scale * delta_train**4
                M = Ka_train @ (Pc_train + (Pd_train + Pc_train + alpha * Id_train) @ W @ (Pc_train + alpha * Id_train)) @ Kb_train
                b = np.linalg.pinv(M) @ B_train
                a = C_train @ (Pc_train + alpha * Id_train) @ Kb_train @ b
                res1 = Y[test] - Ka[np.ix_(test, train)] @ a
                res2 = Ka[np.ix_(test, train)] @ a - Kb[np.ix_(test, train)] @ b
                scores[it].append((res1.T @ Pd_test @ res1)[
                                  0, 0] / (res1.shape[0]**2) 
                                  + (res2.T @ Pc_test @ res2)[
                                  0, 0] / (res2.shape[0]**2) )

        self.alpha_scales = alpha_scales
        self.avg_scores = np.mean(np.array(scores), axis=0)
        self.best_alpha_scale = alpha_scales[np.argmin(self.avg_scores)]

        delta = self._get_delta(n)
        self.best_alpha = self.best_alpha_scale * delta**4

        KbPcKa_inv = np.linalg.pinv(Kb @ Pc @ Ka)
        M = Ka @ (Pc + (Pd + Pc + self.best_alpha * Id) @ Ka @ KbPcKa_inv @ Kb @ (Pc + self.best_alpha * Id)) @ Kb
        
        self.b = np.linalg.pinv(M) @ Ka @ Pd @ Y
        self.a = KbPcKa_inv @ Kb @ (Pc + self.best_alpha * Id) @ Kb @ self.b

        self.A = A.copy()
        self.B = B.copy()

        return self
    
