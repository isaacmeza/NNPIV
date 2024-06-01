# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from .utilities import cross_product



class _SparseLinear2AdversarialGMM:

    def __init__(self, mu=0.01, V1=100, V2=100,
                 eta_alpha='auto', eta_w1='auto', eta_beta='auto', eta_w2='auto',
                 n_iter=2000, tol=1e-2, sparsity=None, fit_intercept=True):
        self.V1 = V1
        self.V2 = V2
        self.mu = mu
        self.eta_alpha = eta_alpha
        self.eta_w1 = eta_w1
        self.eta_beta = eta_beta
        self.eta_w2 = eta_w2
        self.n_iter = n_iter
        self.tol = tol
        self.sparsity = sparsity
        self.fit_intercept = fit_intercept

    def weighted_mean(self, arr, weights, axis=0):
        # Ensure weights is an array
        weights = np.array(weights)

        if arr.ndim == 1 or axis is None:
            # For 1D arrays or when axis is None, no need to expand dimensions
            return np.sum(arr * weights) / np.sum(weights)
        else:
            # For multi-dimensional arrays, expand weights along the specified axis
            return np.sum(arr * weights[:, np.newaxis], axis=axis) / np.sum(weights)

    def _check_input(self, A, B, C, D, Y):
        if self.fit_intercept:
            A = np.hstack([np.ones((A.shape[0], 1)), A])
            B = np.hstack([np.ones((B.shape[0], 1)), B])
            C = np.hstack([np.ones((C.shape[0], 1)), C])
            D = np.hstack([np.ones((D.shape[0], 1)), D])
        return A, B, C, D, Y.flatten()

    def predict(self, B, *args):
        if len(args) == 0:
            if self.fit_intercept:
                B = np.hstack([np.ones((B.shape[0], 1)), B])
            return np.dot(B, self.beta_)
        elif len(args) == 1:
            # Two arguments provided, assume the second is A
            A = args[0]
            if self.fit_intercept:
                B = np.hstack([np.ones((B.shape[0], 1)), B])
                A = np.hstack([np.ones((A.shape[0], 1)), A])
            return (np.dot(B, self.beta_) , np.dot(A, self.alpha_))
        else:
            # More than one additional argument provided, raise an error
            raise ValueError("predict expects at most two arguments, B_test and optionally A_test")

    @property
    def coef(self):
        return self.beta_[1:] if self.fit_intercept else self.beta_

    @property
    def intercept(self):
        return self.beta_[0] if self.fit_intercept else 0


class sparse2_l2vsl2(_SparseLinear2AdversarialGMM):

    def _check_duality_gap(self, A, B, C, D, Y):
        self.max_response_loss_ = np.linalg.norm(self.weighted_mean(D * (Y - np.dot(A, self.alpha_)).reshape(-1, 1), self.weights1, axis=0), ord=2)\
            + np.linalg.norm(self.weighted_mean(C * (np.dot(A, self.alpha_) - np.dot(B, self.beta_)).reshape(-1, 1), self.weights2, axis=0), ord=2)\
            + self.mu * np.linalg.norm(self.alpha_, ord=2)**2 + self.mu * np.linalg.norm(self.beta_, ord=2)**2
            
        self.min_response_loss_ = self.weighted_mean(Y * np.dot(D, self.w1_), self.weights1)\
            + self.V1 * np.clip(self.mu - 2 * np.linalg.norm(self.weighted_mean(A * np.dot(D, self.w1_).reshape(-1, 1), self.weights1,
                                                                            axis=0),
                                                                    ord=2)
                                        + 2 * np.linalg.norm(self.weighted_mean(A * np.dot(C, self.w2_).reshape(-1, 1), self.weights2,
                                                                axis=0),
                                                            ord=2),
                                                   -np.inf, 0)\
            + self.V2 * np.clip(self.mu - 2 * np.linalg.norm(self.weighted_mean(B * np.dot(C, self.w2_).reshape(-1, 1), self.weights2,
                                                                axis=0),
                                                            ord=2),
                                                   -np.inf, 0)
        
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, A, B, C, D, Y):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (A.shape[0])**(2 / 3))
            filt = (np.abs(self.alpha_) < thresh)
            self.alpha_[filt] = 0
            thresh = 1 / (self.sparsity * (B.shape[0])**(2 / 3))
            filt = (np.abs(self.beta_) < thresh)
            self.beta_[filt] = 0
        self._check_duality_gap(A, B, C, D, Y)

    def fit(self, A, B, C, D, Y, subsetted=False, subset_ind1=None, subset_ind2=None):
        A, B, C, D, Y = self._check_input(A, B, C, D, Y) 
        self.weights1 = np.ones(Y.shape[0])
        self.weights2 = np.ones(Y.shape[0])
        if subsetted:
            if subset_ind1 is None:
                raise ValueError("subset_ind1 must be provided when subsetted is True")
            if len(subset_ind1) != len(Y):
                raise ValueError("subset_ind1 must have the same length as Y")
            ind1 = np.where(subset_ind1==0)[0] 
            ind2 = np.where(subset_ind2==0)[0] if subset_ind2 is not None else np.where(subset_ind1==1)[0]  
            self.weights1[ind1] = 0
            self.weights2[ind2] = 0

        T = self.n_iter
        d_a = A.shape[1]
        d_b = B.shape[1]
        d_c = C.shape[1]
        d_d = D.shape[1]
        n = A.shape[0]
        V1 = self.V1
        V2 = self.V2
        eta_alpha = np.sqrt(
            np.log(d_a + 1) / T) if self.eta_alpha == 'auto' else self.eta_alpha
        eta_beta = np.sqrt(
            np.log(d_b + 1) / T) if self.eta_beta == 'auto' else self.eta_beta
        eta_w1 = np.sqrt(
            np.log(d_d + 1) / T) if self.eta_w1 == 'auto' else self.eta_w1
        eta_w2 = np.sqrt(
            np.log(d_c + 1) / T) if self.eta_w2 == 'auto' else self.eta_w2
        mu = self.mu

        yd = self.weighted_mean(Y.reshape(-1, 1) * D, self.weights1, axis=0)
        if d_a * d_d < n**2:
            ad = self.weighted_mean(cross_product(A, D), self.weights1,
                         axis=0).reshape(d_d, d_a).T
        if d_a * d_c < n**2:
            ac = self.weighted_mean(cross_product(A, C), self.weights2,
                         axis=0).reshape(d_c, d_a).T
        if d_b * d_c < n**2:
            bc = self.weighted_mean(cross_product(B, C), self.weights2,
                         axis=0).reshape(d_c, d_b).T
        
        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                alpha = np.zeros(d_a)
                beta = np.zeros(d_b)
                w1 = np.zeros(d_d)
                w2 = np.zeros(d_c)
                alpha_acc = np.zeros(d_a)
                beta_acc = np.zeros(d_b)
                w1_acc = np.zeros(d_d)
                w2_acc = np.zeros(d_c)
                res1 = np.zeros(d_d)
                res2 = np.zeros(d_c)
                res1_pre = np.zeros(d_d)
                res2_pre = np.zeros(d_c)
                cors1 = np.zeros(d_a)
                cors2 = np.zeros(d_b)   
                cors1_pre = np.zeros(d_a)
                cors2_pre = np.zeros(d_b)

            # quantities for updating alpha
            if d_a * d_d < n**2:
                cors1[:] = - ad @ w1 + mu * alpha
            else:
                test_fn = np.dot(D, w1).reshape(-1, 1)
                cors1[:] = - self.weighted_mean(test_fn * A, self.weights1, axis=0) + mu * alpha
            if d_a * d_c < n**2:
                cors1[:] += ac @ w2
            else:
                test_fn = np.dot(C, w2).reshape(-1, 1)
                cors1[:] += self.weighted_mean(test_fn * A, self.weights2, axis=0)

            # quantities for updating beta
            if d_b * d_c < n**2:
                cors2[:] = - bc @ w2 + mu * beta
            else:
                test_fn = np.dot(C, w2).reshape(-1, 1)
                cors2[:] = - self.weighted_mean(test_fn * B, self.weights2, axis=0) + mu * beta

            # quantities for updating w1
            if d_a * d_d < n**2:
                res1[:] = yd - alpha.T @ ad
            else:
                pred_fn = np.dot(A, alpha).reshape(-1, 1)
                res1[:] = yd - self.weighted_mean(D * pred_fn, self.weights1, axis=0) 

            # quantities for updating w2
            if d_c * d_a < n**2:
                res2[:] = alpha.T @ ac 
            else:
                pred_fn = np.dot(A, alpha).reshape(-1, 1)
                res2[:] = self.weighted_mean(C * pred_fn, self.weights2, axis=0)
            if d_c * d_b < n**2:
                res2[:] -= beta.T @ bc
            else:
                pred_fn = np.dot(B, beta).reshape(-1, 1)
                res2[:] -= self.weighted_mean(C * pred_fn, self.weights2, axis=0)

            # update alpha
            alpha[:] = alpha - 2 * eta_alpha * cors1 + eta_alpha * cors1_pre
            normalization = np.linalg.norm(alpha, ord=2)
            if normalization > V1:
                alpha[:] = alpha * V1 / normalization

            # update beta
            beta[:] = beta - 2 * eta_beta * cors2 + eta_beta * cors2_pre
            normalization = np.linalg.norm(beta, ord=2)
            if normalization > V2:
                beta[:] = beta * V2 / normalization

            # update w1
            w1[:] = w1 + 2 * eta_w1 * res1 - eta_w1 * res1_pre
            norm_w1 = np.linalg.norm(w1, ord=2)
            w1[:] = w1 / norm_w1 if norm_w1 > 1 else w1

            # update w2
            w2[:] = w2 + 2 * eta_w2 * res2 - eta_w2 * res2_pre
            norm_w2 = np.linalg.norm(w2, ord=2)
            w2[:] = w2 / norm_w2 if norm_w2 > 1 else w2

            alpha_acc = alpha_acc * (t - 1) / t + alpha / t
            beta_acc = beta_acc * (t - 1) / t + beta / t
            w1_acc = w1_acc * (t - 1) / t + w1 / t
            w2_acc = w2_acc * (t - 1) / t + w2 / t

            res1_pre[:] = res1
            res2_pre[:] = res2
            cors1_pre[:] = cors1
            cors2_pre[:] = cors2

            if t % 50 == 0:
                self.alpha_ = alpha_acc
                self.beta_ = beta_acc
                self.w1_ = w1_acc
                self.w2_ = w2_acc
                if self._check_duality_gap(A, B, C, D, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_alpha /= 2
                    eta_beta /= 2
                    eta_w1 /= 2
                    eta_w2 /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_alpha /= 1.01
                    eta_beta /= 1.01
                    eta_w1 /= 1.01
                    eta_w2 /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.alpha_ = alpha_acc
        self.beta_ = beta_acc
        self.w1_ = w1_acc
        self.w2_ = w2_acc
        
        self._post_process(A, B, C, D, Y)
        return self


class sparse2_ridge_l2vsl2(_SparseLinear2AdversarialGMM):

    def _check_duality_gap(self, A, B, C, D, Y):
        self.max_response_loss_ = np.linalg.norm(self.weighted_mean(D * (Y - np.dot(A, self.alpha_)).reshape(-1, 1), self.weights1, axis=0), ord=2)\
            + np.linalg.norm(self.weighted_mean(C * (np.dot(A, self.alpha_) - np.dot(B, self.beta_)).reshape(-1, 1), self.weights2, axis=0), ord=2)\
            + self.mu * self.alpha_.T @ self.aa @ self.alpha_ + self.mu * self.beta_.T @ self.bb @ self.beta_
            
        self.min_response_loss_ = 2 * self.weighted_mean(Y * np.dot(D, self.w1_), self.weights1)\
            - (self.msvp_a/self.mu) * np.linalg.norm(self.weighted_mean(A * np.dot(D, self.w1_).reshape(-1, 1), self.weights1,
                                                            axis=0)
                                                    - self.weighted_mean(A * np.dot(C, self.w2_).reshape(-1, 1), self.weights2,
                                                            axis=0),
                                                    ord=2)\
            - (self.msvp_b/self.mu) * np.linalg.norm(self.weighted_mean(B * np.dot(C, self.w2_).reshape(-1, 1), self.weights2,
                                                            axis=0),
                                                    ord=2)
        
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, A, B, C, D, Y):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (A.shape[0])**(2 / 3))
            filt = (np.abs(self.alpha_) < thresh)
            self.alpha_[filt] = 0
            thresh = 1 / (self.sparsity * (B.shape[0])**(2 / 3))
            filt = (np.abs(self.beta_) < thresh)
            self.beta_[filt] = 0
        self._check_duality_gap(A, B, C, D, Y)

    def fit(self, A, B, C, D, Y, subsetted=False, subset_ind1=None, subset_ind2=None):
        A, B, C, D, Y = self._check_input(A, B, C, D, Y) 
        self.weights1 = np.ones(Y.shape[0])
        self.weights2 = np.ones(Y.shape[0])
        if subsetted:
            if subset_ind1 is None:
                raise ValueError("subset_ind1 must be provided when subsetted is True")
            if len(subset_ind1) != len(Y):
                raise ValueError("subset_ind1 must have the same length as Y")
            ind1 = np.where(subset_ind1==0)[0] 
            ind2 = np.where(subset_ind2==0)[0] if subset_ind2 is not None else np.where(subset_ind1==1)[0]  
            self.weights1[ind1] = 0
            self.weights2[ind2] = 0

        T = self.n_iter
        d_a = A.shape[1]
        d_b = B.shape[1]
        d_c = C.shape[1]
        d_d = D.shape[1]
        n = A.shape[0]
        V1 = self.V1
        V2 = self.V2
        eta_alpha = np.sqrt(
            np.log(d_a + 1) / T) if self.eta_alpha == 'auto' else self.eta_alpha
        eta_beta = np.sqrt(
            np.log(d_b + 1) / T) if self.eta_beta == 'auto' else self.eta_beta
        eta_w1 = np.sqrt(
            np.log(d_d + 1) / T) if self.eta_w1 == 'auto' else self.eta_w1
        eta_w2 = np.sqrt(
            np.log(d_c + 1) / T) if self.eta_w2 == 'auto' else self.eta_w2
        mu = self.mu

        yd = self.weighted_mean(Y.reshape(-1, 1) * D, self.weights1, axis=0)
        aa = np.mean(cross_product(A, A), axis=0).reshape(d_a, d_a).T
        self.aa = aa
        Sigma = np.linalg.svd(aa, compute_uv=False)
        sigma_min = np.min(Sigma[Sigma > 1e-10])  
        self.msvp_a = 1 / sigma_min

        bb = np.mean(cross_product(B, B), axis=0).reshape(d_b, d_b).T
        self.bb = bb
        Sigma = np.linalg.svd(bb, compute_uv=False)
        sigma_min = np.min(Sigma[Sigma > 1e-10])
        self.msvp_b = 1 / sigma_min

        if d_a * d_d < n**2:
            ad = self.weighted_mean(cross_product(A, D), self.weights1,
                         axis=0).reshape(d_d, d_a).T
        if d_a * d_c < n**2:
            ac = self.weighted_mean(cross_product(A, C), self.weights2,
                         axis=0).reshape(d_c, d_a).T
        if d_b * d_c < n**2:
            bc = self.weighted_mean(cross_product(B, C), self.weights2,
                         axis=0).reshape(d_c, d_b).T
        
        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                alpha = np.zeros(d_a)
                beta = np.zeros(d_b)
                w1 = np.zeros(d_d)
                w2 = np.zeros(d_c)
                alpha_acc = np.zeros(d_a)
                beta_acc = np.zeros(d_b)
                w1_acc = np.zeros(d_d)
                w2_acc = np.zeros(d_c)
                res1 = np.zeros(d_d)
                res2 = np.zeros(d_c)
                res1_pre = np.zeros(d_d)
                res2_pre = np.zeros(d_c)
                cors1 = np.zeros(d_a)
                cors2 = np.zeros(d_b)   
                cors1_pre = np.zeros(d_a)
                cors2_pre = np.zeros(d_b)

            # quantities for updating alpha
            if d_a * d_d < n**2:
                cors1[:] = - ad @ w1 + mu * aa @ alpha
            else:
                test_fn = np.dot(D, w1).reshape(-1, 1)
                cors1[:] = - self.weighted_mean(test_fn * A, self.weights1, axis=0) + mu * aa @ alpha
            if d_a * d_c < n**2:
                cors1[:] += ac @ w2
            else:
                test_fn = np.dot(C, w2).reshape(-1, 1)
                cors1[:] += self.weighted_mean(test_fn * A, self.weights2, axis=0)

            # quantities for updating beta
            if d_b * d_c < n**2:
                cors2[:] = - bc @ w2 + mu * bb @ beta
            else:
                test_fn = np.dot(C, w2).reshape(-1, 1)
                cors2[:] = - self.weighted_mean(test_fn * B, self.weights2, axis=0) + mu * bb @ beta

            # quantities for updating w1
            if d_a * d_d < n**2:
                res1[:] = yd - alpha.T @ ad
            else:
                pred_fn = np.dot(A, alpha).reshape(-1, 1)
                res1[:] = yd - self.weighted_mean(D * pred_fn, self.weights1, axis=0) 

            # quantities for updating w2
            if d_c * d_a < n**2:
                res2[:] = alpha.T @ ac 
            else:
                pred_fn = np.dot(A, alpha).reshape(-1, 1)
                res2[:] = self.weighted_mean(C * pred_fn, self.weights2, axis=0)
            if d_c * d_b < n**2:
                res2[:] -= beta.T @ bc
            else:
                pred_fn = np.dot(B, beta).reshape(-1, 1)
                res2[:] -= self.weighted_mean(C * pred_fn, self.weights2, axis=0)

            # update alpha
            alpha[:] = alpha - 2 * eta_alpha * cors1 + eta_alpha * cors1_pre
            normalization = np.linalg.norm(alpha, ord=2)
            if normalization > V1:
                alpha[:] = alpha * V1 / normalization

            # update beta
            beta[:] = beta - 2 * eta_beta * cors2 + eta_beta * cors2_pre
            normalization = np.linalg.norm(beta, ord=2)
            if normalization > V2:
                beta[:] = beta * V2 / normalization

            # update w1
            w1[:] = w1 + 2 * eta_w1 * res1 - eta_w1 * res1_pre
            norm_w1 = np.linalg.norm(w1, ord=2)
            w1[:] = w1 / norm_w1 if norm_w1 > 1 else w1

            # update w2
            w2[:] = w2 + 2 * eta_w2 * res2 - eta_w2 * res2_pre
            norm_w2 = np.linalg.norm(w2, ord=2)
            w2[:] = w2 / norm_w2 if norm_w2 > 1 else w2

            alpha_acc = alpha_acc * (t - 1) / t + alpha / t
            beta_acc = beta_acc * (t - 1) / t + beta / t
            w1_acc = w1_acc * (t - 1) / t + w1 / t
            w2_acc = w2_acc * (t - 1) / t + w2 / t

            res1_pre[:] = res1
            res2_pre[:] = res2
            cors1_pre[:] = cors1
            cors2_pre[:] = cors2

            if t % 50 == 0:
                self.alpha_ = alpha_acc
                self.beta_ = beta_acc
                self.w1_ = w1_acc
                self.w2_ = w2_acc
                if self._check_duality_gap(A, B, C, D, Y):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_alpha /= 2
                    eta_beta /= 2
                    eta_w1 /= 2
                    eta_w2 /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_alpha /= 1.01
                    eta_beta /= 1.01
                    eta_w1 /= 1.01
                    eta_w2 /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.alpha_ = alpha_acc
        self.beta_ = beta_acc
        self.w1_ = w1_acc
        self.w2_ = w2_acc
        
        self._post_process(A, B, C, D, Y)
        return self