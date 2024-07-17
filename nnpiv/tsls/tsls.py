"""
This module provides implementations of two-stage least squares (TSLS) and regularized TSLS using linear and elastic net regression.

Classes:
    tsls: Two-stage least squares estimator.
    regtsls: Regularized two-stage least squares estimator using Elastic Net.
"""
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, LinearRegression, \
    ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV
import numpy as np
        
class tsls:
    """
    Two-stage least squares estimator.

    This class implements the TSLS estimator.
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, Z, T, Y):
        """
        Fit the TSLS estimator.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        first = LinearRegression()
        first.fit(Z, T)
        predicted_T = first.predict(Z)
        second = LinearRegression()
        second.fit(predicted_T, Y)
        self.coef_ = second.coef_
        self.intercept_ = second.intercept_
        return self
    
    def predict(self, T):
        """
        Predict outcomes based on the fitted model.

        Parameters:
            T (array-like): Treatments.

        Returns:
            array-like: Predicted outcomes.
        """
        return T @ self.coef_.T + self.intercept_

class regtsls:
    """
    Regularized two-stage least squares estimator using Elastic Net.

    This class implements the regularized TSLS estimator using Elastic Net regression.
    """

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, Z, T, Y):
        """
        Fit the regularized TSLS estimator.

        Parameters:
            Z (array-like): Instrumental variables.
            T (array-like): Treatments.
            Y (array-like): Outcomes.

        Returns:
            self: Fitted estimator.
        """
        first = MultiTaskElasticNetCV(cv=3)
        first.fit(Z, T)
        predicted_T = first.predict(Z)
        second = ElasticNetCV(cv=3)
        second.fit(predicted_T, Y.ravel())
        self.coef_ = second.coef_
        self.intercept_ = second.intercept_
        return self
    
    def predict(self, T):
        """
        Predict outcomes based on the fitted model.

        Parameters:
            T (array-like): Treatments.

        Returns:
            array-like: Predicted outcomes.
        """
        yhat = T @ self.coef_.T + self.intercept_
        return yhat.reshape(-1, 1)
