from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, LinearRegression,\
    ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV
import numpy as np
        
class tsls:  
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, Z, T, Y):
        first = LinearRegression()
        first.fit(Z, T)
        predicted_T = first.predict(Z)
        second = LinearRegression()
        second.fit(predicted_T, Y)  
        self.coef_ = second.coef_
        self.intercept_ = second.intercept_
        return self
    
    def predict(self, T):
        return T @ self.coef_.T   + self.intercept_

class regtsls:  
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, Z, T, Y):
        first = MultiTaskElasticNetCV(cv=3)
        first.fit(Z, T)
        predicted_T = first.predict(Z)
        second = ElasticNetCV(cv=3)
        second.fit(predicted_T, Y.ravel())  
        self.coef_ = second.coef_
        self.intercept_ = second.intercept_
        return self
    
    def predict(self, T):
        yhat = T @ self.coef_.T   + self.intercept_
        return yhat.reshape(-1,1)        

class exptsls:  
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, Z, T, Y):
        first = LinearRegression()
        first.fit(Z, T)
        predicted_T = first.predict(Z)
        second = LinearRegression()
        logY = np.log(Y-1)
        second.fit(predicted_T, logY.ravel())  
        self.coef_ = second.coef_
        self.intercept_ = second.intercept_
        return self
    
    def predict(self, T):
        return np.exp(T @ self.coef_.T   + self.intercept_) + 1      
