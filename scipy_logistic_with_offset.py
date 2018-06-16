import os
import numpy as np
import scipy.sparse as sp
from itertools import product
from scipy.optimize import fmin_l_bfgs_b

def sigmoid(x):
    return 1./(1. + np.exp(-x))

class LogisticWithOffset():
    ''' Logistic regression with extra functionality that allows for an offset in the linear index
    required for orthogonal estimation of treatment effects with non-linear link functions
    '''
    def __init__(self, steps=1000, alpha_l1=0.1, alpha_l2=0.1, learning_rate=0.1, fit_intercept=False,
                    learning_schedule='constant', tol=1e-6, batch_size=50):
        ''' Initialize parameters
        steps: number of gradient steps
        alpha_l1: weight of l1 regularization
        alpha_l2: weight of l2 regularization
        learning_rate: step size of each gradient step
        fit_intercept: whether to fit a bias term
        learning_schedule: 'constant' or 'decay', whether to decay or not the learning rate
        tol: early stopping if cost decrease is smaller than tol
        batch_size: samples used at each gradient step, capped at sample size if more than samples
        '''
        self._alpha_l1 = alpha_l1
        self._alpha_l2 = alpha_l2
        self._tol = tol
        self._coef = None
    
    def fit(self, X, y, offset=None, sample_weights=None):
        ''' Fits a logistic model with offset and weights using SGD
        '''
        n_samples, n_features = X.shape

        if offset is None:
            offset = np.zeros((n_samples, 1))
        if sample_weights is None:
            sample_weights = np.ones((n_samples, 1))
        
        def loss_and_jac(extended_coef):
            coef = extended_coef[:n_features] - extended_coef[n_features:]
            index = np.dot(X, coef.reshape(-1, 1))
            y_pred = sigmoid(index + offset)
            m_loss = - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
            loss = np.mean(sample_weights * m_loss) + self._alpha_l1 * np.sum(extended_coef) + 0.5 * self._alpha_l2 * np.sum(extended_coef**2)
            moment = (y_pred - y) * X
            grad = np.mean(sample_weights * moment, axis=0).flatten() 
            jac = np.concatenate([grad, -grad]) + self._alpha_l1 + self._alpha_l2 * extended_coef
            return loss, jac
        
        w, _, _ = fmin_l_bfgs_b(loss_and_jac, np.zeros(2*n_features), 
                                bounds=[(0, None)] * n_features * 2,
                                pgtol=self._tol)

        self._coef = w[:n_features] - w[n_features:]

        return self

    @property
    def coef_(self):
        return self._coef
    
    def predict_proba(self, X):
        y_pred = sigmoid(np.dot(X, self.coef_.reshape(-1, 1)))
        return np.concatenate((1 - y_pred, y_pred), axis=1)
    
    def predict(self, X):
        return sigmoid(np.dot(X, self.coef_.reshape(-1, 1))) >= 0.5
    
    
    def score(self, X, y_true):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, self.predict_proba(X)[:, 1].reshape(y_true.shape))

    def accuracy(self, X, y_true):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, self.predict(X).reshape(y_true.shape))
    
    