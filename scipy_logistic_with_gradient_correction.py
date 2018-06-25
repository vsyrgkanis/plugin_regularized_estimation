import os
import numpy as np
import scipy.sparse as sp
from itertools import product
from scipy.optimize import fmin_l_bfgs_b

def sigmoid(x):
    return 1./(1. + np.exp(-x))

class LogisticWithGradientCorrection():
    def __init__(self, steps=1000, alpha_l1=0.1, alpha_l2=0.1, learning_rate=0.1, tol=1e-6,
                        learning_schedule='constant', batch_size=50):
        self._alpha_l1 = alpha_l1
        self._alpha_l2 = alpha_l2
        self._tol = tol
        self._coef = None

    def fit(self, X, y, offset=None, grad_corrections=None, sample_weights=None):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        if offset is None:
            offset = np.zeros((X.shape[0], 1))
        offset = offset.reshape(-1, 1)
        if grad_corrections is None:
            grad_corrections = np.zeros((X.shape[0], 1))
        grad_corrections = grad_corrections.reshape(-1, 1)
        if sample_weights is None:
            sample_weights = 1. * np.ones((X.shape[0], 1))
        sample_weights = sample_weights.reshape(-1, 1)
        
        def loss_and_jac(extended_coef):
            coef = extended_coef[:n_features] - extended_coef[n_features:]
            index = np.dot(X, coef.reshape(-1, 1))
            y_pred = sigmoid(index + offset)
            m_loss = - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred) + grad_corrections * index
            loss = np.mean(sample_weights * m_loss) + self._alpha_l1 * np.sum(extended_coef) + 0.5 * self._alpha_l2 * np.sum(extended_coef**2)
            moment = (y_pred - y + grad_corrections) * X
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
    
    def predict_proba(self, X, offset=None):    
        if offset is None:
            offset = np.zeros((X.shape[0], 1))
        offset = offset.reshape(-1, 1)
        y_pred = sigmoid(np.dot(X, self.coef_.reshape(-1, 1)) + offset)
        return np.concatenate((1 - y_pred, y_pred), axis=1)
    
    def predict(self, X, offset=None):     
        if offset is None:
            offset = np.zeros((X.shape[0], 1))
        offset = offset.reshape(-1, 1)
        return sigmoid(np.dot(X, self.coef_.reshape(-1, 1)) + offset) >= 0.5
    
    
    def score(self, X, y_true):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, self.predict_proba(X)[:, 1].reshape(y_true.shape))

    def accuracy(self, X, y_true):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, self.predict(X).reshape(y_true.shape))
    
    