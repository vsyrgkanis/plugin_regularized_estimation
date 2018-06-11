import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from itertools import product

class LogisticWithOffset():
    # TODO: allow different subsets for L1 and L2 regularization?
    def __init__(self, steps=1000, alpha_l1=0.1, alpha_l2=0.1, learning_rate=0.1, fit_intercept=False):
        self._steps = steps
        self._alpha_l1 = alpha_l1
        self._alpha_l2 = alpha_l2
        self._learning_rate = learning_rate
        self._fit_intercept = fit_intercept
        self.session = None
        
    def tf_graph_init(self, num_outcomes, num_features):
        '''
        Creates the graph that corresponds to the squared loss with an ell_1 penalty
        only on the subset of features specified by the self._subset variable. Also
        creates the optimizer that minimizes this loss and a persistent tensorflow
        session for the class.
        '''
        self.Y = tf.placeholder("float", [None, num_outcomes], name="outcome")
        self.X = tf.placeholder("float", [None, num_features], name="features")
        self.Offset = tf.placeholder("float", [None, 1], name="offset")
        self.SampleWeights = tf.placeholder("float", [None, 1], name="sample_weights")
        
        self.weights = tf.Variable(tf.random_normal(
                    [num_features, num_outcomes], 0, 0.1), name="weights")
        
        if self._fit_intercept:
            self.bias = tf.Variable(tf.random_normal(
                    [num_outcomes], 0, 0.1), name="biases")
            self.index = tf.add(tf.matmul(self.X, self.weights), self.bias)
        else:
            self.index = tf.matmul(self.X, self.weights)

        self.offset_index = tf.add(self.index, self.Offset)

        #regularization = tf.contrib.layers.apply_regularization(
        #                    tf.contrib.layers.l1_l2_regularizer(
        #                        scale_l1=self._alpha_l1, scale_l2=self._alpha_l2), [self.weights])
        
        self.Y_pred = 1./(1. + tf.exp(-self.offset_index))
        self.m_loss = - tf.add(tf.multiply(self.Y, tf.log(self.Y_pred)), tf.multiply(1-self.Y, tf.log(1 - self.Y_pred)))
        self.cost = tf.reduce_mean(tf.multiply(self.SampleWeights, self.m_loss))
        
        self.optimizer = tf.train.ProximalAdagradOptimizer(tf.constant(self._learning_rate, dtype=tf.float32),
                                                            l1_regularization_strength=float(self._alpha_l1),
                                                            l2_regularization_strength=float(self._alpha_l2))
        #self.optimizer = tf.train.AdagradOptimizer(learning_rate=self._learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        self.session = tf.Session()

    def fit(self, X, y, offset=None, sample_weights=None):
        # TODO: any better way to deal with sparsity?
        if sp.issparse(X):
            X = X.toarray()

        y = y.reshape(-1, 1)

        if not offset:
            offset = np.zeros((X.shape[0], 1))
        if not sample_weights:
            sample_weights = np.ones((X.shape[0]))

        if not self.session:
            self.tf_graph_init(y.shape[1], X.shape[1])
        
        self.session.run(tf.global_variables_initializer())
        for step in range(self._steps):
            self.session.run(self.train, feed_dict={self.X: X,
                                                    self.Offset: offset, 
                                                    self.SampleWeights: sample_weights,
                                                    self.Y: y})
        return self

    def predict(self, X, offset=None):    
        # TODO: any better way to deal with sparsity?
        if sp.issparse(X):
            X = X.toarray()
        if not offset:
            offset = np.zeros((X.shape[0], 1))
        return self.session.run(self.Y_pred, feed_dict={self.X: X, self.Offset: offset})
    
    @property
    def coef_(self):
        return self.session.run(self.weights.value())

    def score(self, X, y_true, offset=None):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, self.predict(X, offset).reshape(y_true.shape))


    