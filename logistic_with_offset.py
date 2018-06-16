import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from itertools import product

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LoopIterator():
    def __init__(self, data, batch_size, random=True):
        self.batch_size = batch_size
        self.data = data
        self.iter = 0
        self.random = random

    def get_next(self):
        if self.random:
            return np.random.choice(self.data, size=self.batch_size)

        if self.data.shape[0] - self.iter < self.batch_size:
            indices = np.concatenate((np.arange(self.iter, self.data.shape[0]),
                                      np.arange(self.batch_size - self.data.shape[0] + self.iter)))
            self.iter = self.batch_size - self.data.shape[0] + self.iter
            #print("Starting a new epoch on the data!")
        else:
            indices = np.arange(self.iter, self.iter + self.batch_size)
            self.iter += self.batch_size
        return self.data[indices]


class LogisticWithOffset():
    def __init__(self, steps=1000, alpha_l1=0.1, alpha_l2=0.1, learning_rate=0.1, fit_intercept=False,
                    learning_schedule='constant', tol=1e-6, batch_size=50):
        self._steps = steps
        self._alpha_l1 = alpha_l1
        self._alpha_l2 = alpha_l2
        self._learning_rate = learning_rate
        self._tol = tol
        self._fit_intercept = fit_intercept
        self._learning_schedule = learning_schedule
        self._batch_size = batch_size
        self.session = None
        
    def tf_graph_init(self, num_outcomes, num_features):
        g = tf.Graph()
        with g.as_default():
            self.Y = tf.placeholder("float", [None, num_outcomes], name="outcome")
            self.X = tf.placeholder("float", [None, num_features], name="features")
            self.Offset = tf.placeholder("float", [None, 1], name="offset")
            self.SampleWeights = tf.placeholder("float", [None, 1], name="sample_weights")
            
            self.weights = tf.Variable(tf.zeros([num_features, num_outcomes]), name="weights")
            
            if self._fit_intercept:
                self.bias = tf.Variable(tf.zeros([num_outcomes]), name="biases")
                self.index = tf.add(tf.matmul(self.X, self.weights), self.bias)
            else:
                self.index = tf.matmul(self.X, self.weights)

            self.offset_index = tf.add(self.index, self.Offset)        
            
            self.Y_pred = 1./(1. + tf.exp(-self.offset_index))
            self.m_loss = - tf.add(tf.multiply(self.Y, tf.log(self.Y_pred + 1e-10)), tf.multiply(1-self.Y, tf.log(1 - self.Y_pred + 1e-10)))
            self.cost = tf.reduce_mean(tf.multiply(self.SampleWeights, self.m_loss))
            
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.constant(self._learning_rate, dtype=tf.float32)
            if not self._learning_schedule == 'constant':
                decay_steps = 1.0
                decay_rate = .5
                learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate)
            self.optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate,
                                                                l1_regularization_strength=float(self._alpha_l1),
                                                                l2_regularization_strength=float(self._alpha_l2))
            
            #l1_regularizer = tf.contrib.layers.l1_regularizer(scale=self._alpha_l1)
            #l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self._alpha_l2)
            #l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, tf.trainable_variables())
            #l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, tf.trainable_variables())
            #self.cost = self.cost + l1_regularization_penalty + l2_regularization_penalty
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate)
            
            self.train = self.optimizer.minimize(self.cost, global_step=global_step)

            self.init = tf.global_variables_initializer()
            self.session = tf.Session()

    def fit(self, X, y, offset=None, sample_weights=None):

        if offset is None:
            offset = np.zeros((X.shape[0], 1))
        if sample_weights is None:
            sample_weights = np.ones((X.shape[0], 1))

        if self.session is None:
            self.tf_graph_init(y.shape[1], X.shape[1])
        
        self.session.run(self.init)
        self._training_cost = []
        self._training_cost.append(self.session.run(self.cost, feed_dict={self.X: X, self.Offset: offset, self.SampleWeights: sample_weights, self.Y: y}))
        batch_size = min(X.shape[0], self._batch_size)
        batch_iterator = LoopIterator(np.arange(X.shape[0]), batch_size, random=False)
        for step in range(self._steps):
            batch_inds = batch_iterator.get_next() 
            self.session.run(self.train, feed_dict={self.X: X[batch_inds],
                                                    self.Offset: offset[batch_inds], 
                                                    self.SampleWeights: sample_weights[batch_inds],
                                                    self.Y: y[batch_inds]})
            # test after every epoch of data
            if step % int(np.ceil(X.shape[0]/batch_size)) == 0 and step // int(np.ceil(X.shape[0]/batch_size)) > 2:
                self._training_cost.append(self.session.run(self.cost, feed_dict={self.X: X, self.Offset: offset, self.SampleWeights: sample_weights, self.Y: y}))
                if np.abs(self._training_cost[-2] - self._training_cost[-1]) <= self._tol:
                    print("early stopping at iter:{}".format(step))
                    break

        return self

    def predict(self, X, offset=None):
        if not offset:
            offset = np.zeros((X.shape[0], 1))
        return self.session.run(self.Y_pred, feed_dict={self.X: X, self.Offset: offset}) >= 0.5
    
    def predict_proba(self, X, offset=None):
        if not offset:
            offset = np.zeros((X.shape[0], 1))
        y_pred = self.session.run(self.Y_pred, feed_dict={self.X: X, self.Offset: offset})
        return np.concatenate((y_pred, 1 - y_pred), axis=1)
    
    @property
    def coef_(self):
        return self.session.run(self.weights.value())

    def score(self, X, y_true, offset=None):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, self.predict(X, offset).reshape(y_true.shape))
    
    @property
    def training_cost_(self):
        return self._training_cost if self._training_cost is not None else None
    