import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, MultiTaskLasso, MultiTaskLassoCV, LinearRegression
from mcpy.utils import cross_product

###############################
# DGPs
###############################

def gen_data(opts):
    """ Generate data from:
    y = <z[support_theta], theta> * t + <x[support_x], alpha_x> + epsilon
    t = <x[support_x], beta_x> + eta
    epsilon ~ Normal(0, sigma_epsilon)
    eta ~ Normal(0, sigma_eta)
    z = x[subset_z] for some subset of x of size dim_z
    alpha_x, beta_x, theta are all equal to 1
    support_x, support_theta, subset_z drawn uniformly at random
    """
    n_samples = opts['n_samples']
    dim_x = opts['dim_x']
    dim_z = opts['dim_z']
    kappa_x = opts['kappa_x']
    kappa_theta = opts['kappa_theta']
    sigma_eta = opts['sigma_eta']
    sigma_epsilon = opts['sigma_epsilon']

    x = np.random.normal(0, 1, size=(n_samples, dim_x))
    if dim_z == 1:
        z = np.ones((n_samples, 1))
    else:
        supp_z = np.random.choice(np.arange(x.shape[1]), dim_z-1, replace=False)
        z = np.concatenate((x[:, supp_z].reshape(n_samples, -1), np.ones((n_samples, 1))), axis=1)
        
    support_x = np.random.choice(np.arange(0, dim_x), kappa_x, replace=False)
    support_theta = np.random.choice(np.arange(0, dim_z), kappa_theta, replace=False)
    alpha_x = np.random.uniform(1, 1, size=(kappa_x, 1))
    beta_x = np.random.uniform(1, 1, size=(kappa_x, 1))
    theta = np.random.uniform(1, 1, size=(kappa_theta, 1))
    t = np.matmul(x[:, support_x], beta_x) + np.random.normal(0, sigma_eta, size=(n_samples, 1))
    y = np.matmul(z[:, support_theta], theta) * t + np.matmul(x[:, support_x], alpha_x) + np.random.normal(0, sigma_epsilon, size=(n_samples, 1))

    
    true_param = np.zeros(dim_z)
    true_param[support_theta] = theta.flatten()

    print("done data generation")
    return (x, t, z, y), true_param

###############################
# Estimation Methods
###############################

def direct_fit(data, opts):
    x, t, z, y = data
    model = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(z.shape[1] + x.shape[1])/x.shape[0]))
    model.fit(np.concatenate((z*t, x), axis=1), y.flatten())
    print("done direct")
    return model.coef_.flatten()[:z.shape[1]]

def dml_fit(data, opts):
    """ Orthogonal estimation of coefficient theta
    """
    x, t, z, y = data
    comp_x = cross_product(z, x)
    n_samples = x.shape[0]
    
    model_t = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(x.shape[1]) * 2. / n_samples))
    model_y = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(z.shape[1] * x.shape[1]) * 2. / n_samples))
    model_f = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(z.shape[1]) * 2. / n_samples), fit_intercept=False)
    
    model_t.fit(x[:n_samples//2], t[:n_samples//2].flatten())
    model_y.fit(comp_x[:n_samples//2], y[:n_samples//2].flatten())
    res_t = t[n_samples//2:] - model_t.predict(x[n_samples//2:]).reshape((n_samples//2, -1))
    res_y = y[n_samples//2:] - model_y.predict(comp_x[n_samples//2:]).reshape((n_samples//2, -1))
    model_f.fit(z[n_samples//2:]*res_t, res_y.flatten())
    print("done dml_fit")
    return model_f.coef_.flatten()

def dml_crossfit(data, opts):
    """ Orthogonal estimation of coefficient theta with cross-fitting
    """
    x, t, z, y = data
    comp_x = cross_product(z, x)
    n_samples = x.shape[0]
    
    model_t = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(x.shape[1]) * 2. / n_samples))
    model_y = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(z.shape[1] * x.shape[1]) * 2. / n_samples))
    model_f = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(z.shape[1]) * 2. / n_samples), fit_intercept=False)
    
    res_y = np.zeros(y.shape)
    res_t = np.zeros(t.shape)
    for train_index, test_index in KFold(n_splits=opts['n_folds']).split(x):
        model_t.fit(x[train_index], t[train_index].flatten())
        model_y.fit(comp_x[train_index], y[train_index].flatten())
        res_t[test_index] = t[test_index] - model_t.predict(x[test_index]).reshape(test_index.shape[0], -1)
        res_y[test_index] = y[test_index] - model_y.predict(comp_x[test_index]).reshape(test_index.shape[0], -1)
    
    model_f.fit(z*res_t, res_y.flatten())
    print("done dml_crossfit")
    return model_f.coef_.flatten()

