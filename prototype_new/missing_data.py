import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import scipy
from scipy.optimize import fmin_l_bfgs_b
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from utils import cross_product

###############################
# DGPs
###############################

def gen_data(opts):
    """ Generate data from:
    y = <x[support_theta], \theta> + <x \cross z[support_z], alpha> + <x \cross (z[support_z]**2 - E[z**2]), alpha> + epsilon
    Prob[d=1 | z, x] = Logistic(<x[support_theta], beta> + <z[support_z], gamma>)
    epsilon ~ Normal(0, sigma_epsilon)
    x, z ~ Normal(0, sigma_x)
    alpha_x, beta_x, theta are all equal to 1
    support_x, support_z drawn uniformly at random

    We observe (x, z, d, d*y)
    """
    n_samples = opts['n_samples'] 
    dim_x = opts['dim_x']
    dim_z = opts['dim_z']
    kappa_theta = opts['kappa_theta']
    kappa_z = opts['kappa_z']
    sigma_x = opts['sigma_x']
    sigma_epsilon = opts['sigma_epsilon']
    sigma_eta = opts['sigma_eta']

    x = np.random.uniform(-sigma_x, sigma_x, size=(n_samples, dim_x))
    z = np.random.uniform(-sigma_x, sigma_x, size=(n_samples, dim_z))

    support_x = np.arange(0, kappa_theta)
    support_z = np.arange(0, kappa_z)
    theta = np.zeros(dim_x)
    theta[support_x] = np.random.uniform(2, 2, size=kappa_theta)
    support_xz = np.array([dim_z * i + support_z for i in support_x]).flatten()
    alpha = np.zeros(dim_z * dim_x)
    alpha[support_xz] = np.random.uniform(1, 2, size=kappa_z * kappa_theta)
    alpha2 = np.zeros(dim_z * dim_x)
    alpha2[support_xz] = np.random.uniform(1, 2, size=kappa_z * kappa_theta)
    beta = np.zeros(dim_x)
    beta[support_x] = np.random.uniform(0, 1, size=kappa_theta) / kappa_theta
    gamma = np.zeros(dim_z)
    gamma[support_z] = np.random.uniform(2, 3, size=kappa_z) / kappa_z
    
    uz = np.matmul(cross_product(z, x), alpha) + 2. * np.matmul(cross_product(z**2 - (sigma_x**2)/3., x), alpha2)
    y = np.matmul(x, theta) + uz + np.random.normal(0, sigma_epsilon, size=n_samples)
    index_d = np.matmul(x, beta) + np.matmul(z, gamma)
    pz = scipy.special.expit(sigma_eta * index_d)
    d = np.random.binomial(1, pz)
    return (x, z, d, d*y, pz, uz), theta

###############################
# Estimation Methods
###############################

def direct_fit(data, opts):
    ''' Direct lasso regression of y[d==1] on x[d==1], (x cross z)[d==1] '''
    x, z, d, dy, _, _ = data
    comp_x = np.concatenate((x, cross_product(z, x)), axis=1)
    model_y = LassoCV()
    model_y.fit(comp_x[d==1], dy[d==1])
    return model_y.coef_[:x.shape[1]]

def non_ortho_oracle(data, opts):
    ''' Non orthogonal inverse propensity estimation with oracle access to propensity p(z) '''
    x, z, d, dy, true_pz, _ = data
    n_samples, n_features = x.shape
    sample_weights = d / true_pz
    model_final = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(n_features)/n_samples), fit_intercept=False)
    model_final.fit(np.sqrt(sample_weights.reshape(-1, 1)) * x, np.sqrt(sample_weights) * dy)
    return model_final.coef_

def ortho_oracle(data, opts):
    ''' Orthogonal inverse propensity estimation with orthogonal correction and oracle
    access to both propensity p(z) and u(z) = E[u(y, x'theta) | z] '''
    x, z, d, dy, true_pz, true_uz = data
    n_samples, n_features = x.shape
    pz = true_pz
    hz = true_uz * (d - pz) / pz
    l1_reg = 5. * opts['lambda_coef'] * np.sqrt(np.log(n_features)/n_samples)
    def loss_and_jac(extended_coef):
        coef = extended_coef[:n_features] - extended_coef[n_features:]
        index = np.matmul(x, coef)
        m_loss = (d / pz) * .5 * (index - dy)**2 + hz * index
        loss = np.mean(m_loss) + l1_reg * np.sum(extended_coef)
        moment = ((d / pz) * (index - dy)).reshape(-1, 1) * x + hz.reshape(-1, 1) * x
        grad = np.mean(moment, axis=0).flatten() 
        jac = np.concatenate([grad, -grad]) + l1_reg
        return loss, jac
    
    w, _, _ = fmin_l_bfgs_b(loss_and_jac, np.zeros(2*n_features), 
                            bounds=[(0, None)] * n_features * 2,
                            pgtol=1e-6)

    return w[:n_features] - w[n_features:]

def non_ortho(data, opts):
    ''' Non orthogonal inverse propensity estimation with a first stage propensity estimation '''
    x, z, d, dy, _, _ = data
    n_samples = x.shape[0]

    # nuisance estimation
    comp_x = np.concatenate((x, z), axis=1)
    kf = KFold(n_splits=opts['n_folds'])
    pz = np.zeros(n_samples)
    for train_index, test_index in kf.split(x):        
        pz[test_index] = LogisticRegressionCV(penalty='l1', solver='liblinear').fit(comp_x[train_index], d[train_index]).predict_proba(comp_x[test_index])[:, 1]

    # final regression
    sample_weights = d / pz
    model_final = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(x.shape[1])/n_samples), fit_intercept=False)
    model_final.fit(np.sqrt(sample_weights.reshape(-1, 1)) * x, np.sqrt(sample_weights) * dy)
    
    return model_final.coef_

def ortho(data, opts):
    ''' Orthogonal inverse propensity estimation with orthogonal correction and first stage
    estimation of both propensity p(z) and residual u(z) = E[u(y, x'theta) | z] '''
    x, z, d, dy, true_pz, true_uz = data
    n_samples, n_features = x.shape
    comp_x = np.concatenate((x, z), axis=1)

    # nuisance estimation
    kf = KFold(n_splits=opts['n_folds'])
    pz = np.zeros(n_samples)
    uz = np.zeros(n_samples)
    for train_index, test_index in kf.split(x):
        # propensity estimation
        model_p = LogisticRegressionCV(penalty='l1', solver='liblinear')
        model_p.fit(comp_x[train_index], d[train_index])
        pz[test_index] = model_p.predict_proba(comp_x[test_index])[:, 1]
        # preliminary theta estimation with non orthogonal IPS method
        data_train = (x[train_index], z[train_index], d[train_index], dy[train_index], true_pz[train_index], true_uz[train_index])
        theta_prel = non_ortho(data_train, opts)
        # conditional residual estimation E[u(y, x'theta) | z], by regressing y on x, z and then subtracting x'theta_prel
        model_uz = RandomForestRegressor(n_estimators=200, min_samples_leaf=20)
        model_uz.fit(comp_x[train_index][d[train_index]==1], dy[train_index][d[train_index]==1])
        uz[test_index] = model_uz.predict(comp_x[test_index])
        uz[test_index] -= np.matmul(x[test_index], theta_prel)
    # orthogonal correction multiplier of the index
    hz = uz * (d - pz) / pz

    # final regression
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(n_features)/n_samples)
    def loss_and_jac(extended_coef):
        coef = extended_coef[:n_features] - extended_coef[n_features:]
        index = np.matmul(x, coef)
        m_loss = (d / pz) * .5 * (index - dy)**2 + hz * index
        loss = np.mean(m_loss) + l1_reg * np.sum(extended_coef)
        moment = ((d / pz) * (index - dy)).reshape(-1, 1) * x + hz.reshape(-1, 1) * x
        grad = np.mean(moment, axis=0).flatten() 
        jac = np.concatenate([grad, -grad]) + l1_reg
        return loss, jac
    
    w, _, _ = fmin_l_bfgs_b(loss_and_jac, np.zeros(2*n_features), 
                            bounds=[(0, None)] * n_features * 2,
                            pgtol=1e-6)

    return w[:n_features] - w[n_features:]
    

    