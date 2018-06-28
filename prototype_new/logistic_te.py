import numpy as np
import scipy
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from orthopy.scipy_logistic_with_gradient_correction import LogisticWithOffsetAndGradientCorrection

###############################
# DGPs
###############################

def gen_data(opts):
    """ Generate data from:
    Pr[y|x,t] = Sigmoid(<z[support_theta], theta> * t + <x[support_x], alpha_x>)
    t = <x[support_x], beta_x> + eta
    epsilon ~ Normal(0, sigma_epsilon)
    eta ~ Normal(0, sigma_eta)
    z = x[:dim_z]
    alpha_x, beta_x, theta are all equal to 1
    support_x, support_theta, subset_z drawn uniformly at random. support_x contains support_theta
    """
    n_samples = opts['n_samples']
    dim_x = opts['dim_x']
    dim_z = opts['dim_z']
    kappa_x = opts['kappa_x']
    kappa_theta = opts['kappa_theta']
    sigma_eta = opts['sigma_eta']
    sigma_x = opts['sigma_x']

    x = np.random.uniform(-sigma_x, sigma_x, size=(n_samples, dim_x))
    z = x[:, :dim_z].reshape(n_samples, -1)
        
    support_theta = np.random.choice(np.arange(0, dim_z), kappa_theta, replace=False)
    support_x = np.random.choice(np.array(list(set(np.arange(0, dim_x)) - set(support_theta))), kappa_x - kappa_theta, replace=False)
    support_x = np.concatenate((support_x, support_theta), axis=0)
    alpha_x = np.ones((kappa_x, 1))
    beta_x = np.ones((kappa_x, 1))
    theta = np.ones((kappa_theta, 1))
    t = np.matmul(x[:, support_x], beta_x) + np.random.normal(0, sigma_eta, size=(n_samples, 1))
    index_y = np.matmul(z[:, support_theta], theta) * t + np.matmul(x[:, support_x], alpha_x)
    p_y = scipy.special.expit(index_y)
    y = np.random.binomial(1, p_y)

    true_param = np.zeros(dim_z)
    true_param[support_theta] = theta.flatten()
    return (x, t, z, y), true_param

###############################
# Estimation Utils
###############################

def direct_models(data, opts):
    x, t, z, y = data
    n_samples = x.shape[0]
    
    # Run lasso for treatment as function of controls
    model_t = Lasso(alpha=np.sqrt(np.log(x.shape[1])/n_samples))
    model_t.fit(x, t.ravel())

    # Run logistic lasso for outcome as function of composite treatments and controls
    comp_x = np.concatenate((z * t, x), axis=1)
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(comp_x.shape[1])/n_samples)
    model_y = LogisticWithOffsetAndGradientCorrection(alpha_l1=l1_reg, alpha_l2=0., tol=1e-6)
    model_y.fit(comp_x, y)

    return model_y, model_t

def nuisance_estimates(x, t, z, y, tr_inds, tst_inds, opts):
    comp_x = np.concatenate((z * t, x), axis=1)
    
    # Get direct regression models fitted on the training set
    data_train = (x[tr_inds], t[tr_inds], z[tr_inds], y[tr_inds])
    model_y, model_t = direct_models(data_train, opts)
    
    # Compute all required quantities for orthogonal estimation on the test set
    # Preliminary theta estimate: \tilde{theta}
    theta_prel = model_y.coef_.flatten()[:z.shape[1]].reshape(-1, 1)
    # Preliminary estimate of f(u): \hat{f}(u) = u'alpha_prel
    alpha_prel = model_y.coef_.flatten()[z.shape[1]:].reshape(-1, 1)
    # Preliminary estimate of \pi(u): \hat{\pi}(u) = model_t.predict(u)
    t_test_pred = model_t.predict(x[tst_inds]).reshape(-1, 1)
    # Preliminary estimate of G(x'\theta + f(u)): G_prel = model_y.predict([x, u])
    y_test_pred = model_y.predict_proba(comp_x[tst_inds])[:, 1].reshape(-1, 1)
    # Preliminary estimate of q(u): \hat{q}(u) = \hat{\pi}(u) * B(u)'\tilde{theta} + u'\tilde{alpha}
    q_test = t_test_pred * np.dot(z[tst_inds], theta_prel) + np.dot(x[tst_inds], alpha_prel)
    # Preliminary estimate of V(z) = G(index) * (1 - G(index))
    V_test = y_test_pred * (1 - y_test_pred)    
    # Residual treatment: x - \hat{h}(u) = B(u) * (tau - pi(u))
    res_test = t[tst_inds] - t_test_pred
    comp_res_test = z[tst_inds] * res_test

    return comp_res_test, q_test, V_test


###############################
# Estimation Methods
###############################

def direct_fit(data, opts):
    _, _, z, _ = data
    model_y, _ = direct_models(data, opts)
    return model_y.coef_.flatten()[:z.shape[1]]

def dml_crossfit(data, opts):
    """ Orthogonal estimation of coefficient theta with cross-fitting
    """
    x, t, z, y = data
    n_samples = x.shape[0]
    
    # Build first stage nuisance estimates for each sample using cross-fitting
    comp_res = np.zeros(z.shape)
    offsets = np.zeros((x.shape[0], 1))
    V = np.zeros((x.shape[0], 1))
    for train_index, test_index in KFold(n_splits=opts['n_folds']).split(x):
        comp_res[test_index], offsets[test_index], V[test_index] = nuisance_estimates(x, t, z, y, train_index, test_index, opts)
    
    # Calculate normalized sample weights. Clipping for instability
    sample_weights = (1./np.clip(V, 0.01, 1))/np.mean((1./np.clip(V, 0.01, 1)))

    # Fit second stage regression with plugin nuisance estimates
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(z.shape[1])/n_samples)
    model_final = LogisticWithOffsetAndGradientCorrection(alpha_l1=l1_reg, alpha_l2=0., tol=1e-6)

    model_final.fit(comp_res, y, offsets=offsets, sample_weights=sample_weights)

    return model_final.coef_.flatten()


