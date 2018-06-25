import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import joblib
import scipy
from scipy.optimize import fmin_l_bfgs_b
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
plt.style.use('ggplot')


def cross_product(X1, X2):
    """ Cross product of features in matrices X1 and X2
    """
    n = np.shape(X1)[0]
    assert n == np.shape(X2)[0]
    return (X1.reshape(n,1,-1) * X2.reshape(n,-1,1)).reshape(n,-1) 

def gen_data(n_samples, dim_x, dim_z, kappa_theta, kappa_z, sigma_x, sigma_epsilon, sigma_eta):
    """ Generate data from:
    y = <x[support_theta], \theta> + <x \cross z[support_z], alpha> + <x \cross (z[support_z]**2 - E[z**2]), alpha> + epsilon
    Prob[d=1 | z, x] = Logistic(<x[support_theta], beta> + <z[support_z], gamma>)
    epsilon ~ Normal(0, sigma_epsilon)
    x, z ~ Normal(0, sigma_x)
    alpha_x, beta_x, theta are all equal to 1
    support_x, support_z drawn uniformly at random

    We observe (x, z, d, d*y)
    """
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
    
    uz = cross_product(z, x) @ alpha + 2. * cross_product(z**2 - (sigma_x**2)/3., x) @ alpha2 
    y = x @ theta + uz + np.random.normal(0, sigma_epsilon, size=n_samples)
    index_d = x @ beta + z @ gamma
    pz = scipy.special.expit(sigma_eta * index_d)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(pz)
    plt.subplot(1, 2, 2)
    plt.hist(index_d)
    plt.savefig('propensity.png')
    plt.close()
    d = np.random.binomial(1, pz)
    return x, z, d, d*y, theta, pz, uz 

def direct_fit(x, z, d, dy, true_pz, true_uz, opts):
    ''' Direct lasso regression of y[d==1] on x[d==1], (x cross z)[d==1] '''
    comp_x = np.concatenate((x, cross_product(z, x)), axis=1)
    model_y = LassoCV()
    model_y.fit(comp_x[d==1], dy[d==1])
    return model_y.coef_[:x.shape[1]]

def non_ortho_oracle(x, z, d, dy, true_pz, true_uz, opts):
    ''' Non orthogonal inverse propensity estimation with oracle access to propensity p(z) '''
    n_samples, n_features = x.shape
    sample_weights = d / true_pz
    model_final = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(n_features)/n_samples), fit_intercept=False)
    model_final.fit(np.sqrt(sample_weights.reshape(-1, 1)) * x, np.sqrt(sample_weights) * dy)
    return model_final.coef_

def ortho_oracle(x, z, d, dy, true_pz, true_uz, opts):
    ''' Orthogonal inverse propensity estimation with orthogonal correction and oracle
    access to both propensity p(z) and u(z) = E[u(y, x'theta) | z] '''
    n_samples, n_features = x.shape
    pz = true_pz
    hz = true_uz * (d - pz) / pz
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(n_features)/n_samples)
    def loss_and_jac(extended_coef):
        coef = extended_coef[:n_features] - extended_coef[n_features:]
        index = x @ coef
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

def non_ortho(x, z, d, dy, true_pz, true_uz, opts):
    ''' Non orthogonal inverse propensity estimation with a first stage propensity estimation '''
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

def ortho(x, z, d, dy, true_pz, true_uz, opts):
    ''' Orthogonal inverse propensity estimation with orthogonal correction and first stage
    estimation of both propensity p(z) and residual u(z) = E[u(y, x'theta) | z] '''
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
        theta_prel = non_ortho(x[train_index], z[train_index], d[train_index], dy[train_index], true_pz[train_index], true_uz[train_index], opts)
        # conditional residual estimation E[u(y, x'theta) | z], by regressing y on x, z and then subtracting x'theta_prel
        model_uz = RandomForestRegressor(n_estimators=200, min_samples_leaf=20)
        model_uz.fit(comp_x[train_index][d[train_index]==1], dy[train_index][d[train_index]==1])
        uz[test_index] = model_uz.predict(comp_x[test_index])
        uz[test_index] -= x[test_index] @ theta_prel
    # orthogonal correction multiplier of the index
    hz = uz * (d - pz) / pz

    # final regression
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(n_features)/n_samples)
    def loss_and_jac(extended_coef):
        coef = extended_coef[:n_features] - extended_coef[n_features:]
        index = x @ coef
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
    

def experiment(exp_id, methods, opts):
    np.random.seed(exp_id)
    # Generate data
    x, z, d, dy, theta, true_pz, true_uz =\
            gen_data(opts['n_samples'], opts['dim_x'], opts['dim_z'], opts['kappa_theta'], opts['kappa_z'], 
            opts['sigma_x'], opts['sigma_epsilon'], opts['sigma_eta'])
    print('True coef:' + ', '.join(["({}: {:.3f})".format(ind, c) for ind, c in enumerate(theta) if np.abs(c)>0.001]))

    coefs = {}
    l1_error = {}
    l2_error = {}
    for m_name, m_func in methods.items():
        coefs[m_name] = m_func(x, z, d, dy, true_pz, true_uz, opts)
        l1_error[m_name] = np.linalg.norm(coefs[m_name] - theta, ord=1)
        l2_error[m_name] = np.linalg.norm(coefs[m_name] - theta, ord=2)
        print('{} l1: {:.3f}'.format(m_name, l1_error[m_name]))
        print('{} l2: {:.3f}'.format(m_name, l2_error[m_name]))

    return l1_error, l2_error, coefs

def main(opts, target_dir='.', reload_results=True):
    random_seed = 123
    
    methods = {'Direct': direct_fit, 'IPS oracle': non_ortho_oracle, 'Ortho oracle': ortho_oracle, 'IPS': non_ortho, 'Ortho': ortho}
    results_file = os.path.join(target_dir, 'logistic_te_errors_{}.jbl'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()])))
    if reload_results and os.path.exists(results_file):
        results = joblib.load(results_file)
    else:
        results = Parallel(n_jobs=-1, verbose=1)(
                delayed(experiment)(random_seed + exp_id, methods, opts) 
                for exp_id in range(opts['n_experiments']))
    joblib.dump(results, results_file)
    
    l1_errors = {}
    l2_errors = {}
    coefs = {}
    for m_name, _ in methods.items():
        l1_errors[m_name] = np.array([results[i][0][m_name] for i in range(opts['n_experiments'])])
        l2_errors[m_name] = np.array([results[i][1][m_name] for i in range(opts['n_experiments'])])
        coefs[m_name] = np.array([results[i][2][m_name] for i in range(opts['n_experiments'])])

    n_methods = len(methods)
    n_coefs = opts['dim_x']
    plt.figure(figsize=(4 * n_coefs, 2 * n_methods))
    for it, (m_name, _) in enumerate(methods.items()):
        for i in range(coefs[m_name].shape[1]):
            plt.subplot(n_methods, n_coefs, it * n_coefs + i + 1)
            plt.hist(coefs[m_name][:, i])
            plt.title("{}[{}]. $\mu$: {:.2f}, $\sigma$: {:.2f}".format(m_name, i, np.mean(coefs[m_name][:, i]), np.std(coefs[m_name][:, i])))
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'dist_{}.png'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))), dpi=300)
    plt.close()

    plt.figure(figsize=(1.5 * n_methods, 2.5))
    plt.violinplot([l2_errors[m_name] for m_name in methods.keys()], showmedians=True)
    plt.xticks(np.arange(1, n_methods + 1), list(methods.keys()))
    plt.ylabel('$\ell_2$ error')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'l2_errors_{}.png'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    plt.figure(figsize=(1.5 * n_methods, 2.5))
    plt.violinplot([l2_errors[m_name] - l2_errors['Ortho'] for m_name in methods.keys() if m_name != 'Ortho'], showmedians=True)
    plt.xticks(np.arange(1, n_methods), [m_name for m_name in methods.keys() if m_name != 'Ortho'])
    plt.ylabel('$\ell_2$[method] - $\ell_2$[Ortho]')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'l2_decrease_{}.png'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))), dpi=300)
    plt.close()

    return l1_errors, l2_errors, coefs

if __name__=="__main__":

    opts= {'n_experiments': 1000, # number of monte carlo experiments
            'n_samples': 5000, # samples used for estimation
            'dim_x': 20, # dimension of controls x
            'dim_z': 20, # dimension of variables used for heterogeneity (subset of x)
            'kappa_theta': 1, # support size of target parameter
            'kappa_z': 1, # support size of nuisance
            'sigma_epsilon': 1., # variance of error in secondary moment equation
            'sigma_eta': .1, # variance of error in secondary moment equation, i.e. multiplier in logistic index
            'sigma_x': 3, # variance parameter for co-variate distribution
            'lambda_coef': 1.0, # coeficient in front of the asymptotic rate for regularization lambda
            'n_folds': 3, # number of folds used in cross-fitting
    }
    reload_results = True
    target_dir = 'results_missing'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    main(opts, target_dir=target_dir, reload_results=reload_results)
    