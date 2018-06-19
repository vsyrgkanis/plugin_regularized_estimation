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


def cross_product(X1, X2):
    """ Cross product of features in matrices X1 and X2
    """
    n = np.shape(X1)[0]
    assert n == np.shape(X2)[0]
    return (X1.reshape(n,1,-1) * X2.reshape(n,-1,1)).reshape(n,-1) 

def gen_data(n_samples, dim_x, dim_z, kappa_theta, kappa_z, sigma_x, sigma_epsilon, sigma_eta):
    """ Generate data from:
    y = <x[support_theta], \theta> + <z[support_z], alpha_z> + epsilon
    Prob[d=1 | z, x] = Logistic(<z[support_z], gamma_z>)
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
    alpha = np.zeros(dim_z * dim_x)
    support_xz = np.array([dim_z * i + support_z for i in range(dim_x)]).flatten()
    alpha[support_xz] = np.random.uniform(1, 1, size=kappa_z * dim_x)
    beta = np.zeros(dim_x)
    beta[support_x] = np.random.uniform(0, 0, size=kappa_theta) / kappa_theta
    gamma = np.zeros(dim_z)
    gamma[support_z] = np.random.uniform(2, 2, size=kappa_z) / kappa_z
    
    y = x @ theta + cross_product(x, z) @ alpha + 1. * cross_product(x, z**2 - sigma_x**2/3.) @ alpha + np.random.normal(0, sigma_epsilon, size=n_samples)
    index_d = x @ beta + z @ gamma
    p_d = scipy.special.expit(sigma_eta * index_d)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(p_d)
    plt.subplot(1, 2, 2)
    plt.hist(index_d)
    plt.savefig('propensity.png')
    plt.close()
    d = np.random.binomial(1, p_d)
    return x, z, d, d*y, theta, alpha, beta, gamma

def direct_fit(x, z, d, dy, opts):
    comp_x = np.concatenate((x, cross_product(x, z)), axis=1)
    model_y = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(x.shape[1])/x.shape[0]), fit_intercept=False) #LassoCV(fit_intercept=False) #alpha=opts['lambda_coef'] * np.sqrt(np.log(comp_x.shape[1])/comp_x.shape[0]), fit_intercept=False)
    model_y.fit(comp_x[d==1], dy[d==1])
    return model_y.coef_[:x.shape[1]]

def non_ortho_oracle(x, z, d, dy, opts, alpha, beta, gamma):
    sample_weights = d / scipy.special.expit(opts['sigma_eta'] * (x @ beta + z @ gamma))
    model_final = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(x.shape[1])/n_samples), fit_intercept=False)
    model_final.fit(np.sqrt(sample_weights.reshape(-1, 1)) * x, np.sqrt(sample_weights) * dy)
    
    return model_final.coef_

def ortho_oracle(x, z, d, dy, opts, alpha, beta, gamma):
    n_samples, n_features = x.shape
    pz = scipy.special.expit(opts['sigma_eta'] * (x @ beta + z @ gamma))
    hz = (cross_product(x, z) @ alpha) * (d - pz) / pz
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

def non_ortho(x, z, d, dy, opts, alpha, beta, gamma):
    n_samples = x.shape[0]

    # nuisance estimation
    comp_x = np.concatenate((x, z), axis=1)
    kf = KFold(n_splits=opts['n_folds'])
    pz = np.zeros(n_samples)
    for train_index, test_index in kf.split(x):        
        pz[test_index] = LogisticRegressionCV(penalty='l1', solver='liblinear').fit(comp_x[train_index], d[train_index]).predict_proba(comp_x[test_index])[:, 1]
    sample_weights = d / pz
    plt.figure()
    plt.hist(pz)
    plt.savefig('fitted_weights.png')
    plt.close()

    # final regression
    model_final = Lasso(alpha=opts['lambda_coef'] * np.sqrt(np.log(x.shape[1])/n_samples), fit_intercept=False)
    model_final.fit(np.sqrt(sample_weights.reshape(-1, 1)) * x, np.sqrt(sample_weights) * dy)
    
    return model_final.coef_

def ortho(x, z, d, dy, opts, alpha, beta, gamma):
    n_samples, n_features = x.shape
    comp_x = np.concatenate((x, z), axis=1)

    # nuisance estimation
    kf = KFold(n_splits=opts['n_folds'])
    pz = np.zeros(n_samples)
    uz = np.zeros(n_samples)
    for train_index, test_index in kf.split(x):
        
        model_p = LogisticRegressionCV(penalty='l1', solver='liblinear')
        model_p.fit(comp_x[train_index], d[train_index])
        pz[test_index] = model_p.predict_proba(comp_x[test_index])[:, 1]
        theta_prel = non_ortho(x[train_index], z[train_index], d[train_index], dy[train_index], opts, alpha, beta, gamma)
        uz[test_index] = RandomForestRegressor().fit(comp_x[train_index][d[train_index]==1], dy[train_index][d[train_index]==1]).predict(comp_x[test_index])
        uz[test_index] -= x[test_index] @ theta_prel
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
    

def experiment(exp_id, opts):
    np.random.seed(exp_id)
    # Generate data
    x, z, d, dy, theta, alpha, beta, gamma =\
            gen_data(opts['n_samples'], opts['dim_x'], opts['dim_z'], opts['kappa_theta'], opts['kappa_z'], 
            opts['sigma_x'], opts['sigma_epsilon'], opts['sigma_eta'])

    print('True coef:' + ', '.join(["({}: {:.3f})".format(ind, c) for ind, c in enumerate(theta) if np.abs(c)>0.001]))

    # Direct lasso for all coefficients
    direct_coef = direct_fit(x, z, d, dy, opts)
    l1_direct = np.linalg.norm(direct_coef - theta, ord=1)
    l2_direct = np.linalg.norm(direct_coef - theta, ord=2)
    print('Direct coef:' + ', '.join(["({}: {:.3f})".format(ind, c) for ind, c in enumerate(direct_coef) if np.abs(c)>0.001]))
    print('Direct l1: {:.3f}'.format(l1_direct))

    eps = .5
    alpha, beta, gamma = alpha + np.random.uniform(-eps, eps, alpha.shape), beta + np.random.uniform(-eps, eps, beta.shape), gamma + np.random.uniform(-eps, eps, gamma.shape)
    non_ortho_coef = non_ortho(x, z, d, dy, opts, alpha, beta, gamma)
    l1_non_ortho = np.linalg.norm(non_ortho_coef - theta, ord=1)
    l2_non_ortho = np.linalg.norm(non_ortho_coef - theta, ord=2)
    print('Direct coef:' + ', '.join(["({}: {:.3f})".format(ind, c) for ind, c in enumerate(non_ortho_coef) if np.abs(c)>0.001]))
    print('Direct l1: {:.3f}'.format(l1_non_ortho))
    
    # Orthogonal lasso estimation
    ortho_coef = ortho(x, z, d, dy, opts, alpha, beta, gamma)
    l1_cross_ortho = np.linalg.norm(ortho_coef - theta, ord=1)
    l2_cross_ortho = np.linalg.norm(ortho_coef - theta, ord=2)
    print('CrossOrtho coef:' + ', '.join(["({}: {:.3f})".format(ind, c) for ind, c in enumerate(ortho_coef) if np.abs(c)>0.001]))
    print('CrossOrtho l1: {:.3f}'.format(l1_cross_ortho))

    return l1_direct, l2_direct, l1_non_ortho, l2_non_ortho, l1_cross_ortho, l2_cross_ortho, direct_coef, non_ortho_coef, ortho_coef

def main(opts, target_dir='.', reload_results=True):
    random_seed = 123

    results_file = os.path.join(target_dir, 'logistic_te_errors_{}.jbl'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()])))
    if reload_results and os.path.exists(results_file):
        results = joblib.load(results_file)
    else:
        results = Parallel(n_jobs=-1, verbose=1)(
                delayed(experiment)(random_seed + exp_id, opts) 
                for exp_id in range(opts['n_experiments']))
    joblib.dump(results, results_file)
    
    l1_direct = np.array([results[i][0] for i in range(opts['n_experiments'])])
    l2_direct = np.array([results[i][1] for i in range(opts['n_experiments'])])
    l1_non_ortho = np.array([results[i][2] for i in range(opts['n_experiments'])])
    l2_non_ortho = np.array([results[i][3] for i in range(opts['n_experiments'])])
    l1_cross_ortho = np.array([results[i][4] for i in range(opts['n_experiments'])])
    l2_cross_ortho = np.array([results[i][5] for i in range(opts['n_experiments'])])
    direct_coef = np.array([results[i][6] for i in range(opts['n_experiments'])])
    non_ortho_coef = np.array([results[i][7] for i in range(opts['n_experiments'])])
    ortho_coef = np.array([results[i][8] for i in range(opts['n_experiments'])])

    plt.figure(figsize=(3 * ortho_coef.shape[1], 3))
    for i in range(ortho_coef.shape[1]):
        plt.subplot(3, ortho_coef.shape[1], i + 1)
        plt.hist(direct_coef[:, i])
        plt.title("$\mu$: {:.2f}, $\sigma$: {:.2f}".format(np.mean(direct_coef[:, i]), np.std(direct_coef[:, i])))
    for i in range(ortho_coef.shape[1]):
        plt.subplot(3, ortho_coef.shape[1], ortho_coef.shape[1] + i + 1)
        plt.hist(non_ortho_coef[:, i])
        plt.title("$\mu$: {:.2f}, $\sigma$: {:.2f}".format(np.mean(non_ortho_coef[:, i]), np.std(non_ortho_coef[:, i])))
    for i in range(ortho_coef.shape[1]):
        plt.subplot(3, ortho_coef.shape[1], 2*ortho_coef.shape[1] + i + 1)
        plt.hist(ortho_coef[:, i])
        plt.title("$\mu$: {:.2f}, $\sigma$: {:.2f}".format(np.mean(ortho_coef[:, i]), np.std(ortho_coef[:, i])))
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'dist_{}.png'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))), dpi=300)
    plt.close()


    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.violinplot([np.array(l2_direct), np.array(l2_non_ortho), np.array(l2_cross_ortho)], showmedians=True)
    plt.xticks([1, 2, 3], ['direct', 'non_ortho', 'ortho'])
    plt.ylabel('$\ell_2$ error')
    plt.subplot(1, 3, 2)
    plt.violinplot([np.array(l2_direct) - np.array(l2_cross_ortho), np.array(l2_direct) - np.array(l2_non_ortho)], showmedians=True)
    plt.xticks([1, 2], ['direct vs ortho', 'direct vs non_ortho'])
    plt.ylabel('$\ell_2$ error decrease')
    plt.subplot(1, 3, 3)
    plt.violinplot([np.array(l2_non_ortho) - np.array(l2_cross_ortho)], showmedians=True)
    plt.xticks([1], ['non_ortho vs ortho'])
    plt.ylabel('$\ell_2$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'l2_errors_{}.png'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    return l1_direct, l1_cross_ortho, l2_direct, l2_cross_ortho

if __name__=="__main__":

    opts= {'n_experiments': 10, # number of monte carlo experiments
            'n_samples': 50000, # samples used for estimation
            'dim_x': 20, # dimension of controls x
            'dim_z': 50, # dimension of variables used for heterogeneity (subset of x)
            'kappa_theta': 2, # support size of target parameter
            'kappa_z': 30, # support size of nuisance
            'sigma_epsilon': 1., # variance of error in secondary moment equation
            'sigma_eta': .1, # variance of error in secondary moment equation, i.e. multiplier in logistic index
            'sigma_x': 3, # variance parameter for co-variate distribution
            'lambda_coef': 0.5, # coeficient in front of the asymptotic rate for regularization lambda
            'n_folds': 2, # number of folds used in cross-fitting
    }
    reload_results = False
    target_dir = 'results_missing'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    main(opts, target_dir=target_dir, reload_results=reload_results)
    