import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy_logistic_with_offset import LogisticWithOffset
from joblib import Parallel, delayed
import joblib


def gen_data(n_samples, dim_x, dim_z, kappa_x, kappa_theta, sigma_eta, sigma_x):
    """ Generate data from:
    Pr[y|x,t] = Sigmoid(<z[support_theta], theta> * t + <x[support_x], alpha_x>)
    t = <x[support_x], beta_x> + eta
    epsilon ~ Normal(0, sigma_epsilon)
    eta ~ Normal(0, sigma_eta)
    z = x[:dim_z]
    alpha_x, beta_x, theta are all equal to 1
    support_x, support_theta, subset_z drawn uniformly at random. support_x contains support_theta
    """
    x = np.random.uniform(-sigma_x, sigma_x, size=(n_samples, dim_x))
    z = x[:, :dim_z].reshape(n_samples, -1)
        
    support_theta = np.random.choice(np.arange(0, dim_z), kappa_theta, replace=False)
    support_x = np.random.choice(np.array(list(set(np.arange(0, dim_x)) - set(support_theta))), kappa_x - kappa_theta, replace=False)
    support_x = np.concatenate((support_x, support_theta), axis=0)
    alpha_x = np.random.uniform(1, 1, size=(kappa_x, 1))
    beta_x = np.random.uniform(1, 1, size=(kappa_x, 1))
    theta = np.random.uniform(1, 1, size=(kappa_theta, 1))
    t = np.dot(x[:, support_x], beta_x) + np.random.normal(0, sigma_eta, size=(n_samples, 1))
    index_y = np.dot(z[:, support_theta], theta) * t + np.dot(x[:, support_x], alpha_x)
    p_y = 1/(1+np.exp(-index_y))
    y = np.random.binomial(1, p_y)
    return x, t, z, y, support_x, support_theta, alpha_x, beta_x, theta

def direct_fit(x, t, z, y, opts):
    n_samples = x.shape[0]
    
    # Run lasso for treatment as function of controls
    model_t = Lasso(alpha=np.sqrt(np.log(x.shape[1])/n_samples))
    model_t.fit(x, t.ravel())

    # Run logistic lasso for outcome as function of composite treatments and controls
    comp_x = np.concatenate((z * t, x), axis=1)
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(comp_x.shape[1])/n_samples)
    model_y = LogisticWithOffset(alpha_l1=l1_reg, 
                                    alpha_l2=0., tol=1e-6)
    model_y.fit(comp_x, y)

    # Return direct models
    return model_y, model_t
    

def first_stage_estimates(x, t, z, y, tr_inds, tst_inds, opts):
    n_samples = x.shape[0]
    comp_x = np.concatenate((z * t, x), axis=1)
    
    # Get direct regression models fitted on the training set
    model_y, model_t = direct_fit(x[tr_inds], t[tr_inds], z[tr_inds], y[tr_inds], opts)
    
    # Compute all required quantities for orthogonal estimation on the test set
    # Preliminary theta estimate: \tilde{theta}
    theta_prel = model_y.coef_.flatten()[:z.shape[1]].reshape(-1, 1)
    # Preliminary estimate of f(u): \hat{f}(u) = u'alpha_prel
    alpha_prel = model_y.coef_.flatten()[z.shape[1]:].reshape(-1, 1)
    # Preliminary estimate of h(u): \hat{h}(u) = u'beta_prel
    beta_prel = model_t.coef_.flatten().reshape(-1, 1)
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

def dml_crossfit(x, t, z, y, opts):
    """ Orthogonal estimation of coefficient theta with cross-fitting
    """
    n_samples = x.shape[0]
    
    # Build first stage nuisance estimates for each sample using cross-fitting
    kf = KFold(n_splits=opts['n_folds'])
    comp_res = np.zeros(z.shape)
    offset = np.zeros((x.shape[0], 1))
    V = np.zeros((x.shape[0], 1))
    for train_index, test_index in kf.split(x):
        comp_res[test_index], offset[test_index], V[test_index] = first_stage_estimates(x, t, z, y, train_index, test_index, opts)
    
    # Calculate normalized sample weights. Clipping for instability
    sample_weights = (1./np.clip(V, 0.01, 1))/np.mean((1./np.clip(V, 0.01, 1)))

    # Fit second stage regression with plugin nuisance estimates
    l1_reg = np.sqrt(np.log(z.shape[1])/n_samples) #+ np.log(x.shape[1])*opts['kappa_x']**2/ ((1 - 1./opts['n_folds']) * n_samples)
    print(np.log(x.shape[1])*opts['kappa_x']**2/ ((1 - 1./opts['n_folds']) * n_samples))
    print(l1_reg)
    model_final = LogisticWithOffset(alpha_l1=opts['lambda_coef'] * l1_reg, 
                                    alpha_l2=0., tol=1e-6)

    model_final.fit(comp_res, y, offset=offset, sample_weights=sample_weights)

    return model_final


def experiment(exp_id, opts):
    np.random.seed(exp_id)
    # Generate data
    x, t, z, y, support_x, support_theta, alpha_x, beta_x, theta =\
            gen_data(opts['n_samples'], opts['dim_x'], opts['dim_z'], opts['kappa_x'],
                        opts['kappa_theta'], opts['sigma_eta'], opts['sigma_x'])

    true_coef = np.zeros((z.shape[1], 1))
    true_coef[support_theta] = theta
    print('True coef:' + ', '.join(["({}: {:.3f})".format(ind, c) for ind, c in enumerate(true_coef.flatten()) if np.abs(c)>0.001]))


    # Direct lasso for all coefficients
    model_y, model_t= direct_fit(x, t, z, y, opts)
    direct_coef = model_y.coef_.flatten()[:z.shape[1]]
    l1_direct = np.linalg.norm(direct_coef - true_coef.flatten(), ord=1)
    l2_direct = np.linalg.norm(direct_coef - true_coef.flatten(), ord=2)
    print('Direct coef:' + ', '.join(["({}: {:.3f})".format(ind, c) for ind, c in enumerate(direct_coef) if np.abs(c)>0.001]))
    print('Direct l1: {:.3f}'.format(l1_direct))
    
    # Orthogonal lasso estimation
    model_dml = dml_crossfit(x, t, z, y, opts)
    ortho_coef = model_dml.coef_.flatten()
    l1_cross_ortho = np.linalg.norm(ortho_coef.flatten() - true_coef.flatten(), ord=1)
    l2_cross_ortho = np.linalg.norm(ortho_coef.flatten() - true_coef.flatten(), ord=2)
    print('CrossOrtho coef:' + ', '.join(["({}: {:.3f})".format(ind, c) for ind, c in enumerate(ortho_coef.flatten()) if np.abs(c)>0.001]))
    print('CrossOrtho l1: {:.3f}'.format(l1_cross_ortho))

    return l1_direct, l1_cross_ortho, l2_direct, l2_cross_ortho

def main(opts, target_dir='.', reload_results=True):
    random_seed = 123

    results_file = os.path.join(target_dir, 'logistic_te_errors_{}.jbl'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()])))
    if reload_results and os.path.exists(results_file):
        results = joblib.load(results_file)
    else:
        results = Parallel(n_jobs=-1, verbose=1)(
                delayed(experiment)(random_seed + exp_id, opts) 
                for exp_id in range(opts['n_experiments']))
    results = np.array(results)    
    joblib.dump(results, results_file)
    
    l1_direct = results[:, 0]
    l1_cross_ortho = results[:, 1]
    l2_direct = results[:, 2]
    l2_cross_ortho = results[:, 3]
    
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.violinplot([np.array(l2_direct),np.array(l2_cross_ortho)], showmedians=True)
    plt.xticks([1, 2], ['direct', 'crossfit_ortho'])
    plt.ylabel('$\ell_2$ error')
    plt.subplot(1, 2, 2)
    plt.violinplot([np.array(l2_direct) - np.array(l2_cross_ortho)], showmedians=True)
    plt.xticks([1], ['direct vs crossfit_ortho'])
    plt.ylabel('$\ell_2$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'logistic_te_l2_errors_{}.pdf'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.violinplot([np.array(l1_direct),np.array(l1_cross_ortho)], showmedians=True)
    plt.xticks([1, 2], ['direct', 'crossfit_ortho'])
    plt.ylabel('$\ell_1$ error')
    plt.subplot(1, 2, 2)
    plt.violinplot([np.array(l1_direct) - np.array(l1_cross_ortho)], showmedians=True)
    plt.xticks([1], ['direct vs crossfit_ortho'])
    plt.ylabel('$\ell_1$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'logistic_te_l1_errors_{}.pdf'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    return l1_direct, l1_cross_ortho, l2_direct, l2_cross_ortho

if __name__=="__main__":
    
    opts= {'n_experiments': 100, # number of monte carlo experiments
            'n_samples': 5000, # samples used for estimation
            'dim_x': 2000, # dimension of controls x
            'dim_z': 2000, # dimension of variables used for heterogeneity (subset of x)
            'kappa_theta': 2, # support size of target parameter
            'sigma_eta': 3, # variance of error in secondary moment equation
            'sigma_x': .5, # variance parameter for co-variate distribution
            'lambda_coef': .5, # coeficient in front of the asymptotic rate for regularization lambda
            'n_folds': 2, # number of folds used in cross-fitting
    }
    reload_results = False
    kappa_grid = np.arange(5, 6, 3)
    target_dir = 'results_logistic_te'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    l2_direct_list = []
    l2_cross_ortho_list = []
    l1_direct_list = []
    l1_cross_ortho_list = []

    for kappa_x in kappa_grid:
        opts['kappa_x'] = kappa_x
        l1_direct, l1_cross_ortho, l2_direct, l2_cross_ortho = main(opts, target_dir=target_dir, reload_results=reload_results)
        l2_direct_list.append(l2_direct)
        l2_cross_ortho_list.append(l2_cross_ortho)
        l1_direct_list.append(l1_direct)
        l1_cross_ortho_list.append(l1_cross_ortho)
    
    param_str = '_'.join(['{}_{}'.format(k, v) for k,v in opts.items() if k!='kappa_x'])

    joblib.dump(np.array(l2_direct_list), os.path.join(target_dir, 'logistic_te_l2_direct_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l1_direct_list), os.path.join(target_dir, 'logistic_te_l1_direct_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l2_cross_ortho_list), os.path.join(target_dir, 'logistic_te_l2_cross_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l1_cross_ortho_list), os.path.join(target_dir, 'logistic_te_l1_cross_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    
    plt.figure(figsize=(5, 3))
    plt.plot(kappa_grid, np.median(l2_direct_list, axis=1), label='direct')
    plt.fill_between(kappa_grid, np.percentile(l2_direct_list, 100, axis=1), np.percentile(l2_direct_list, 0, axis=1), alpha=0.3)
    plt.plot(kappa_grid, np.median(l2_cross_ortho_list, axis=1), label='cross_ortho')
    plt.fill_between(kappa_grid, np.percentile(l2_cross_ortho_list, 100, axis=1), np.percentile(l2_cross_ortho_list, 0, axis=1), alpha=0.3)
    plt.legend()
    plt.xlabel('support size $k_g$')
    plt.ylabel('$\ell_2$ error')
    plt.tight_layout()
    param_str = '_'.join(['{}_{}'.format(k, v) for k,v in opts.items() if k!='kappa_x'])
    plt.savefig(os.path.join(target_dir, 'logistic_te_l2_growing_kappa_x_{}.pdf'.format(param_str)))

    plt.figure(figsize=(5, 3))
    plt.plot(kappa_grid, np.median(l1_direct_list, axis=1), label='direct')
    plt.fill_between(kappa_grid, np.percentile(l1_direct_list, 100, axis=1), np.percentile(l1_direct_list, 0, axis=1), alpha=0.3)
    plt.plot(kappa_grid, np.median(l1_cross_ortho_list, axis=1), label='cross_ortho')
    plt.fill_between(kappa_grid, np.percentile(l1_cross_ortho_list, 100, axis=1), np.percentile(l1_cross_ortho_list, 0, axis=1), alpha=0.3)
    plt.legend()
    plt.xlabel('support size $k_g$')
    plt.ylabel('$\ell_1$ error')
    plt.tight_layout()
    param_str = '_'.join(['{}_{}'.format(k, v) for k,v in opts.items() if k!='kappa_x'])
    plt.savefig(os.path.join(target_dir, 'logistic_te_l1_growing_kappa_x_{}.pdf'.format(param_str)))
