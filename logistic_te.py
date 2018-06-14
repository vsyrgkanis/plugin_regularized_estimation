import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from logistic_with_offset import LogisticWithOffset
from joblib import Parallel, delayed
import joblib


def gen_data(n_samples, dim_x, dim_z, kappa_x, kappa_theta, sigma_eta):
    """ Generate data from:
    Pr[y|x,t] = Sigmoid(<z[support_theta], theta> * t + <x[support_x], alpha_x>)
    t = <x[support_x], beta_x> + eta
    epsilon ~ Normal(0, sigma_epsilon)
    eta ~ Normal(0, sigma_eta)
    z = x[subset_z] for some subset of x of size dim_z
    alpha_x, beta_x, theta are all equal to 1
    support_x, support_theta, subset_z drawn uniformly at random
    """
    x = np.random.uniform(-.2, .2, size=(n_samples, dim_x))
    supp_z = np.random.choice(np.arange(x.shape[1]), dim_z, replace=False)
    z = x[:, supp_z].reshape(n_samples, -1)
        
    support_x = np.random.choice(np.arange(0, dim_x), kappa_x, replace=False)
    support_theta = np.random.choice(np.arange(0, dim_z), kappa_theta, replace=False)
    alpha_x = np.random.uniform(1, 1, size=(kappa_x, 1))
    beta_x = np.random.uniform(1, 1, size=(kappa_x, 1))
    theta = np.random.uniform(1, 1, size=(kappa_theta, 1))
    t = np.dot(x[:, support_x], beta_x) + np.random.normal(0, sigma_eta, size=(n_samples, 1))
    index_y = np.dot(z[:, support_theta], theta) * t + np.dot(x[:, support_x], alpha_x)
    p_y = 1/(1+np.exp(-index_y))
    y = np.random.binomial(1, p_y)
    return x, t, z, y, support_x, support_theta, alpha_x, beta_x, theta

def direct_fit(x, t, z, y):
    n_samples = x.shape[0]
    
    model_t = Lasso(alpha=np.sqrt(np.log(x.shape[1])/n_samples))
    model_t.fit(x, t.ravel())

    comp_x = np.concatenate((z * t, x), axis=1)
    l1_reg = np.sqrt(np.log(comp_x.shape[1])/(4*n_samples))
    model_y = LogisticRegression(penalty='l1', C=1./l1_reg)
    model_y.fit(comp_x, y.ravel())
    
    return model_y, model_t
    

def first_stage_estimates(x, t, z, y, tr_inds, tst_inds):
    n_samples = x.shape[0]
    comp_x = np.concatenate((z * t, x), axis=1)
    
    model_y, model_t = direct_fit(x[tr_inds], t[tr_inds], z[tr_inds], y[tr_inds])
    
    # \tilde{theta}
    theta_prel = model_y.coef_.flatten()[:z.shape[1]].reshape(-1, 1)
    # \tilde{f}(u) = u'alpha_prel
    alpha_prel = model_y.coef_.flatten()[z.shape[1]:].reshape(-1, 1)
    # \tilde{h}(u) = u'beta_prel
    beta_prel = model_t.coef_.flatten().reshape(-1, 1)
    # \tilde{\pi}(u) = model_t.predict(u)
    t_test_pred = model_t.predict(x[tst_inds]).reshape(-1, 1)
    # G_prel = model_y.predict([x, u])
    y_test_pred = model_y.predict_proba(comp_x[tst_inds])[:, 1].reshape(-1, 1)
    # \hat{q}(u) = \pi(u) * B(u)'\tilde{theta} + u'\tilde{alpha}
    q_test = t_test_pred * np.dot(z[tst_inds], theta_prel) + np.dot(x[tst_inds], alpha_prel)
    # V(z) = G(index) * (1 - G(index))
    V_test = y_test_pred * (1 - y_test_pred)    
    # res = x - \hat{h}(u) = B(u)*(tau - pi(u))
    res_test = t[tst_inds] - t_test_pred
    comp_res_test = z[tst_inds] * res_test

    return comp_res_test, q_test, V_test

def dml_fit(x, t, z, y):
    """ Orthogonal estimation of coefficient theta
    """
    n_samples = x.shape[0]
    tr_inds, tst_inds = np.arange(n_samples//2), np.arange(n_samples//2, n_samples)
    comp_res_test, q_test, V_test = first_stage_estimates(x, t, z, y, tr_inds, tst_inds)
    sample_weights_test = (1./np.clip(V_test, 0.0001, 1))/np.mean((1./np.clip(V_test, 0.0001, 1)))

    steps = 10000
    lr = 1/np.sqrt(steps)
    l1_reg = np.sqrt(np.log(z.shape[1])/(4*tst_inds.shape[0]))
    model_final = LogisticWithOffset(alpha_l1=l1_reg, alpha_l2=0.,\
                                 steps=steps, learning_rate=lr, tol=1e-7)
    
    model_final.fit(comp_res_test, y[tst_inds], offset=q_test, sample_weights=sample_weights_test)
    
    return model_final.coef_.flatten()


def dml_crossfit(x, t, z, y):
    """ Orthogonal estimation of coefficient theta with cross-fitting
    """
    n_samples = x.shape[0]
    
    kf = KFold(n_splits=4)
    comp_res = np.zeros(z.shape)
    offset = np.zeros((x.shape[0], 1))
    V = np.zeros((x.shape[0], 1))
    for train_index, test_index in kf.split(x):
        comp_res[test_index], offset[test_index], V[test_index] = first_stage_estimates(x, t, z, y, train_index, test_index)
    
    sample_weights = (1./np.clip(V, 0.0001, 1))/np.mean((1./np.clip(V, 0.0001, 1)))

    steps = 10000
    lr = 1/np.sqrt(steps)
    l1_reg = np.sqrt(np.log(z.shape[1])/(4*n_samples))
    model_final = LogisticWithOffset(alpha_l1=l1_reg, alpha_l2=0.,\
                                 steps=steps, learning_rate=lr, tol=1e-7)

    model_final.fit(comp_res, y, offset=offset, sample_weights=sample_weights)
    return model_final.coef_.flatten()


def experiment(exp_id, n_samples, dim_x, dim_z, kappa_x, kappa_theta, sigma_eta, lambda_coef):
    np.random.seed(exp_id)
    # Generate data
    x, t, z, y, support_x, support_theta, alpha_x, beta_x, theta =\
            gen_data(n_samples, dim_x, dim_z, kappa_x, kappa_theta, sigma_eta)

    true_coef = np.zeros((dim_z,1))
    true_coef[support_theta] = theta

    # Direct lasso for all coefficients
    model_y, model_t= direct_fit(x, t, z, y)
    l1_direct = np.linalg.norm(model_y.coef_.flatten()[:z.shape[1]] - true_coef.flatten(), ord=1)
    l2_direct = np.linalg.norm(model_y.coef_.flatten()[:z.shape[1]].flatten() - true_coef.flatten(), ord=2)
    
    # Orthogonal lasso estimation
    ortho_coef = dml_fit(x, t, z, y)
    l1_ortho = np.linalg.norm(ortho_coef.flatten() - true_coef.flatten(), ord=1)
    l2_ortho = np.linalg.norm(ortho_coef.flatten() - true_coef.flatten(), ord=2)

    ortho_coef = dml_crossfit(x, t, z, y)
    l1_cross_ortho = np.linalg.norm(ortho_coef - true_coef.flatten(), ord=1)
    l2_cross_ortho = np.linalg.norm(ortho_coef - true_coef.flatten(), ord=2)

    return l1_direct, l1_ortho, l1_cross_ortho, l2_direct, l2_ortho, l2_cross_ortho

def main(opts, target_dir='.', reload_results=True):
    random_seed = 123

    results_file = os.path.join(target_dir, 'logistic_te_errors_{}.jbl'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()])))
    if reload_results and os.path.exists(results_file):
        results = joblib.load(results_file)
    else:
        results = Parallel(n_jobs=-1, verbose=1)(
                delayed(experiment)(random_seed + exp_id, opts['n_samples'], 
                                                            opts['dim_x'], 
                                                            opts['dim_z'], 
                                                            opts['kappa_x'],
                                                            opts['kappa_theta'],
                                                            opts['sigma_eta'],
                                                            opts['lambda_coef']) 
                for exp_id in range(opts['n_experiments']))
    results = np.array(results)    
    joblib.dump(results, results_file)
    
    l1_direct = results[:, 0]
    l1_ortho = results[:, 1]
    l1_cross_ortho = results[:, 2]
    l2_direct = results[:, 3]
    l2_ortho = results[:, 4]
    l2_cross_ortho = results[:, 5]
    
    plt.figure(figsize=(5, 3))
    plt.violinplot([np.array(l2_direct) - np.array(l2_ortho), np.array(l2_direct) - np.array(l2_cross_ortho)], showmedians=True)
    plt.xticks([1,2], ['direct vs ortho', 'direct vs crossfit_ortho'])
    plt.ylabel('$\ell_2$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'logistic_te_l2_errors_{}.pdf'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.violinplot([np.array(l1_direct) - np.array(l1_ortho), np.array(l1_direct) - np.array(l1_cross_ortho)], showmedians=True)
    plt.xticks([1,2], ['direct vs ortho', 'direct vs crossfit_ortho'])
    plt.ylabel('$\ell_1$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'logistic_te_l1_errors_{}.pdf'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    return l1_direct, l1_ortho, l1_cross_ortho, l2_direct, l2_ortho, l2_cross_ortho

if __name__=="__main__":
    
    opts= {'n_experiments': 100, # number of monte carlo experiments
            'n_samples': 5000, # samples used for estimation
            'dim_x': 200, # dimension of controls x
            'dim_z': 200, # dimension of variables used for heterogeneity (subset of x)
            'kappa_theta': 2, # support size of target parameter
            'sigma_eta': 5, # variance of error in secondary moment equation
            'lambda_coef': 1 # coeficient in front of the asymptotic rate for regularization lambda
    }
    reload_results = False
    kappa_grid = np.arange(2, 40, 3)
    target_dir = 'results_logistic_te'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    l2_direct_list = []
    l2_ortho_list = []
    l2_cross_ortho_list = []
    l1_direct_list = []
    l1_ortho_list = []
    l1_cross_ortho_list = []

    for kappa_x in kappa_grid:
        opts['kappa_x'] = kappa_x
        l1_direct, l1_ortho, l1_cross_ortho, l2_direct, l2_ortho, l2_cross_ortho = main(opts, target_dir=target_dir, reload_results=reload_results)
        l2_direct_list.append(l2_direct)
        l2_ortho_list.append(l2_ortho)
        l2_cross_ortho_list.append(l2_cross_ortho)
        l1_direct_list.append(l1_direct)
        l1_ortho_list.append(l1_ortho)
        l1_cross_ortho_list.append(l1_cross_ortho)
    
    param_str = '_'.join(['{}_{}'.format(k, v) for k,v in opts.items() if k!='kappa_x'])

    joblib.dump(np.array(l2_direct_list), os.path.join(target_dir, 'logistic_te_l2_direct_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l1_direct_list), os.path.join(target_dir, 'logistic_te_l1_direct_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l2_ortho_list), os.path.join(target_dir, 'logistic_te_l2_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l1_ortho_list), os.path.join(target_dir, 'logistic_te_l1_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l2_cross_ortho_list), os.path.join(target_dir, 'logistic_te_l2_cross_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l1_cross_ortho_list), os.path.join(target_dir, 'logistic_te_l1_cross_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    
    plt.figure(figsize=(5, 3))
    plt.plot(kappa_grid, np.median(l2_direct_list, axis=1), label='direct')
    plt.fill_between(kappa_grid, np.percentile(l2_direct_list, 100, axis=1), np.percentile(l2_direct_list, 0, axis=1), alpha=0.3)
    plt.plot(kappa_grid, np.median(l2_ortho_list, axis=1), label='ortho')
    plt.fill_between(kappa_grid, np.percentile(l2_ortho_list, 100, axis=1), np.percentile(l2_ortho_list, 0, axis=1), alpha=0.3)
    plt.plot(kappa_grid, np.median(l2_cross_ortho_list, axis=1), label='cross_ortho')
    plt.fill_between(kappa_grid, np.percentile(l2_cross_ortho_list, 100, axis=1), np.percentile(l2_cross_ortho_list, 0, axis=1), alpha=0.3)
    plt.legend()
    plt.xlabel('support size $k_g$')
    plt.ylabel('$\ell_2$ error')
    plt.tight_layout()
    param_str = '_'.join(['{}_{}'.format(k, v) for k,v in opts.items() if k!='kappa_x'])
    plt.savefig(os.path.join(target_dir, 'logistic_te_growing_kappa_x_{}.pdf'.format(param_str)))
