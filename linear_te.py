import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, MultiTaskLasso, MultiTaskLassoCV, LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import joblib

def gen_data(n_samples, dim_x, dim_z, kappa_x, kappa_theta, sigma_eta, sigma_epsilon):
    """ Generate data from:
    y = <z[support_theta], theta> * t + <x[support_x], alpha_x> + epsilon
    t = <x[support_x], beta_x> + eta
    epsilon ~ Normal(0, sigma_epsilon)
    eta ~ Normal(0, sigma_eta)
    z = (x[subset_z], 1) for some subset of x of size dim_z - 1
    alpha_x, beta_x, theta are all equal to 1
    support_x, support_theta, subset_z drawn uniformly at random
    """
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
    t = np.dot(x[:, support_x], beta_x) + np.random.normal(0, sigma_eta, size=(n_samples, 1))
    y = np.dot(z[:, support_theta], theta) * t + np.dot(x[:, support_x], alpha_x) + np.random.normal(0, sigma_epsilon, size=(n_samples, 1))
    return x, t, z, y, support_x, support_theta, alpha_x, beta_x, theta


def cross_product(X1, X2):
    """ Cross product of features in matrices X1 and X2
    """
    n = np.shape(X1)[0]
    assert n == np.shape(X2)[0]
    return (X1.reshape(n,1,-1) * X2.reshape(n,-1,1)).reshape(n,-1) 

def dml_fit(x, t, z, y, model_t, model_y, model_f):
    """ Orthogonal estimation of coefficient theta
    """
    n_samples = x.shape[0]
    comp_x = cross_product(z, x)
    
    model_t.fit(x[:n_samples//2], t[:n_samples//2].flatten())
    model_y.fit(comp_x[:n_samples//2], y[:n_samples//2].flatten())
    res_t = t[n_samples//2:] - model_t.predict(x[n_samples//2:]).reshape((n_samples//2, -1))
    res_y = y[n_samples//2:] - model_y.predict(comp_x[n_samples//2:]).reshape((n_samples//2, -1))
    model_f.fit(z[n_samples//2:]*res_t, res_y.flatten())
    return model_f.coef_.flatten(), model_t.coef_.flatten(), model_y.coef_.flatten()

def dml_crossfit(x, t, z, y, model_t, model_y, model_f):
    """ Orthogonal estimation of coefficient theta with cross-fitting
    """
    n_samples = x.shape[0]
    comp_x = cross_product(z, x)
    
    kf = KFold(n_splits=2)
    res_y = np.zeros(y.shape)
    res_t = np.zeros(t.shape)
    for train_index, test_index in kf.split(x):
        model_t.fit(x[train_index], t[train_index].flatten())
        model_y.fit(comp_x[train_index], y[train_index].flatten())
        res_t[test_index] = t[test_index] - model_t.predict(x[test_index]).reshape(test_index.shape[0], -1)
        res_y[test_index] = y[test_index] - model_y.predict(comp_x[test_index]).reshape(test_index.shape[0], -1)
    
    model_f.fit(z*res_t, res_y.flatten())
    return model_f.coef_.flatten(), model_t.coef_.flatten(), model_y.coef_.flatten()


def experiment(exp_id, n_samples, dim_x, dim_z, kappa_x, kappa_theta, sigma_eta, sigma_epsilon, lambda_coef):
    np.random.seed(exp_id)
    # Generate data
    x, t, z, y, support_x, support_theta, alpha_x, beta_x, theta = gen_data(n_samples, dim_x, dim_z, kappa_x, kappa_theta, sigma_eta, sigma_epsilon)

    true_coef = np.zeros((dim_z,1))
    true_coef[support_theta] = theta

    # Direct lasso for all coefficients
    ols = Lasso(alpha=np.sqrt(np.log(dim_z + dim_x)/n_samples))
    ols.fit(np.concatenate((z*t, x), axis=1), y.flatten())
    l1_direct = np.linalg.norm(ols.coef_.flatten()[:dim_z] - true_coef.flatten(), ord=1)
    l2_direct = np.linalg.norm(ols.coef_.flatten()[:dim_z] - true_coef.flatten(), ord=2)
    print("done ols")
    # Orthogonal lasso estimation
    ortho_coef, t_coef, y_coef = dml_fit(x, t, z, y,
                                            Lasso(alpha=lambda_coef * np.sqrt(np.log(dim_x)*2. / n_samples)),
                                            Lasso(alpha=lambda_coef * np.sqrt(np.log(dim_z*dim_x)*2. / n_samples)),
                                            Lasso(alpha=lambda_coef * np.sqrt(np.log(dim_z)*2. / n_samples), fit_intercept=False))
    l1_ortho = np.linalg.norm(ortho_coef - true_coef.flatten(), ord=1)
    l2_ortho = np.linalg.norm(ortho_coef - true_coef.flatten(), ord=2)
    print("done ortho")
    # Crossfit Orthogonal lasso estimation
    ortho_coef, t_coef, y_coef = dml_crossfit(x, t, z, y,
                                                Lasso(alpha=lambda_coef * np.sqrt(np.log(dim_x)*2. / n_samples)),
                                                Lasso(alpha=lambda_coef * np.sqrt(np.log(dim_z*dim_x)*2. / n_samples)),
                                                Lasso(alpha=lambda_coef * np.sqrt(np.log(dim_z)*2. / n_samples), fit_intercept=False))
    l1_cross_ortho = np.linalg.norm(ortho_coef - true_coef.flatten(), ord=1)
    l2_cross_ortho = np.linalg.norm(ortho_coef - true_coef.flatten(), ord=2)
    print("done crossfit ortho")
    return l1_direct, l1_ortho, l1_cross_ortho, l2_direct, l2_ortho, l2_cross_ortho

def main(opts, target_dir='.', reload_results=True):
    random_seed = 123
    print("Running")
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
                                                            opts['sigma_epsilon'],
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
    plt.xticks([1,2], ['direct - ortho', 'direct - crossfit_ortho'])
    plt.ylabel('$\ell_2$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'linear_te_l2_errors_{}.pdf'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.violinplot([np.array(l1_direct) - np.array(l1_ortho), np.array(l1_direct) - np.array(l1_cross_ortho)], showmedians=True)
    plt.xticks([1,2], ['direct - ortho', 'direct - crossfit_ortho'])
    plt.ylabel('$\ell_1$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'linear_te_l1_errors_{}.pdf'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    return l1_direct, l1_ortho, l1_cross_ortho, l2_direct, l2_ortho, l2_cross_ortho

if __name__=="__main__":
    
    opts= {'n_experiments': 100, # number of monte carlo experiments
            'n_samples': 5000, # samples used for estimation
            'dim_x': 200, # dimension of controls x
            'dim_z': 200, # dimension of variables used for heterogeneity (subset of x)
            'kappa_theta': 2, # support size of target parameter
            'sigma_eta': 1, # variance of error in secondary moment equation
            'sigma_epsilon': 1, # variance of error in primary moment equation
            'lambda_coef': 1 # coeficient in front of the asymptotic rate for regularization lambda
    }

    target_dir = 'results_corrected'
    reload_results = True

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    kappa_grid = np.arange(2, 40, 3)
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

    joblib.dump(np.array(l2_direct_list), os.path.join(target_dir, 'linear_te_l2_direct_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l1_direct_list), os.path.join(target_dir, 'linear_te_l1_direct_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l2_ortho_list), os.path.join(target_dir, 'linear_te_l2_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l1_ortho_list), os.path.join(target_dir, 'linear_te_l1_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l2_cross_ortho_list), os.path.join(target_dir, 'linear_te_l2_cross_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    joblib.dump(np.array(l1_cross_ortho_list), os.path.join(target_dir, 'linear_te_l1_cross_ortho_with_growing_kappa_x_{}.jbl'.format(param_str)))
    
    plt.figure(figsize=(5, 3))
    plt.plot(kappa_grid, np.median(l2_direct_list, axis=1), label='direct')
    plt.fill_between(kappa_grid, np.percentile(l2_direct_list, 95, axis=1), np.percentile(l2_direct_list, 5, axis=1), alpha=0.3)
    plt.plot(kappa_grid, np.median(l2_ortho_list, axis=1), label='ortho')
    plt.fill_between(kappa_grid, np.percentile(l2_ortho_list, 95, axis=1), np.percentile(l2_ortho_list, 5, axis=1), alpha=0.3)
    plt.plot(kappa_grid, np.median(l2_cross_ortho_list, axis=1), label='cross_ortho')
    plt.fill_between(kappa_grid, np.percentile(l2_cross_ortho_list, 95, axis=1), np.percentile(l2_cross_ortho_list, 5, axis=1), alpha=0.3)
    plt.legend()
    #plt.title('$\ell_2$ error')
    plt.xlabel('support size $k_g$')
    plt.ylabel('$\ell_2$ error')
    plt.tight_layout()
    param_str = '_'.join(['{}_{}'.format(k, v) for k,v in opts.items() if k!='kappa_x'])
    plt.savefig(os.path.join(target_dir, 'linear_te_growing_kappa_x_{}.pdf'.format(param_str)))
