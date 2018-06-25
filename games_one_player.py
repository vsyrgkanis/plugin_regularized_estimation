'''
@author: Vasilis Syrgkanis
'''
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pylab as plt
#import matplotlib.mlab as mlab
import multiprocessing
import joblib
from joblib import Parallel, delayed
from itertools import product
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model, clone
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from scipy_logistic_with_gradient_correction import LogisticWithGradientCorrection
#from tf_logistic_with_gradient_correction import LogisticWithGradientCorrection

'''
Data Generation
'''
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def gen_data(n_samples, n_dim, kappa_gamma, sigma_x=1, kappa_gamma_aux=1):
    '''
    Compute stylized dgp where opponents strategy is really a logistic regression
    '''
    n_players = 2
    # Market interaction coefficients
    beta = np.zeros(n_players)
    beta[0] = np.random.uniform(-2, -1, 1)
    # Feature related coefficients for each player
    gamma = np.zeros((n_players, n_dim))
    support_gamma = np.random.choice(np.arange(0, n_dim), kappa_gamma, replace=False)
    gamma[0, support_gamma] = np.random.uniform(-1, 1, size=kappa_gamma)
    comp_support_gamma = np.array(list(set(np.arange(0, n_dim)) - set(support_gamma)))
    support_gamma_aux =  np.concatenate((np.random.choice(comp_support_gamma, kappa_gamma_aux - kappa_gamma, replace=False), support_gamma), axis=0)
    gamma[1, support_gamma_aux] = np.random.uniform(-1, 1, size=kappa_gamma_aux)
    # Sample standard normal features
    x_samples = np.random.uniform(-sigma_x, sigma_x, (n_samples, n_dim))
    # Matrix of entry probability for each feature vector
    sigma = np.zeros((n_samples, n_players))
    # Matrix of entry decisions for each feature vector
    y = np.zeros((n_samples, n_players))

    # Compute equilibrium probabilities for each feature vector
    sigma[:, 1] = sigmoid(np.dot(x_samples, gamma[1, :].reshape(-1, 1))).flatten()
    sigma[:, 0] = sigmoid(np.dot(x_samples, gamma[0, :].reshape(-1, 1)) + beta[0] * sigma[:, 1].reshape(-1, 1)).flatten()
    
    # Draw sample of entry decisions from probabilities of entry
    y_samples = np.random.binomial(1, sigma[:, 0])
    y_samples_op = np.random.binomial(1, sigma[:, 1])

    return x_samples, y_samples, y_samples_op, sigma, beta, gamma
    
'''
Estimation
'''

def first_stage_sigma(X, y_op, opts):
    '''
    Train first stage estimates of opponent entry with Logistic Lasso
    '''
    est = linear_model.LogisticRegressionCV()
    est.fit(X, y_op.ravel())
    return est


def second_stage_logistic(X, y, sigma_hat_op, opts):
    n_samples, n_dim = X.shape
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(n_dim)/n_samples)
    estimator = LogisticWithGradientCorrection(alpha_l1=l1_reg, alpha_l2=0., tol=1e-6)
    estimator.fit(np.concatenate((X, sigma_hat_op), axis=1), y)
    return estimator


def two_stage_no_split(X, y, y_op, opts):
    n_samples = X.shape[0]
    est = first_stage_sigma(X, y_op, opts)
    sigma_hat_op = est.predict_proba(X)[:, [1]]
    final_est = second_stage_logistic(X, y, sigma_hat_op, opts)
    return final_est.coef_.flatten()[-1], final_est.coef_.flatten()[:-1]

def two_stage_non_orthogonal(X, y, y_op, opts):
    n_samples = X.shape[0]
    kf = KFold(n_splits=3)
    sigma_hat_op = np.zeros((n_samples, 1))
    for train, test in kf.split(X):
        est = first_stage_sigma(X[train], y_op[train], opts)
        sigma_hat_op[test, :] = est.predict_proba(X[test])[:, [1]]
    final_est = second_stage_logistic(X, y, sigma_hat_op, opts)
    return final_est.coef_.flatten()[-1], final_est.coef_.flatten()[:-1]

def two_stage_crossfit_orthogonal(X, y, y_op, opts):
    n_samples, n_dim = X.shape
    
    kf = KFold(n_splits=3)
    sigma_hat_op = np.zeros((n_samples, 1))
    grad_corrections = np.zeros((n_samples, 1))
    for train, test in kf.split(X):
        # Fit on train
        first_est = first_stage_sigma(X[train], y_op[train], opts)
        sigma_hat_train = first_est.predict_proba(X[train])[:, [1]]
        estimator_prel = second_stage_logistic(X[train], y[train], sigma_hat_train, opts)
        # Predict on test
        beta_prel = estimator_prel.coef_.flatten()[-1]
        sigma_hat_op[test] = first_est.predict_proba(X[test])[:, [1]]
        g_prel = estimator_prel.predict_proba(np.concatenate((X[test], sigma_hat_op[test]), axis=1))[:, [1]]
        grad_corrections[test] = beta_prel * g_prel * (1 - g_prel) * (y_op[test].reshape(-1, 1) - sigma_hat_op[test])

    # Final stage estimation
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(n_dim)/(n_samples))
    estimator = LogisticWithGradientCorrection(alpha_l1=l1_reg, alpha_l2=0., tol=1e-6)
    estimator.fit(np.concatenate((X, sigma_hat_op), axis=1), y, grad_corrections=grad_corrections)
    beta_est = estimator.coef_.flatten()[-1]
    gamma_est = estimator.coef_.flatten()[:-1]
    
    print("Max correction: {}".format(np.linalg.norm(grad_corrections.flatten(), ord=np.inf)))
    
    # Final stage without correction
    estimator = LogisticWithGradientCorrection(alpha_l1=l1_reg, alpha_l2=0., tol=1e-6)
    estimator.fit(np.concatenate((X, sigma_hat_op), axis=1), y)
    beta_est_no_cor = estimator.coef_.flatten()[-1]
    gamma_est_no_cor = estimator.coef_.flatten()[:-1]

    return beta_est, gamma_est, beta_est_no_cor, gamma_est_no_cor


def experiment(exp_id, opts):
    np.random.seed(exp_id)
    # Draw a sample of (feature, equilibrium) data points
    X, y, y_op, true_sigma, true_beta, true_gamma = gen_data(
        opts['n_samples'], opts['n_dim'], opts['kappa_gamma'],
        sigma_x=opts['sigma_x'], kappa_gamma_aux=opts['kappa_gamma_aux'])
    s00 = np.sum((y==0) & (y_op==0))
    s01 = np.sum((y>0) & (y_op==0))
    s10 = np.sum((y==0) & (y_op>0))
    s11 = np.sum((y>0) & (y_op>0))
    
    plt.figure()
    plt.hist(true_sigma)
    plt.legend(['player 1', 'player 2'])
    plt.savefig('entry_probs.png')
    plt.close()

    print("Statistics: (0, 0): {}, (0, 1): {}, (1, 0): {}, (1, 1): {}".format(s00, s01, s10, s11))
    true_theta = np.concatenate(([true_beta[0]], true_gamma[0, :]))
    print('True coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(true_theta) if np.abs(c)>0.01]))

    first_est = first_stage_sigma(X, y_op, opts)
    print("True gamma_op:" + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(true_gamma[1, :]) if np.abs(c)>0.01]))
    print("First gamma_op:" + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(first_est.coef_.flatten()) if np.abs(c)>0.01]))
    print("Ell_1 first stage error: {}".format(np.linalg.norm(first_est.coef_.flatten() - true_gamma[1,:].flatten(), ord=1)))
    first_stage_l1 = np.linalg.norm(first_est.coef_.flatten() - true_gamma[1,:].flatten(), ord=1)

    # Simple logistic after first stage
    beta_direct, gamma_direct = two_stage_non_orthogonal(X, y, y_op, opts)
    theta_direct = np.concatenate(([beta_direct], gamma_direct))
    l1_direct = np.linalg.norm(theta_direct.flatten() - true_theta.flatten(), ord=1)
    l2_direct = np.linalg.norm(theta_direct.flatten() - true_theta.flatten(), ord=2)
    print('Direct coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(theta_direct) if np.abs(c)>0.01]))
    
    # Orthogonal lasso estimation
    beta_ortho, gamma_ortho, beta_ortho_no_cor, gamma_ortho_no_cor = two_stage_crossfit_orthogonal(X, y, y_op, opts)
    theta_ortho = np.concatenate(([beta_ortho], gamma_ortho))
    l1_ortho = np.linalg.norm(theta_ortho.flatten() - true_theta.flatten(), ord=1)
    l2_ortho = np.linalg.norm(theta_ortho.flatten() - true_theta.flatten(), ord=2)
    print('Ortho coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(theta_ortho) if np.abs(c)>0.01]))

    theta_ortho_no_cor = np.concatenate(([beta_ortho_no_cor], gamma_ortho_no_cor))
    l1_ortho_no_cor = np.linalg.norm(theta_ortho_no_cor.flatten() - true_theta.flatten(), ord=1)
    l2_ortho_no_cor = np.linalg.norm(theta_ortho_no_cor.flatten() - true_theta.flatten(), ord=2)
    print('Ortho_no_cor coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(theta_ortho_no_cor) if np.abs(c)>0.01]))

    oracle_est = second_stage_logistic(X, y, true_sigma[:, [1]], opts)
    beta_oracle, gamma_oracle = oracle_est.coef_.flatten()[-1], oracle_est.coef_.flatten()[:-1]
    theta_oracle = np.concatenate(([beta_oracle], gamma_oracle))
    l1_oracle = np.linalg.norm(theta_oracle.flatten() - true_theta.flatten(), ord=1)
    l2_oracle = np.linalg.norm(theta_oracle.flatten() - true_theta.flatten(), ord=2)
    print('Oracle coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(theta_oracle) if np.abs(c)>0.01]))



    return l1_direct, l1_ortho, l2_direct, l2_ortho, l1_ortho_no_cor, l2_ortho_no_cor, l1_oracle, l2_oracle, first_stage_l1

def main(opts, target_dir='.', reload_results=True):
    random_seed = 123

    results_file = os.path.join(target_dir, 'games_errors_{}.jbl'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()])))
    if reload_results and os.path.exists(results_file):
        results = joblib.load(results_file)
    else:
        results = Parallel(n_jobs=-1, verbose=1)(
                delayed(experiment)(random_seed + exp_id, opts) 
                for exp_id in range(opts['n_experiments']))
    results = np.array(results)
    joblib.dump(results, results_file)
    
    l1_direct = results[:, 0]
    l1_ortho = results[:, 1]
    l2_direct = results[:, 2]
    l2_ortho = results[:, 3]
    l1_ortho_no_cor = results[:, 4]
    l2_ortho_no_cor = results[:, 5]
    l1_oracle = results[:, 6]
    l2_oracle = results[:, 7]
    first_stage_l1 = results[:, 8]
    
    
    plt.figure(figsize=(8, 3))
    plt.subplot(1,2,1)
    plt.scatter(first_stage_l1, l2_direct, label='direct')
    plt.scatter(first_stage_l1, l2_ortho, label='ortho')
    plt.scatter(first_stage_l1, l2_ortho_no_cor, label='ortho_no_cor')
    plt.xlabel('First stage $\ell_1$')
    plt.ylabel('Final $\ell_2$')
    plt.legend()
    plt.subplot(1,2,2)
    thr = np.median(first_stage_l1)
    plt.violinplot([l2_direct[first_stage_l1>thr] - l2_ortho[first_stage_l1>thr], l2_direct[first_stage_l1>thr] - l2_ortho_no_cor[first_stage_l1>thr]], showmedians=True)
    plt.xticks([1, 2], ['direc vs ortho', 'direct vs no_cor'])
    plt.ylabel('$\ell_2$ error with high first stage')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'error_correlation_{}.png'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.violinplot([l2_direct, l2_ortho, l2_ortho_no_cor, l2_oracle], showmedians=True)
    plt.xticks([1, 2, 3, 4], ['direct', 'ortho', 'no_cor', 'oracle'])
    plt.ylabel('$\ell_2$ error')
    plt.subplot(1, 2, 2)
    plt.violinplot([l2_direct - l2_ortho, l2_direct - l2_ortho_no_cor, l2_direct - l2_oracle], showmedians=True)
    plt.xticks([1, 2, 3], ['direct vs ortho', 'direct vs no_cor', 'direct vs oracle'])
    plt.ylabel('$\ell_2$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'l2_{}.png'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.violinplot([np.array(l1_direct), np.array(l1_ortho), np.array(l1_ortho_no_cor)], showmedians=True)
    plt.xticks([1, 2, 3], ['direct', 'ortho', 'no_cor'])
    plt.ylabel('$\ell_1$ error')
    plt.subplot(1, 2, 2)
    plt.violinplot([np.array(l1_direct) - np.array(l1_ortho), np.array(l1_direct) - np.array(l1_ortho_no_cor)], showmedians=True)
    plt.xticks([1, 2], ['direct vs ortho', 'direct vs no_cor'])
    plt.ylabel('$\ell_1$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'l1_{}.png'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    return l1_direct, l1_ortho, l2_direct, l2_ortho


if __name__ == '__main__':
    '''
    Entry game: definition of parameters. Utility of each player is of the form:
    u_i = beta_i \sum_{j\neq i} \sigma_j(x) + <gamma_i, x>
    where beta_i is the interaction coefficient, gamma_i
    are the feature related coefficients, x is the feature vector, sigma_j(x) is the
    entry probability of player j
    '''
    opts= {'n_experiments': 10, # number of monte carlo experiments
            'n_samples': 80000, # samples used for estimation
            'n_dim': 5000, # dimension of the feature space
            'sigma_x': 1, # variance of the features
            'kappa_gamma': 10, # support size of the target parameter
            'kappa_gamma_aux': 50, # support size for opponent in stylized dgp
            'lambda_coef': 1., # coeficient in front of the asymptotic rate for regularization lambda
    }
    reload_results = False
    target_dir = 'results_games'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    main(opts, target_dir=target_dir, reload_results=reload_results)
    