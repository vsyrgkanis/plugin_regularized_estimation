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
Equilibrium computation and Data Generation
'''


def sigmoid(x):
    '''
    The sigmoid function
    '''
    return 1 / (1 + np.exp(-x))


def compute_equilibrium(n_players, x, beta, gamma, epsilon=.0001, iter_cutoff=10000):
    '''
    Compute equilibrium entry probabilities via iterated best response
    '''
    # initialize probabilities
    sigma = .5 * np.ones(n_players)
    # Do iterated best response until convergence to fixed point occurs
    t = 1
    while True:
        sigma_n = np.zeros(n_players)
        for i in range(n_players):
            # Sum of probabilities of entry of opponents
            sigma_minus_i = np.sum(sigma[np.arange(n_players) != i])
            # Logistic best response
            sigma_n[i] = sigmoid(beta[i] * sigma_minus_i + np.dot(gamma[i], x))

        # Check if L infinty norm of the two iterates is smaller than precision
        if np.linalg.norm(sigma_n - sigma, ord=np.inf) <= epsilon:
            break
        # Check if we exceeded maximum iterations for equilibrium computation
        if t > iter_cutoff:
            print("Reached maximum of {} iterations, did not find equilibrium!".format(
                iter_cutoff))
            break

        # Update probability vector to new vector
        for i in range(n_players):
            sigma[i] = sigma_n[i]

        # Increase iteration count
        t = t + 1

    return sigma


def equilibrium_data(n_samples, n_dim, n_players, kappa_gamma, sigma_x=1, epsilon=.0001, iter_cutoff=100):
    '''
    Create equilibrium data with given parameters, i.e. pairs of features
    and probabilities of entry
    '''
    # Market interaction coefficients
    beta = np.random.uniform(-5, -4, n_players)
    # Feature related coefficients for each player
    gamma = np.zeros((n_players, n_dim))
    for player_id in range(n_players):
        support_gamma = np.random.choice(np.arange(0, n_dim), kappa_gamma, replace=False)
        gamma[player_id, support_gamma] = 1
    # Sample standard normal features
    x_samples = np.random.normal(0, sigma_x, (n_samples, n_dim))
    # Matrix of entry probability for each feature vector
    sigma = .5 * np.ones((n_samples, n_players))
    # Matrix of entry decisions for each feature vector
    y = np.zeros((n_samples, n_players))

    # Compute equilibrium probabilities for each feature vector
    for s_id in range(n_samples):
        sigma[s_id, :] = compute_equilibrium(
            n_players, x_samples[s_id], beta, gamma, epsilon, iter_cutoff)
    
    # Draw sample of entry decisions from probabilities of entry
    y_samples = np.random.binomial(1, sigma)

    return x_samples, y_samples, sigma, beta, gamma

def stylized_data(n_samples, n_dim, kappa_gamma, sigma_x=1, kappa_gamma_aux=1):
    '''
    Compute stylized dgp where opponents strategy is really a logistic regression
    '''
    n_players = 2
    # Market interaction coefficients
    beta = np.zeros(n_players)
    beta[0] = np.random.uniform(-5, -4, 1)
    # Feature related coefficients for each player
    gamma = np.zeros((n_players, n_dim))
    support_gamma = np.random.choice(np.arange(0, n_dim), kappa_gamma, replace=False)
    gamma[0, support_gamma] = 1
    comp_support_gamma = np.array(list(set(np.arange(0, n_dim)) - set(support_gamma)))
    support_gamma_aux =  np.concatenate((np.random.choice(comp_support_gamma, kappa_gamma_aux - kappa_gamma, replace=False), support_gamma), axis=0)
    gamma[1, support_gamma_aux] = 1
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
    y_samples = np.random.binomial(1, sigma)

    return x_samples, y_samples, sigma, beta, gamma

def gen_data(n_samples, n_dim, n_players, kappa_gamma, sigma_x=1, epsilon=.0001, iter_cutoff=100, dgp_type='stylized', kappa_gamma_aux=1):
    if dgp_type == 'stylized':
        return stylized_data(n_samples, n_dim, kappa_gamma, sigma_x=sigma_x, kappa_gamma_aux=kappa_gamma_aux)
    else:
        return equilibrium_data(n_samples, n_dim, n_players, kappa_gamma, sigma_x=sigma_x, epsilon=epsilon, iter_cutoff=iter_cutoff)

'''
Estimation
'''

def first_stage_sigma(x, y, tr_inds, tst_inds):
    '''
    Train first stage estimates with ANN or Logistic or KernelRidge
    '''
    first_stage_estimators = []
    for player_id in range(y.shape[1]):
        param_grid = {'C': np.array([0.01, 0.1, 1, 10, 100, 1000])}
        estimator = linear_model.LogisticRegression(penalty='l1', fit_intercept=False)
        #param_grid={"alpha": [10, 1, 1e-1, 1e-2], "gamma": [.1, .2, 1]}
        #estimator = KernelRidge(kernel='rbf', gamma=0.1)
        #param_grid = {"alpha": [10, 1, 1e-1, 1e-2],
        #              "hidden_layer_sizes": [(5,), (20,), (100,)]}
        #estimator = MLPClassifier(solver='adam', random_state=1, max_iter=10000)
        num_cores = multiprocessing.cpu_count()
        gcv = GridSearchCV(estimator, param_grid,
                        n_jobs=1, iid=False, refit=True,
                        cv=3)
        gcv.fit(x[tr_inds], y[tr_inds, player_id])
        first_stage_estimators.append(gcv.best_estimator_)
    
    # First stage estimates of entry probabilities for the new features
    sigma_hat_test = np.zeros((tst_inds.shape[0], y.shape[1]))
    for i in range(y.shape[1]):
        sigma_hat_test[:, i] = first_stage_estimators[i].predict_proba(x[tst_inds])[:, 1]
    
    return sigma_hat_test


def second_stage_logistic(x_samples, y_samples, sigma_hat, opts):
    n_players = y_samples.shape[1]
    n_dim = x_samples.shape[1]
    n_samples = x_samples.shape[0]
    beta_est = np.zeros(n_players)
    gamma_est = np.zeros((n_players, n_dim))
    estimator = []
    for player_id in range(n_players):
        sigma_hat_minus_i = np.sum(
            sigma_hat[:, np.arange(n_players) != player_id], axis=1).reshape(-1, 1)
        extended_x = np.concatenate((x_samples, sigma_hat_minus_i), axis=1)

        l1_reg = opts['lambda_coef'] * np.sqrt(np.log(n_dim)/n_samples)
        estimator.append(LogisticWithGradientCorrection(alpha_l1=l1_reg, alpha_l2=0.,
                                 steps=opts['steps'], learning_schedule=opts['lr_schedule'], 
                                 learning_rate=opts['lr'], tol=1e-4, batch_size=opts['bs']))
        estimator[player_id].fit(extended_x, y_samples[:, player_id].reshape(-1, 1))
        #estimator.append(linear_model.LogisticRegression(penalty='l1', C=1./(l1_reg * n_samples), max_iter=opts['steps'], fit_intercept=False))
        #estimator[player_id].fit(extended_x, y_samples[:, player_id].ravel())
        
        beta_est[player_id] = estimator[player_id].coef_.flatten()[-1]
        gamma_est[player_id] = estimator[player_id].coef_.flatten()[:-1]

    return beta_est, gamma_est, estimator

def two_stage_non_orthogonal(x_samples, y_samples, opts):
    n_samples = x_samples.shape[0]
    fold1, fold2 = np.arange(n_samples//2), np.arange(n_samples//2, n_samples)
    sigma_hat = np.zeros((n_samples, 2))
    sigma_hat[fold2, :] = first_stage_sigma(x_samples, y_samples, fold1, fold2)
    sigma_hat[fold1, :] = first_stage_sigma(x_samples, y_samples, fold2, fold1)
    beta_est, gamma_est, _ = second_stage_logistic(x_samples, y_samples, sigma_hat, opts)
    return beta_est, gamma_est

def fold_estimates(x_samples, y_samples, tr_inds, tst_inds, final_inds, opts):
    n_samples = x_samples.shape[0]
    n_players = np.shape(y_samples)[1]
    n_dim = np.shape(x_samples)[1]

    # Cross-fit First stage using only tr_inds and tst_inds for training
    sigma_hat = np.zeros((n_samples, n_players))
    fold1, fold2 = tr_inds, np.concatenate((tst_inds, final_inds), axis=0)
    sigma_hat[fold2, :] = first_stage_sigma(x_samples, y_samples, fold1, fold2)
    fold1, fold2 = tst_inds, np.concatenate((tr_inds, final_inds), axis=0)
    sigma_hat[fold2, :] = first_stage_sigma(x_samples, y_samples, fold1, fold2)
    
    # Second stage estimates from cross-fitted estimates, train using only tr_inds and tst_inds. Predict on final_inds
    train_fold = np.concatenate((tr_inds, tst_inds), axis=0)
    beta_est_prel, gamma_est_prel, estimator_prel = second_stage_logistic(x_samples[train_fold], y_samples[train_fold], sigma_hat[train_fold], opts)
    y_pred_prel = np.zeros((final_inds.shape[0], n_players))
    for player_id in np.arange(n_players):
        sigma_hat_minus_i = np.sum(
            sigma_hat[final_inds][:, np.arange(n_players) != player_id], axis=1).reshape(-1, 1)
        extended_x = np.concatenate((x_samples[final_inds, :], sigma_hat_minus_i), axis=1)
        y_pred_prel[:, player_id] = estimator_prel[player_id].predict_proba(extended_x)[:, 1]

    # Final stage construct quantities on final_inds required for orthogonal loss
    extended_x = []
    grad_corrections = []
    for player_id in np.arange(n_players):
        y_pred_prel_minus_i = np.sum(
            y_pred_prel[:, np.arange(n_players) != player_id], axis=1).reshape(-1, 1)
        y_pred_prel_i = y_pred_prel[:, player_id].reshape(-1, 1)
        y_minus_i = np.sum(y_samples[final_inds][:, np.arange(n_players) != player_id], axis=1).reshape(-1, 1)
        extended_x.append(np.concatenate((x_samples[final_inds, :], y_pred_prel_minus_i), axis=1))
        grad_corrections.append(beta_est_prel[player_id] * y_pred_prel_i * (1 - y_pred_prel_i) * (y_minus_i - y_pred_prel_minus_i))

    return extended_x, grad_corrections


def two_stage_crossfit_orthogonal(x_samples, y_samples, opts):
    n_samples = x_samples.shape[0]
    n_players = np.shape(y_samples)[1]
    n_dim = np.shape(x_samples)[1]
    
    kf = KFold(n_splits=2)
    extended_x = np.zeros((n_players, n_samples, n_dim + 1))
    grad_corrections = np.zeros((n_players, n_samples, 1))
    for train_index, final_inds in kf.split(x_samples):
        tr_inds = train_index[:train_index.shape[0]//2]
        tst_inds = train_index[train_index.shape[0]//2:]
        extended_x_list, grad_corrections_list = fold_estimates(x_samples, y_samples, tr_inds, tst_inds, final_inds, opts)
        print("done with fold estiamtes")
        for player_id in range(n_players):
            extended_x[player_id, final_inds, :] = extended_x_list[player_id]
            grad_corrections[player_id, final_inds, :] = grad_corrections_list[player_id]
    
    # Final stage estimation
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(x_samples.shape[1])/(n_samples))

    beta_est = np.zeros(n_players)
    gamma_est = np.zeros((n_players, n_dim))
    for player_id in range(n_players):
        estimator = LogisticWithGradientCorrection(alpha_l1=l1_reg, alpha_l2=0.,\
                                 steps=opts['steps'], learning_schedule=opts['lr_schedule'], 
                                 learning_rate=opts['lr'], tol=1e-4, batch_size=opts['bs'])
        estimator.fit(extended_x[player_id], y_samples[:, player_id].reshape(-1, 1), grad_corrections=grad_corrections[player_id])
        beta_est[player_id] = estimator.coef_.flatten()[-1]
        gamma_est[player_id] = estimator.coef_.flatten()[:-1]
    
    print("Max correction: {}".format(np.linalg.norm(grad_corrections.flatten(), ord=np.inf)))
    
    # Final stage without correction
    beta_est_no_cor = np.zeros(n_players)
    gamma_est_no_cor = np.zeros((n_players, n_dim))
    for player_id in range(n_players):
        estimator = LogisticWithGradientCorrection(alpha_l1=l1_reg, alpha_l2=0.01*l1_reg,\
                                 steps=opts['steps'], learning_schedule=opts['lr_schedule'], 
                                 learning_rate=opts['lr'], tol=1e-4, batch_size=opts['bs'])
        estimator.fit(extended_x[player_id], y_samples[:, player_id].reshape(-1, 1))
        #estimator = linear_model.LogisticRegression(penalty='l1', C=1./(l1_reg * n_samples), max_iter=opts['steps'])
        #estimator.fit(extended_x[player_id], y_samples[:, player_id].ravel())
        beta_est_no_cor[player_id] = estimator.coef_.flatten()[-1]
        gamma_est_no_cor[player_id] = estimator.coef_.flatten()[:-1]

    return beta_est, gamma_est, beta_est_no_cor, gamma_est_no_cor


def experiment(exp_id, opts):
    np.random.seed(exp_id)
    # Draw a sample of (feature, equilibrium) data points
    x, y, sigma, true_beta, true_gamma = gen_data(
        opts['n_samples'], opts['n_dim'], opts['n_players'], opts['kappa_gamma'],
        sigma_x=opts['sigma_x'], dgp_type=opts['dgp_type'], kappa_gamma_aux=opts['kappa_gamma_aux'])
    s00 = np.sum((y[:, 0]==0) & (y[:, 1]==0))
    s01 = np.sum((y[:, 0]>0) & (y[:, 1]==0))
    s10 = np.sum((y[:, 0]==0) & (y[:, 1]>0))
    s11 = np.sum((y[:, 0]>0) & (y[:, 1]>0))
    print("Statistics: (0, 0): {}, (0, 1): {}, (1, 0): {}, (1, 1): {}".format(s00, s01, s10, s11))
    true_theta = np.concatenate((true_beta.reshape(-1, 1), true_gamma), axis=1)
    print('True coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(true_theta[0, :]) if np.abs(c)>0.01]))
    
    # Simple logistic after first stage
    beta_direct, gamma_direct = two_stage_non_orthogonal(x, y, opts)
    theta_direct = np.concatenate((beta_direct.reshape(-1, 1), gamma_direct), axis=1)
    l1_direct = np.linalg.norm((theta_direct - true_theta)[0, :].flatten(), ord=1)
    l2_direct = np.linalg.norm((theta_direct - true_theta)[0, :].flatten(), ord=2)
    print('Direct coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(theta_direct[0, :]) if np.abs(c)>0.01]))
    
    # Orthogonal lasso estimation
    beta_ortho, gamma_ortho, beta_ortho_no_cor, gamma_ortho_no_cor = two_stage_crossfit_orthogonal(x, y, opts)
    theta_ortho = np.concatenate((beta_ortho.reshape(-1, 1), gamma_ortho), axis=1)
    l1_ortho = np.linalg.norm((theta_ortho - true_theta)[0, :].flatten(), ord=1)
    l2_ortho = np.linalg.norm((theta_ortho - true_theta)[0, :].flatten(), ord=2)
    print('Ortho coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(theta_ortho[0, :]) if np.abs(c)>0.01]))

    theta_ortho_no_cor = np.concatenate((beta_ortho_no_cor.reshape(-1, 1), gamma_ortho_no_cor), axis=1)
    l1_ortho_no_cor = np.linalg.norm((theta_ortho_no_cor - true_theta)[0, :].flatten(), ord=1)
    l2_ortho_no_cor = np.linalg.norm((theta_ortho_no_cor - true_theta)[0, :].flatten(), ord=2)
    print('Ortho_no_cor coef 0:' + ', '.join(["({}: {:.2f})".format(ind, c) for ind, c in enumerate(theta_ortho_no_cor[0, :]) if np.abs(c)>0.01]))


    return l1_direct, l1_ortho, l2_direct, l2_ortho, l1_ortho_no_cor, l2_ortho_no_cor

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
    
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.violinplot([np.array(l2_direct), np.array(l2_ortho), np.array(l2_ortho_no_cor)], showmedians=True)
    plt.xticks([1, 2, 3], ['direct', 'ortho', 'no_cor'])
    plt.ylabel('$\ell_2$ error')
    plt.subplot(1, 2, 2)
    plt.violinplot([np.array(l2_direct) - np.array(l2_ortho), np.array(l2_direct) - np.array(l2_ortho_no_cor)], showmedians=True)
    plt.xticks([1, 2], ['direct vs ortho', 'direct vs no_cor'])
    plt.ylabel('$\ell_2$ error decrease')
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, 'l2_{}.pdf'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
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
    plt.savefig(os.path.join(target_dir, 'l1_{}.pdf'.format('_'.join(['{}_{}'.format(k, v) for k,v in opts.items()]))))
    plt.close()

    return l1_direct, l1_ortho, l2_direct, l2_ortho


if __name__ == '__main__':
    '''
    Entry game: definition of parameters. Utility of each player is of the form:
    u_i = beta_i \sum_{j\neq i} \sigma_j(x) + <gamma_i, x>
    where alpha_i is a constant offset, beta_i is the interaction coefficient, gamma_i
    are the feature related coefficients, x is the feature vector, sigma_j(x) is the
    entry probability of player j
    '''
    opts= {'n_experiments': 10, # number of monte carlo experiments
            'n_samples': 100, # samples used for estimation
            'n_dim': 20, # dimension of the feature space
            'n_players': 2, # number of players
            'sigma_x': 3., # variance of the features
            'kappa_gamma': 2, # support size of the target parameter
            'kappa_gamma_aux': 20, # support size for opponent in stylized dgp
            'dgp_type': 'stylized', # whether to compute real equilibrium data or logistic regression opponent
            'lambda_coef': .5, # coeficient in front of the asymptotic rate for regularization lambda
            'steps': 400, # training steps for gradient based optimizations
            'bs': 10000, # batch size for gradient based optimizations
            'lr': 0.05, # learning rate (base if varying schedule or constant)
            'lr_schedule': 'constant' # 'constant' or 'decay' for linear decay
    }
    reload_results = False
    target_dir = 'results_games'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    main(opts, target_dir=target_dir, reload_results=reload_results)
    