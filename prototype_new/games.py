'''
@author: Vasilis Syrgkanis
'''
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model, clone
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from orthopy.scipy_logistic_with_gradient_correction import LogisticWithOffsetAndGradientCorrection
import scipy


###############################
# DGPs
###############################

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
            sigma_n[i] = scipy.special.expit(beta[i] * sigma_minus_i + np.dot(gamma[i], x))

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

def equilibrium_data(opts):
    '''
    Create equilibrium data with given parameters, i.e. pairs of features
    and probabilities of entry
    '''
    n_samples = opts['n_samples']
    n_dim = opts['n_dim']
    kappa_gamma = opts['kappa_gamma']
    sigma_x = opts['sigma_x']
    kappa_gamma_aux = opts['kappa_gamma_aux']
    n_players = 2

    # Market interaction coefficients
    beta = np.array([-2.0, -3.0])
    # Feature related coefficients for each player
    gamma = np.zeros((n_players, n_dim))
    gamma[0, :kappa_gamma] = 1.0
    gamma[1, :kappa_gamma_aux] = np.random.uniform(-2, 2, size=kappa_gamma_aux)

    ### Generate sample from instance
    x_samples = np.random.uniform(-sigma_x, sigma_x, (n_samples, n_dim))
    # Matrix of entry probability for each feature vector
    sigma = np.zeros((n_samples, n_players))
    # Matrix of entry decisions for each feature vector
    y = np.zeros((n_samples, n_players))

    # Compute equilibrium probabilities for each feature vector
    for it, x in enumerate(x_samples):
        sigma[it, :] = compute_equilibrium(n_players, x, beta, gamma)
    
    # Draw sample of entry decisions from probabilities of entry
    y_samples = np.random.binomial(1, sigma[:, 0])
    y_samples_op = np.random.binomial(1, sigma[:, 1])

    return (x_samples, y_samples, y_samples_op, sigma), np.concatenate((gamma[0, :], [beta[0]]))

def stylized_data(opts):
    '''
    Compute stylized dgp where opponents strategy is really a logistic regression
    '''
    n_samples = opts['n_samples']
    n_dim = opts['n_dim']
    kappa_gamma = opts['kappa_gamma']
    sigma_x = opts['sigma_x']
    kappa_gamma_aux = opts['kappa_gamma_aux']
    n_players = 2

    ### Generate instance
    beta = -2.0
    gamma = np.zeros((n_players, n_dim))
    gamma[0, :kappa_gamma] = 1.0
    gamma[1, :kappa_gamma_aux] = np.random.uniform(-2, 2, size=kappa_gamma_aux)

    ### Generate sample from instance
    x_samples = np.random.uniform(-sigma_x, sigma_x, (n_samples, n_dim))
    # Matrix of entry probability for each feature vector
    sigma = np.zeros((n_samples, n_players))
    # Matrix of entry decisions for each feature vector
    sigma[:, 1] = scipy.special.expit(np.matmul(x_samples, gamma[1, :]))
    sigma[:, 0] = scipy.special.expit(np.matmul(x_samples, gamma[0, :]) + beta * sigma[:, 1])
    # Draw sample of entry decisions from probabilities of entry
    y_samples = np.random.binomial(1, sigma[:, 0])
    y_samples_op = np.random.binomial(1, sigma[:, 1])

    return (x_samples, y_samples, y_samples_op, sigma), np.concatenate((gamma[0, :], [beta]))
    

###############################
# Estimation Utils
###############################

def first_stage_sigma(X, y_op, opts):
    '''
    Train first stage estimates of opponent entry with Logistic Lasso
    '''
    est = linear_model.LogisticRegressionCV()
    est.fit(X, y_op.ravel())
    return est


def second_stage_logistic(X, y, sigma_hat_op, opts):
    n_samples, n_dim = X.shape
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(n_dim + 1)/n_samples)
    estimator = LogisticWithOffsetAndGradientCorrection(alpha_l1=l1_reg, alpha_l2=0., tol=1e-6)
    estimator.fit(np.concatenate((X, sigma_hat_op), axis=1), y)
    return estimator


###############################
# Estimation Methods
###############################

def two_stage_oracle(data, opts):
    X, y, _, sigma = data
    est = second_stage_logistic(X, y, sigma[:, [1]], opts)
    return est.coef_.flatten()

def two_stage_no_split(data, opts):
    X, y, y_op, _ = data
    est = first_stage_sigma(X, y_op, opts)
    sigma_hat_op = est.predict_proba(X)[:, [1]]
    final_est = second_stage_logistic(X, y, sigma_hat_op, opts)
    return final_est.coef_.flatten()

def two_stage_non_orthogonal(data, opts):
    X, y, y_op, _ = data
    n_samples = X.shape[0]
    sigma_hat_op = np.zeros((n_samples, 1))
    for train, test in KFold(n_splits=opts['n_splits']).split(X):
        est = first_stage_sigma(X[train], y_op[train], opts)
        sigma_hat_op[test, :] = est.predict_proba(X[test])[:, [1]]
    final_est = second_stage_logistic(X, y, sigma_hat_op, opts)
    return final_est.coef_.flatten()

def two_stage_crossfit_orthogonal(data, opts):
    X, y, y_op, _ = data
    n_samples, n_dim = X.shape
    sigma_hat_op = np.zeros((n_samples, 1))
    grad_corrections = np.zeros((n_samples, 1))
    for train, test in KFold(n_splits=opts['n_splits']).split(X):
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
    l1_reg = opts['lambda_coef'] * np.sqrt(np.log(n_dim + 1)/(n_samples))
    estimator = LogisticWithOffsetAndGradientCorrection(alpha_l1=l1_reg, alpha_l2=0., tol=1e-6)
    estimator.fit(np.concatenate((X, sigma_hat_op), axis=1), y, grad_corrections=grad_corrections)
    print("Max correction: {}".format(np.linalg.norm(grad_corrections.flatten(), ord=np.inf)))

    return estimator.coef_.flatten()

