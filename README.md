# Introduction

Code associated with paper: <i>Regularized Orthogonal Estimation of High-Dimensional Parameters in Nonlinear Semiparametric Models</i>, <b> Nekipelov, Semenova, Syrgkanis</b>, 2018

Code requires Python 3.6. It also requires basic python packages like, `numpy`, `scipy` and `scikit-learn`.

# Running the Monte Carlo Simulations

## Linear Heterogeneous Treatment Effects
For replicating the experiments with the linear heterogeneous treatment effect estimation, run:
```{r, engine='bash'}
cp sweep_config_linear.py_example sweep_config_linear.py
python sweep_mc_from_config.py --config sweep_config_linear
```
The DGP and estimation methods for this application are contained in `linear_te.py`. 

The above code produces the following figures:
<p align="center">
  <img src="figs/linear_te/ell1_error_dgp_dgp1_growing_kappa.png" height="100" title="ell1_error_dgp_dgp1_growing_kappa">
  <img src="figs/linear_te/ell2_error_dgp_dgp1_growing_kappa.png" height="100" alt="ell2_error_dgp_dgp1_growing_kappa">
  <br/>
  <i>Growing support size of nuisance</i>
</p>
<p align="center">
 <img src="figs/linear_te/plot2_ell2_varying_sigma_eta.png" height="100" alt="plot2_ell2_varying_sigma_eta">
  <img src="figs/linear_te/plot3_ell2_varying_sigma_epsilon.png" height="100" title="plot3_ell2_varying_sigma_epsilon">
  <br/>
  <i>Varying support size and variances of errors</i>
</p>

## Heterogeneous Treatment Effects with a Logistic Link
For replicating the experiments with the logistic heterogeneous treatment effect estimation, run: 
```{r, engine='bash'}
cp config_logistic.py_example config_logistic.py
python mc_from_config.py --config config_logistic
```
The DGP and estimation methods for this application are contained in `logistic_te.py`.

The above code produces the following figures:
<p align="center">
  <img src="figs/logistic_te/l2_error_logistic.png" height="100" title="l2_error_logistic">
  <img src="figs/logistic_te/l2_decrease_logistic.png" height="100" alt="l1_decrease_logistic">
  <img src="figs/logistic_te/l1_error_logistic.png" height="100" alt="l1_error_logistic">
  <img src="figs/logistic_te/l1_decrease_logistic.png" height="100" title="l1_decrease_logistic">
  <br/>
  <i>l1 and l2 errors and error decreases (Direct - Ortho)</i>
</p>


## Conditional Moment Models with Missing Data
For replicating the experiments related to estimation of conditional moment models with missing data, run: 
```{r, engine='bash', missing data experiments}
cp config_missing_data.py_example config_missing_data.py
python mc_from_config.py --config config_missing_data

cp sweep_config_missing_data.py_example sweep_config_missing_data.py
python sweep_mc_from_config.py --config sweep_config_missing_data
```
The DGP and estimation methods for this application are contained in `missing_data.py`.

The above code produces the following figures:
<p align="center">
    <img src="figs/missing_data/l2_errors_missing.png" height="100" alt="l2_errors_missing">
  <img src="figs/missing_data/l2_decrease_missing.png" height="100" title="l2_decrease_missing">
  <br/>
  <i>l2 errors and error decreases for each benchmark method</i>
</p>
<p align="center">
  <img src="figs/missing_data/ipmr_kz_1_sx_3_dx_se.png" height="100" title="ipmr_kz_1_sx_3_dx_se">
  <img src="figs/missing_data/dmr_kz_1_sx_3_dx_se.png" height="100" alt="dmr_kz_1_sx_3_dx_se">
  <img src="figs/missing_data/ipmr_kz_1_se_1_dx_sx.png" height="100" alt="ipmr_kz_1_se_1_dx_sx">
  <img src="figs/missing_data/dmr_kz_1_se_1_dx_sx.png" height="100" title="dmr_kz_1_se_1_dx_sx">
  <br/>
  <i>Percentage decrease of error with varying dimension of nuisance and error variances</i>
</p>

## Games of Incomplete Information
For replicating the experiments related to estimation in games of incomplete information, run: 
```{r, engine='bash', games of incomplete information experiments}
cp config_games.py_example config_games.py
python mc_from_config.py --config config_games

cp sweep_config_games.py_example sweep_config_games.py
python sweep_mc_from_config.py --config sweep_config_games
```
The DGP and estimation methods for this application are contained in `games.py`.

The above code produces the following figures:
<p align="center">
  <img src="figs/games/l2_error_one_player.png" height="100" title="l2_error_one_player">
  <img src="figs/games/l2_decrease_one_player.png" height="100" alt="l2_decrease_one_player">
  <img src="figs/games/l1_error_one_player.png" height="100" alt="l1_error_one_player">
  <img src="figs/games/l1_decrease_one_player.png" height="100" title="l1_decrease_one_player">
  <br/>
  <i>One player DGP</i>
</p>
<p align="center">
  <img src="figs/games/l2_error_two_player.png" height="100" title="l2_error_two_player">
  <img src="figs/games/l2_decrease_two_player.png" height="100" alt="l2_decrease_two_player">
  <img src="figs/games/l1_error_two_player.png" height="100" alt="l1_error_two_player">
  <img src="figs/games/l1_decrease_two_player.png" height="100" title="l1_decrease_two_player">
  <br/>
  <i>Two player DGP</i>
</p>
<p align="center">
  <img src="figs/games/l2_games_varying_p_sigma_n_5000.png" height="100" title="l2_games_varying_p_sigma_n_5000">
  <img src="figs/games/l2_games_varying_p_sigma_n_10000.png" height="100" alt="l2_games_varying_p_sigma_n_10000">
  <br/>
  <i>One player DGP, varying dimension and variance of co-variates</i>
</p>

# MCPY library
The library folder ```mcpy``` contains library code related to running generic monte carlo experiments from config files and saving and running the results. 
Check out the notebook ```example_mcpy.ipynb``` for a simple example of how to use the library.

A simple config dictionary allows you to run monte carlo experiments for some configuration of the parameters of the dgp and the estimation methods and allows you to specify arbitrary methods to use to estimate for each sample, arbitrary set of dgps to use to generate samples, arbitrary metrics to evaluate, and arbitrary plots to create from the experimental results. The monte carlo class will run many experiments, each time generating a sample from each dgp, running each estimation method on each sample and calculating each metric on the returned result. Subsequently the plotting functions receive the collection of all experimental results and create figures. The package offers a basic set of plotting functions but the user can define their own plotting functions and add them to their config dictionary. See e.g. `config_games_py_example`, `config_logistic.py_example` and `config_missing_data.py_example` for sample config variable definitions.

Consider the following simple example: suppose we want to generate data from a linear model and test the performance of OLS and Lasso for estimating the coefficients and as the dimension of the features changes. We can then create a dgp function that takes as input a dictionary of options and returns a data-set and the true parameter:
```python
def dgp(dgp_opts):
    true_param = np.zeros(opts['n_dim'])
    true_param[:opts['kappa']] = 1.0
    x = np.random.normal(0, 1, size=(opts['n_samples'], opts['n_dim']))
    y = np.matmul(x, true_param)
    return (x, y), true_param
```
Then we also define two functions that take as input a data-set and a dictionary of options and return the estimated coefficient based on each method:
```python
def ols(data, method_opts):
    x, y = data
    from sklearn.linear_model import LinearRegression
    return LinearRegression().fit(x, y).coef_

def lasso(data, method_opts):
    x, y = data
    from sklearn.linear_model import Lasso
    return Lasso(alpha=opts['l1_reg']).fit(x, y).coef_
```
Now we are ready to write our config file that will specify the monte carlo simulation we want to run as well as the metrics of performance to compute and the plots to generate at the end:
```python
from mcpy import metrics
from mcpy import plotting
CONFIG = {
    # Functions to be used for data generation
    'dgps': {'linear_dgp': dgp},
    # Dictionary of parameters to the dgp functions
    'dgp_opts': {'n_dim': 10, 'n_samples': 100, 'kappa': 2},
    # Estimation methods to evaluate
    'methods': {'ols': ols, 'lasso': lasso},
    # Dictionary of parameters to the estimation functions
    'method_opts': {'l1_reg': 0.01},
    # Metrics to evaluate. Each metric takes two inputs: estimated param, true param
    'metrics': {'l1_error': metrics.l1_error, 'l2_error': metrics.l2_error},
    # Options for the monte carlo simulation
    'mc_opts': {'n_experiments': 10, 'seed': 123},
    # Which of the methods is the proposed one vs a benchmark method. Used in plotting
    'proposed_method': 'lasso',
    # Target folder for saving plots and results
    'target_dir': 'test_ols',
    # Whether to reload monte carlo results from the folder if results with
    # the same config spec exist from a previous run
    'reload_results': False,
    # Which plots to generate. Could either be a dictionary of a plot spec or an ad hoc plot function. 
    # A plot spec contains: {'metrics', 'methods', 'dgps', 'metric_transforms'}, 
    # that specify a subset of each to plot (defaults to all if not specified).
    # An ad hoc function should be taking as input (param_estimates, metric_results, config)
    'plots': {'all_metrics': {}, 'param_hist': plotting.plot_param_histograms}
}
```
We can then run this monte-carlo simulation as easy as:
```python
from mcpy.monte_carlo import MonteCarlo
estimates, metric_results = MonteCarlo(CONFIG).run()
```
This code will save the plots in the target_dir. In particular it will save the following two figures that depict the distribution of l1 and l2 errors across the 10 experiments:

<p align="center">
  <img src="figs/test_ols/test_ols_l1_error.png" height="200" title="test_ols_l1_error">
  <img src="figs/test_ols/test_ols_l2_error.png" height="200" title="test_ols_l2_error">
</p>

A sweep config dictionary, allows you to specify for each dgp option a whole list of parameters, rather than a single value. Then the MonteCarloSweep class will execute monte carlo experiments for each combination of parameters. Subsequently the plotting functions provided can for instance plot how each metric varies as a single parameter varies and averaging out the performance over the settings of the rest of the parameters. Such plots are created for each dgp and metric, and each plot contains the results for each method. This is for instance used in the case of the linear treatment effect experiment. See e.g. `sweep_config_linear.py_example` for a sample sweep-config variable definition.

For instance, back to the linear example, we could try to understand how the l1 and l2 errors change as a function of the dimension of the features or the number of samples. We can perform such a sweeping monte carlo by defining a sweep config:
```python
SWEEP_CONFIG = {
    'dgps': {'linear_dgp': dgp},
    'dgp_opts': {'n_dim': [10, 100, 200], 'n_samples': [100, 200, 300], 'kappa': 2},
    'methods': {'ols': ols, 'lasso': lasso},
    'method_opts': {'l1_reg': 0.01},
    'metrics': {'l1_error': metrics.l1_error, 'l2_error': metrics.l2_error},
    'mc_opts': {'n_experiments': 10, 'seed': 123},
    'proposed_method': 'lasso',
    'target_dir': 'test_sweep_ols',
    'reload_results': False,
    # Let's not plot anything per instance
    'plots': {},
    # Let's make some plots across the sweep of parameters
    'sweep_plots': {
        # Let's plot the errors as a function of the dimensions, holding fixed the samples to 100
        'var_dim_at_100_samples': {'varying_params': ['n_dim'], 'select_vals': {'n_samples': [100]}},
        # Let's plot the errors as a function of n_samples, holding fixed the dimensions to 100
        'var_samples_at_100_dim': {'varying_params': ['n_samples'], 'select_vals': {'n_dim': [100]}},
        # Let's plot a 2d contour of the median metric of each method as two parameters vary simultaneously
        'var_samples_and_dim': {'varying_params': [('n_samples', 'n_dim')]},
        # Let's plot the difference between each method in a designated list with the 'proposed_method' in the config
        'error_diff_var_samples_and_dim': {'varying_params': [('n_samples', 'n_dim')], 'methods': ['ols'], 
                                           'metric_transforms': {'diff': metrics.transform_diff}}
    }
}
```
We can then run our sweeping monte carlo with the following command:
```python
from mcpy.monte_carlo import MonteCarloSweep
sweep_keys, sweep_estimates, sweep_metric_results = MonteCarloSweep(SWEEP_CONFIG).run()
```
The sweep plots allows you to define which types of plots to save as some subset of parameters vary while others take a subset of the values.
For instance, the above four sweep plots will create 8 plots, one for each metric. The four plots corresponding to the l2 error are as follows:
<p align="center">
  <img src="figs/test_ols/var_dim_at_100_samples.png" height="100" title="var_dim_at_100_samples">
  <img src="figs/test_ols/var_samples_at_100_dim.png" height="100" alt="var_samples_at_100_dim">
  <img src="figs/test_ols/var_samples_and_dim.png" height="100" alt="var_samples_and_dim">
  <img src="figs/test_ols/error_diff.png" height="100" title="error_diff">
</p>
Showing how lasso out-performs ols when the number of samples is smaller than the dimension.


# ORTHOPY library
The library folder ```orthopy``` contains modifications to standard estimation methods, such as the logistic regression, that are required for orthogonal estimation, e.g. adding
and orthogonal correction term to the loss or adding an offset to the index.

More concretely, the class ```LogisticWithOffsetAndGradientCorrection``` is an estimator adhering to the fit and predict specification of sklearn that enables fitting an "orthogonal" logistic regression. Its fit method minimizes a regularized modified weighted logistic loss, with sample weights, a gradient correction and an index offset:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=L_S(\theta)&space;=&space;\frac{1}{n}\sum_{i=1}^n&space;\ell(y_i,&space;x_i,&space;w_i,&space;v_i,&space;g_i;&space;\theta)&space;&plus;&space;\alpha_1&space;\|\theta\|_1&space;&plus;&space;\frac{1}{2}\alpha_2&space;\|\theta\|_2^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L_S(\theta)&space;=&space;\frac{1}{n}\sum_{i=1}^n&space;\ell(y_i,&space;x_i,&space;w_i,&space;v_i,&space;g_i;&space;\theta)&space;&plus;&space;\alpha_1&space;\|\theta\|_1&space;&plus;&space;\frac{1}{2}\alpha_2&space;\|\theta\|_2^2" title="L_S(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i, w_i, v_i, g_i; \theta) + \alpha_1 \|\theta\|_1 + \frac{1}{2}\alpha_2 \|\theta\|_2^2" /></a><p>
where the modified logistic loss for each sample is defined as:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\ell(y,&space;x,&space;w,&space;v,&space;g;&space;\theta)&space;=&space;w&space;\left(&space;y&space;\log(\mathcal{L}(x'\theta&space;&plus;&space;v))&space;&plus;&space;(1-y)&space;\log(1-&space;\mathcal{L}(x'\theta&plus;v))&space;&plus;&space;g\,&space;x'\theta&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ell(y,&space;x,&space;w,&space;v,&space;g;&space;\theta)&space;=&space;w&space;\left(&space;y&space;\log(\mathcal{L}(x'\theta&space;&plus;&space;v))&space;&plus;&space;(1-y)&space;\log(1-&space;\mathcal{L}(x'\theta&plus;v))&space;&plus;&space;g\,&space;x'\theta&space;\right&space;)" title="\ell(y, x, w, v, g; \theta) = w \left( y \log(\mathcal{L}(x'\theta + v)) + (1-y) \log(1- \mathcal{L}(x'\theta+v)) + g\, x'\theta \right )" /></a>
</p>

The specification of the class follows a standard sklearn paradigm:
```python
class LogisticWithOffsetAndGradientCorrection():
    def __init__(self, alpha_l1=0.1, alpha_l2=0.1, tol=1e-6):
    ''' Initialize
    alpha_l1 : ell_1 regularization weight
    alpha_l2 : ell_2 regularization weight
    tol : minimization tolerance
    '''

    def fit(self, X, y, offsets=None, grad_corrections=None, sample_weights=None):
    ''' Fits coefficient theta by minimizing regularized modified logistic loss
    Parameters
    X (n x d) matrix of features
    y (n x 1) matrix of labels
    offsets (n x 1) matrix of index offsets v
    grad_corrections (n x 1) matrix of grad corrections g
    sample_weights (n x 1) matrix of sample weights w
    '''

    @property
    def coef_(self):
    ''' Fitted coefficient '''
    
    def predict_proba(self, X, offsets=None):
    ''' Probabilistic prediction. Returns (n x 2) matrix of probabilities '''
    
    def predict(self, X, offsets=None):
    ''' Binary prediction '''

    def score(self, X, y_true, offsets=None):
    ''' AUC score '''

    def accuracy(self, X, y_true, offsets=None):
    ''' Prediction accuracy '''
```
### Empirical application
