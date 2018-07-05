# Introduction

Code associated with paper: <i>Plug-in Regularized Estimation of High-Dimensional Parameters in Nonlinear Semiparametric Models</i>, <b>Chernozhukov, Nekipelov, Semenova, Syrgkanis</b>, 2018

Code assumes Python 3.6, but should also work with Python 2. It also requires basic python packages like, `numpy`, `scipy` and `scikit-learn`.

# File Descriptions

## Linear Heterogeneous Treatment Effects
For replicating the experiments with the linear heterogeneous treatment effect estimation, run:
```{r, engine='bash'}
cp sweep_config_linear.py_example sweep_config_linear.py
python sweep_mc_from_config.py --config sweep_config_linear
```
The DGP and estimation methods for this application are contained in `linear_te.py`.

## Heterogeneous Treatment Effects with a Logistic Link
For replicating the experiments with the logistic heterogeneous treatment effect estimation, run: 
```{r, engine='bash'}
cp config_logistic.py_example config_logistic.py
python mc_from_config.py --config config_logistic
```
The DGP and estimation methods for this application are contained in `logistic_te.py`.

## Conditional Moment Models with Missing Data
For replicating the experiments related to estimation of conditional moment models with missing data, run: 
```{r, engine='bash', missing data experiments}
cp config_missing_data.py_example config_missing_data.py
python mc_from_config.py --config config_missing_data
```
The DGP and estimation methods for this application are contained in `missing_data.py`.

## Games of Incomplete Information
For replicating the experiments related to estimation in games of incomplete information, run: 
```{r, engine='bash', games of incomplete information experiments}
cp config_games.py_example config_games.py
python mc_from_config.py --config config_games
```
The DGP and estimation methods for this application are contained in `games.py`.

## MCPY library
The library folder ```mcpy``` contains library code related to running generic monte carlo experiments from config files and saving and running the results. 
Check out the notebook ```example_mcpy.ipynb``` for a simple example of how to use the library.

A simple config dictionary allows you to run monte carlo experiments for some configuration of the parameters of the dgp and the estimation methods and allows you to specify arbitrary methods to use to estimate for each 
sample, arbitrary set of dgps to use to generate samples, arbitrary metrics to evaluate, and arbitrary plots to create from the experimental results. The monte carlo class will 
run many experiments, each time generating a sample from each dgp, running each estimation method on each sample and calculating each metric on the returned result. Subsequently the
plotting functions receive the collection of all experimental results and create figures. The package offers a basic set of plotting functions but the user can define their own
plotting functions and add them to their config dictionary. See e.g. `config_games_py_example`, `config_logistic.py_example` and `config_missing_data.py_example` for sample config
variable definitions.

A sweep config dictionary, allows you to specify for each dgp option a whole list of parameters. Then the MonteCarloSweep class will execute monte carlo experiments for each 
combination of parameters. Subsequently the plotting functions provided can for instance plot how each metric varies as a single parameter varies and averaging out the performance
over the settings of the rest of the parameters. Such plots are created for each dgp and metric, and each plot contains the results for each method. This is for instance used
in the case of the linear treatment effect experiment. See e.g. `sweep_config_linear.py_example` for a sample sweep-config variable definition.

## ORTHOPY library
The library folder ```orthopy``` contains modifications to standard estimation methods, such as the logistic regression, that are required for orthogonal estimation, e.g. adding
and orthogonal correction term to the loss or adding an offset to the index.
