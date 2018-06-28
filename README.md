# Introduction

Code associated with paper: <i>Plug-in Regularized Estimation of High-Dimensional Parameters in Nonlinear Semiparametric Models</i>, <b>Chernozhukov, Nekipelov, Semenova, Syrgkanis</b>, 2018

Code assumes Python 3.6, but should also work with Python 2. It also requires basic python packages like, `numpy`, `scipy` and `scikit-learn`.

# File Descriptions

* For replicating the experiments with the linear heterogeneous treatment effect estimation, run: `python linear_growing_kappa.py`
```{r, engine='bash'}
python linear_growing_kappa.py
```

* For replicating the experiments with the logistic heterogeneous treatment effect estimation, run: 
```{r, engine='bash'}
cp config_logistic.py_example config_logistic.py
python mc_from_config.py --config config_logistic
```

* For replicating the experiments related to estimation of conditional moment models with missing data, run: 
```{r, engine='bash', missing data experiments}
cp config_missing_data.py_example config_missing_data.py
python mc_from_config.py --config config_missing_data
```

* For replicating the experiments related to estimation in games of incomplete information, run: 
```{r, engine='bash', games of incomplete information experiments}
cp config_games.py_example config_games.py
python mc_from_config.py --config config_games
```

* The library folder ```mcpy``` contains library code related to running generic monte carlo experiments from config files and saving and running the results

* The library folder ```orthopy``` contains modifications to standard estimation methods, such as the logistic regression, that are required for orthogonal estimation, e.g. adding
and orthogonal correction term to the loss or adding an offset to the index.
