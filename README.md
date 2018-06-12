# Introduction

Code associated with paper: <i>Plug-in Regularized Estimation of High-Dimensional Parameters in Nonlinear Semiparametric Models</i>, <b>Chernozhukov, Nekipelov, Semenova, Syrgkanis</b>, 2018

# File Descriptions

* For replicating the experiments with the linear heterogeneous treatment effect estimation, run the python script: `linear_te.py`

* For replicating the experiments with the logistic heterogeneous treatment effect estimation, run the python script: `logistic_te.py`

* The jupyter notebooks `linear_te.ipynb` and `logistic_te.ipynb` contain example calls to library functions contained in the scripts above.

* The file `logistic_with_offset.py` contains a class that corresponds to a tensorflow based implementation of the weighted logistic regression with index offsets and `l1` and `l2` regularization, required in the final stage of the orthogonal estimation of treatment effect models with a logistic link.
