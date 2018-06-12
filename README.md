Code associated with paper: "Plug-in Regularized Estimation of High-Dimensional Parameters in Nonlinear Semiparametric Models", Chernozhukov, Nekipelov, Semenova, Syrgkanis, 2018

For replicating the experiments with the linear heterogeneous treatment effect estimation, run the python script: linear_te.py

For replicating the experiments with the logistic heterogeneous treatment effect estimation, run the python script: logistic_te.py

The jupyter notebooks contain example calls to library functions contained in the scripts above.

The file logistic\_with\_offset.py contains a class that corresponds to a tensorflow based implementation of the weighted logistic regression with index offsets and $\ell_1$ and $\ell_2$ regularization, required in the final stage of the orthogonal estimation of treatment effect models with a logistic link.