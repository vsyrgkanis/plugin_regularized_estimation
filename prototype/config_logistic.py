import numpy as np
import logistic_te
import metrics

CONFIG = {
        "opts": {'n_experiments': 100, # number of monte carlo experiments
            'n_samples': 5000, # samples used for estimation
            'dim_x': 2000, # dimension of controls x
            'dim_z': 2000, # dimension of variables used for heterogeneity (subset of x)
            'kappa_theta': 2, # support size of target parameter
            'kappa_x': 5, # support of nuisance parameter
            'sigma_eta': 3, # variance of error in secondary moment equation
            'sigma_x': .5, # variance parameter for co-variate distribution
            'lambda_coef': .5, # coeficient in front of the asymptotic rate for regularization lambda
            'n_folds': 2, # number of folds used in cross-fitting
        },
        "methods": {
            "Direct": logistic_te.direct_fit,
            "Ortho": logistic_te.dml_crossfit
        },
        "metrics": {
            "$\\ell_2$ error": metrics.l2_error,
            "$\\ell_1$ error": metrics.l1_error
        },
        "mc_opts": {
            "target_dir": "results_missing",
            "reload_results": True,
            "plot_params": False,
            "random_seed": 123,
            "proposed_method": "Ortho",
            "gen_data": logistic_te.gen_data
        }
    }