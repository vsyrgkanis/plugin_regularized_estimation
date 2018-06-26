import numpy as np
import missing_data
import metrics

CONFIG = {
        "opts": {'n_experiments': 1000, # number of monte carlo experiments
            'n_samples': 5000, # samples used for estimation
            'dim_x': 20, # dimension of controls x
            'dim_z': 20, # dimension of variables used for heterogeneity (subset of x)
            'kappa_theta': 1, # support size of target parameter
            'kappa_z': 1, # support size of nuisance
            'sigma_epsilon': 1., # variance of error in secondary moment equation
            'sigma_eta': .1, # variance of error in secondary moment equation, i.e. multiplier in logistic index
            'sigma_x': 3, # variance parameter for co-variate distribution
            'lambda_coef': 1.0, # coeficient in front of the asymptotic rate for regularization lambda
            'n_folds': 3, # number of folds used in cross-fitting
        },
        "methods": {
            "Direct": missing_data.direct_fit,
            "IPS oracle": missing_data.non_ortho_oracle, 
            "Ortho oracle": missing_data.ortho_oracle,
            "IPS": missing_data.non_ortho,
            "Ortho": missing_data.ortho
        },
        "metrics": {
            "$\\ell_2$ error": metrics.l2_error,
            "$\\ell_1$ error": metrics.l1_error
        },
        "mc_opts": {
            "target_dir": "results_missing",
            "reload_results": False,
            "plot_params": True,
            "random_seed": 123,
            "proposed_method": "Ortho",
            "gen_data": missing_data.gen_data
        }
    }