import os
import numpy as np
import linear_te
import metrics
import plotting
from monte_carlo import MonteCarlo
import joblib
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

BASE_CONFIG = {
        "dgps": {
            "dgp1": linear_te.gen_data
        },
        "dgp_opts": {
            'n_samples': 5000, # samples used for estimation
            'dim_x': 20, # dimension of controls x
            'dim_z': 20, # dimension of variables used for heterogeneity (subset of x)
            'kappa_theta': 2, # support size of target parameter
            'kappa_x': 1, # support size of nuisance
            'sigma_eta': 1.0, # variance of error in secondary moment equation
            'sigma_epsilon': 1.0 # variance of error in primary moment equation
        },
        "methods": {
            "Direct": linear_te.direct_fit,
            "Ortho": linear_te.dml_fit,
            "CrossOrtho": linear_te.dml_crossfit
        },
        "method_opts": {
            'lambda_coef': 1.0, # coeficient in front of the asymptotic rate for regularization lambda
            'n_folds': 3 # number of folds in crossfitting
        },
        "metrics": {
            "$\\ell_2$ error": metrics.l2_error,
            "$\\ell_1$ error": metrics.l1_error
        },
        "plots": {
            "metrics": plotting.plot_metrics,
            "metric_comparisons": plotting.plot_metric_comparisons
        },
        "mc_opts": {
            'n_experiments': 100, # number of monte carlo experiments
            "seed": 123
        },
        "proposed_method": "CrossOrtho",
        "target_dir": "results_linear_growing",
        "reload_results": False
    }

if __name__=="__main__":
    kappa_grid = np.arange(1, 18, 3)
    
    if BASE_CONFIG['reload_results'] and os.path.exists(os.path.join(BASE_CONFIG['target_dir'], 'results_growing_kappa')):
        metric_results_growing_kappa = joblib.load(os.path.exists(os.path.join(BASE_CONFIG['target_dir'], 'results_growing_kappa')))
    else:
        metric_results_growing_kappa = []
        for kappa_x in kappa_grid:
            BASE_CONFIG['dgp_opts']['kappa_x'] = kappa_x
            _, metric_results = MonteCarlo(BASE_CONFIG).run()
            metric_results_growing_kappa.append(metric_results)
        joblib.dump(metric_results_growing_kappa, os.path.join(BASE_CONFIG['target_dir'], 'results_growing_kappa'))

    for dgp in BASE_CONFIG['dgps'].keys():
        for metric in BASE_CONFIG['metrics'].keys():
            plt.figure(figsize=(5, 3))
            for method in BASE_CONFIG['methods'].keys():
                medians = [np.median(mres[dgp][method][metric]) for mres in metric_results_growing_kappa]
                mins = [np.min(mres[dgp][method][metric]) for mres in metric_results_growing_kappa]
                maxs = [np.max(mres[dgp][method][metric]) for mres in metric_results_growing_kappa]
                plt.plot(kappa_grid, medians, label=method)
                plt.fill_between(kappa_grid, maxs, mins, alpha=0.3)
            plt.legend()
            plt.xlabel('support size $k_g$')
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_CONFIG['target_dir'], '{}_dgp_{}_growing_kappa.png'.format(utils.filesafe(metric), dgp)), dpi=300)
            plt.close()