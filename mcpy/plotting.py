import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mcpy.utils import filesafe


def plot_subset_param_histograms(param_estimates, metric_results, config, subset):
    for dgp_name, pdgp in param_estimates.items():
        n_methods = len(list(pdgp.keys()))
        n_params = config['dgp_opts']['kappa_gamma'] + 1
        plt.figure(figsize=(4 * n_params, 2 * n_methods))
        for it, m_name in enumerate(pdgp.keys()):
            for inner_it, i in enumerate(subset):
                plt.subplot(n_methods, n_params, it * n_params + inner_it + 1)
                plt.hist(pdgp[m_name][:, i])
                plt.title("{}[{}]. $\\mu$: {:.2f}, $\\sigma$: {:.2f}".format(m_name, i, np.mean(pdgp[m_name][:, i]), np.std(pdgp[m_name][:, i])))
        plt.tight_layout()
        plt.savefig(os.path.join(config['target_dir'], 'dist_dgp_{}_{}.png'.format(dgp_name, config['param_str'])), dpi=300)
        plt.close()
    return 

def plot_param_histograms(param_estimates, metric_results, config):
    for dgp_name, pdgp in param_estimates.items():
        n_methods = len(list(pdgp.keys()))
        n_params = next(iter(pdgp.values())).shape[1]
        plt.figure(figsize=(4 * n_params, 2 * n_methods))
        for it, m_name in enumerate(pdgp.keys()):
            for i in range(pdgp[m_name].shape[1]):
                plt.subplot(n_methods, n_params, it * n_params + i + 1)
                plt.hist(pdgp[m_name][:, i])
                plt.title("{}[{}]. $\\mu$: {:.2f}, $\\sigma$: {:.2f}".format(m_name, i, np.mean(pdgp[m_name][:, i]), np.std(pdgp[m_name][:, i])))
        plt.tight_layout()
        plt.savefig(os.path.join(config['target_dir'], 'dist_dgp_{}_{}.png'.format(dgp_name, config['param_str'])), dpi=300)
        plt.close()
    return 

def plot_metrics(param_estimates, metric_results, config):
    for dgp_name, mdgp in metric_results.items():
        n_methods = len(list(mdgp.keys()))
        for metric_name in next(iter(mdgp.values())).keys():
            plt.figure(figsize=(1.5 * n_methods, 2.5))
            plt.violinplot([mdgp[method_name][metric_name] for method_name in mdgp.keys()], showmedians=True)
            plt.xticks(np.arange(1, n_methods + 1), list(mdgp.keys()))
            plt.ylabel(metric_name)
            plt.tight_layout()
            plt.savefig(os.path.join(config['target_dir'], '{}_dgp_{}_{}.png'.format(filesafe(metric_name), dgp_name, config['param_str'])), dpi=300)
            plt.close()
    return

def plot_metric_comparisons(param_estimates, metric_results, config):
    for dgp_name, mdgp in metric_results.items():
        n_methods = len(list(mdgp.keys()))
        for metric_name in next(iter(mdgp.values())).keys():
            plt.figure(figsize=(1.5 * n_methods, 2.5))
            plt.violinplot([mdgp[method_name][metric_name] - mdgp[config['proposed_method']][metric_name] for method_name in mdgp.keys() if method_name != config['proposed_method']], showmedians=True)
            plt.xticks(np.arange(1, n_methods), [method_name for method_name in mdgp.keys() if method_name != config['proposed_method']])
            plt.ylabel('Decrease in {}'.format(metric_name))
            plt.tight_layout()
            plt.savefig(os.path.join(config['target_dir'], '{}_decrease_dgp_{}_{}.png'.format(filesafe(metric_name), dgp_name, config['param_str'])), dpi=300)
            plt.close()
    return