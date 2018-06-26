import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import joblib
import argparse
import importlib
plt.style.use('ggplot')

def filesafe(str):
    return "".join([c for c in str if c.isalpha() or c.isdigit() or c==' ']).rstrip()


class MonteCarlo:

    def __init__(self, config):
        self.opts = config['opts']
        self.mc_opts = config['mc_opts']
        self.methods = config['methods']
        self.metrics = config['metrics']     
        self.param_str = '_'.join(['{}_{}'.format(k, v) for k,v in self.opts.items()])
        return

    def experiment(self, exp_id):
        np.random.seed(exp_id)
        data, true_param = self.mc_opts['gen_data'](self.opts)
        params = {}
        metric_results = {}
        for method_name, method in self.methods.items():
            params[method_name] = method(data, self.opts)
            metric_results[method_name] = {}
            for metric_name, metric in self.metrics.items():
                metric_results[method_name][metric_name] = metric(params[method_name], true_param)

        return params, metric_results

    def plot_param_histograms(self, params):
        n_methods = len(self.methods)
        n_params = next(iter(params.values())).shape[1]
        plt.figure(figsize=(4 * n_params, 2 * n_methods))
        for it, m_name in enumerate(self.methods.keys()):
            for i in range(params[m_name].shape[1]):
                plt.subplot(n_methods, n_params, it * n_params + i + 1)
                plt.hist(params[m_name][:, i])
                plt.title("{}[{}]. $\mu$: {:.2f}, $\sigma$: {:.2f}".format(m_name, i, np.mean(params[m_name][:, i]), np.std(params[m_name][:, i])))
        plt.tight_layout()
        plt.savefig(os.path.join(self.mc_opts['target_dir'], 'dist_{}.png'.format(self.param_str)), dpi=300)
        plt.close()

    def plot_metrics(self, metrics):
        n_methods = len(self.methods)
        for metric_name in self.metrics.keys():
            plt.figure(figsize=(1.5 * n_methods, 2.5))
            plt.violinplot([metrics[method_name][metric_name] for method_name in self.methods.keys()], showmedians=True)
            plt.xticks(np.arange(1, n_methods + 1), list(self.methods.keys()))
            plt.ylabel(metric_name)
            plt.tight_layout()
            plt.savefig(os.path.join(self.mc_opts['target_dir'], '{}_{}.png'.format(filesafe(metric_name), self.param_str)), dpi=300)
            plt.close()

    def plot_metric_comparisons(self, metrics):
        n_methods = len(self.methods)
        for metric_name in self.metrics.keys():
            plt.figure(figsize=(1.5 * n_methods, 2.5))
            plt.violinplot([metrics[method_name][metric_name] - metrics[self.mc_opts['proposed_method']][metric_name] for method_name in self.methods.keys() if method_name != self.mc_opts['proposed_method']], showmedians=True)
            plt.xticks(np.arange(1, n_methods), [method_name for method_name in self.methods.keys() if method_name != self.mc_opts['proposed_method']])
            plt.ylabel('Decrease in {}'.format(metric_name))
            plt.tight_layout()
            plt.savefig(os.path.join(self.mc_opts['target_dir'], '{}_decrease_{}.png'.format(filesafe(metric_name), self.param_str)), dpi=300)
            plt.close()

    def run(self):
        random_seed = self.mc_opts['random_seed']

        if not os.path.exists(self.mc_opts['target_dir']):
            os.makedirs(self.mc_opts['target_dir'])

        results_file = os.path.join(self.mc_opts['target_dir'], 'results_{}.jbl'.format(self.param_str))
        if self.mc_opts['reload_results'] and os.path.exists(results_file):
            results = joblib.load(results_file)
        else:
            results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(self.experiment)(random_seed + exp_id) 
                    for exp_id in range(self.opts['n_experiments']))
        joblib.dump(results, results_file)
        
        params = {}
        metrics = {}
        for method_name in self.methods.keys():
            params[method_name] = np.array([results[i][0][method_name] for i in range(self.opts['n_experiments'])])
            metrics[method_name] = {}
            for metric_name in self.metrics.keys():
                metrics[method_name][metric_name] = np.array([results[i][1][method_name][metric_name] for i in range(self.opts['n_experiments'])])
        
        if self.mc_opts['plot_params']:
            self.plot_param_histograms(params)
        
        self.plot_metrics(metrics)
        self.plot_metric_comparisons(metrics)

        return params, metrics



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='config file')
    args = parser.parse_args(sys.argv[1:])
    
    config = importlib.import_module(args.config)
    mc = MonteCarlo(config.CONFIG)
    mc.run()
    