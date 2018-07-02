import numpy as np

def l1_error(x, y): return np.linalg.norm(x-y, ord=1)
def l2_error(x, y): return np.linalg.norm(x-y, ord=2)
def bias(x, y): return x - y
def raw_estimate(x, y): return x
def truth(x, y): return y
def raw_estimate_nonzero_truth(x, y): return x[y>0]


def transform_identity(x, dgp, method, metric, config):
    return x[dgp][method][metric]

def transform_diff(x, dgp, method, metric, config):
    return x[dgp][method][metric] - x[dgp][config['proposed_method']][metric]

def transform_ratio(x, dgp, method, metric, config):
    return 100 * (x[dgp][method][metric] - x[dgp][config['proposed_method']][metric]) / x[dgp][method][metric]
