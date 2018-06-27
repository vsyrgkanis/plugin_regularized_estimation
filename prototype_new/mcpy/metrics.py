import numpy as np

def l1_error(x, y): return np.linalg.norm(x-y, ord=1)
def l2_error(x, y): return np.linalg.norm(x-y, ord=2)