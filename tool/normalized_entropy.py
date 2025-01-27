import numpy as np
import os

def normalized_entropy(s):
    s = np.array(s)
    k = len(s)
    entropy = -np.sum(s * np.log(s + 1e-10))
    normalized_entropy = entropy / np.log(k)
    return normalized_entropy


