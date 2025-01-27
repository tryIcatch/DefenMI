import numpy as np


def convert_to_one_hot_max(arr):
    max_indices = np.argmax(arr, axis=1)

    one_hot = np.zeros_like(arr)
    one_hot[np.arange(arr.shape[0]), max_indices] = 1

    return one_hot