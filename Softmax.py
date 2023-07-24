import numpy as np


def softmax(f):
    f -= np.max(f)
    return np.exp(f) / np.sum(np.exp(f))


print(softmax(np.array([3, 4, 7])))
