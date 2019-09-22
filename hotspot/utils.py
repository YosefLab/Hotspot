import numpy as np
from numba import jit


@jit(nopython=True)
def center_values(vals, mu, var):
    out = np.zeros_like(vals)

    for i in range(len(vals)):
        std = var[i]**0.5
        if std == 0:
            out[i] = 0
        else:
            out[i] = (vals[i] - mu[i])/std

    return out
