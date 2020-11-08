import secrets as sec
import numpy as np


def uniform(limit, shape=1):
    arr = np.zeros(shape, dtype=int)
    for idx, _ in np.ndenumerate(arr):
        arr[idx] = sec.randbelow(limit)
    return arr


def binomial(limit, shape=1):
    arr = np.zeros(shape, dtype=int)
    for idx, _ in np.ndenumerate(arr):
        for _ in range(2 * limit + 1):
            arr[idx] += sec.randbelow(2)
    return arr - limit


def mround(arr):
    return np.around(arr).astype(int)


def mod_between(x, lbound, ubound):
    if lbound > ubound:
        return x <= ubound or lbound <= x
    else:
        return lbound <= x <= ubound
