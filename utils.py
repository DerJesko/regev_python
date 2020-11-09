import secrets as sec
import numpy as np
import math
import struct


def single_float_uniform():
    """ Returns a uniformly random 32 bit float """
    return (sec.randbits(32)) / ((1 << 32) - 1)


def gaussian_iter():
    while True:
        a = single_float_uniform()
        a = math.sqrt(-2*math.log(a))
        b = single_float_uniform()
        b = 2 * math.pi * b
        yield a * math.sin(b)
        yield a * math.cos(b)


def gaussian(bound, shape=1):
    new_gauss = gaussian_iter()
    std_deviation = bound * 2
    arr = np.zeros(shape, dtype=float)
    for idx, _ in np.ndenumerate(arr):
        a = next(new_gauss) * std_deviation
        while abs(a) > bound:
            a = next(new_gauss) * std_deviation
        arr[idx] = a
    return mround(arr)


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
