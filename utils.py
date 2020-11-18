import secrets as sec
import numpy as np
import math
import struct
from Crypto.Cipher import AES
from Crypto.Util import Counter


class SeededRNG:
    def __init__(self, seed):
        self.stream = AES.new(seed, AES.MODE_CTR, counter=Counter.new(128))

    def next_bytes(self, num_bytes) -> bytes:
        return self.stream.encrypt(b"\x00" * num_bytes)

    def randbelow(self, i) -> int:
        i_bytes = math.ceil(i.bit_length()/8)
        i_len = math.ceil(i.bit_length())
        counter = 0
        while True:
            counter += 1
            a = self.next_bytes(i_bytes)
            a = int.from_bytes(a, 'little') >> (i_bytes * 8) - i_len
            if a < i:
                return a


def single_float_uniform(rng: SeededRNG):
    """ Returns a uniformly random 32 bit float """
    return (int.from_bytes(rng.next_bytes(4), "little")) / ((1 << 32) - 1)


def gaussian_iter(rng):
    while True:
        a = single_float_uniform(rng)
        a = math.sqrt(-2*math.log(a))
        b = single_float_uniform(rng)
        b = 2 * math.pi * b
        yield a * math.sin(b)
        yield a * math.cos(b)


def gaussian(bound, rng, shape=1):
    new_gauss = gaussian_iter(rng)
    std_deviation = bound * 2
    arr = np.zeros(shape, dtype=float)
    for idx, _ in np.ndenumerate(arr):
        a = next(new_gauss) * std_deviation
        while abs(a) > bound:
            a = next(new_gauss) * std_deviation
        arr[idx] = a
    return mround(arr)


def uniform(ubound: int, rng, shape=1, lbound: int = 0):
    arr = np.zeros(shape, dtype=int)
    for idx, _ in np.ndenumerate(arr):
        arr[idx] = rng.randbelow(ubound - lbound) + lbound
    return arr


def mround(arr):
    return np.around(arr).astype(int)


def mod_between(x, lbound, ubound):
    if lbound > ubound:
        return x <= ubound or lbound <= x
    else:
        return lbound <= x <= ubound


def serialize_ndarray(arr: np.array, mod_len: int) -> bytearray:
    b = bytearray()
    for x in np.nditer(arr):
        b.extend(bytearray(x.data)[:mod_len])
    return b


def deserialize_ndarray(b: bytes, shape, mod_len: int):
    arr = np.zeros(shape, dtype=int)
    i = 0
    for j, _ in np.ndenumerate(arr):
        arr[j] = int.from_bytes(b[mod_len * i:mod_len * (i+1)], "little")
        i += 1
    return arr
