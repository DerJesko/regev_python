import secrets as sec
import numpy as np
import math
import struct
from Crypto.Cipher import AES
from Crypto.Util import Counter


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
