import array
import math
cimport numpy as np
from numpy cimport ndarray
import numpy


def serialize_ndarray(ndarray[np.int64_t, ndim=2] arr, int mod_len):
    cdef Py_ssize_t n = arr.shape[0]
    cdef Py_ssize_t m = arr.shape[1]
    cdef list b = []
    cdef np.int64_t x
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    for i in range(n):
        for j in range(m):
            x = arr[(i,j)]
            for k in range(mod_len):
                b.append((x & 255))
                x = x >> 8
    return bytes(b)

def deserialize_ndarray(bytearray b, tuple shape, int mod_len):
    cdef Py_ssize_t n = shape[0]
    cdef Py_ssize_t m = shape[1]
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef np.int64_t x
    cpdef ndarray[np.int64_t, ndim=2] arr = numpy.zeros(shape, dtype=int)
    for i in range(n):
        for j in range(m):
            arr[(i,j)] = int.from_bytes(b[(i*m+j)*mod_len:(i*m+j+1)*mod_len],"little")
    return arr
    
