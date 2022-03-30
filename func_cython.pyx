import numpy as np
import pandas as pd
cimport cython
from libcpp cimport bool
from libc.math cimport isnan, log, exp, NAN
from libc.stdlib cimport rand, RAND_MAX
from libcpp.vector cimport vector
from cython.parallel import prange, parallel

cpdef long meixner(long n, long m) nogil:
    """
    Mexiner polynomial Mn(m) = \sum_{k=0}^m (-1)^k {m \choose k} {n \choose k}
    :param n: int
    :param m: int
    """
    cdef:
        long summ = 1, summand = 1, n1 = n + 1, m1 = m + 1
    for k in range(1, min(m,n)+1):
        summand *= - (m1/k - 1) * (n1/k - 1)
        summ += summand
    return summ

cpdef long[:] eval_meixner(long n, long[:] m):
    """
    Evalutaiton of Mexiner polynomial Mn at ms = (m1, ..., ml)
    :param n: int
    :param m: array
    """
    cdef:
        long[:] output = m
    with nogil, parallel(num_threads=4):
        for i in prange(len(m), schedule='dynamic'):
            output[i] = meixner(n, m[i])
    return output
