import numpy as np
import pandas as pd
cimport cython
from libcpp cimport bool
from libc.math cimport isnan, log, exp, NAN
from libc.stdlib cimport rand, RAND_MAX
from libcpp.vector cimport vector
from cython.parallel import prange, parallel


cpdef long discrete_bessel_one_path(double t, long x0) nogil:
    """
    simulate a birth-death proecss X at time t.
    
    :param t: float, terminal time
    :param x0: initial value of X
    :return: in-place
    """
    cdef:
        double s = 0, time_to_arrival, birth_rate, death_rate, arrival_rate, U
        long state = x0

    while True:
        birth_rate = state + 1.0
        death_rate = state + 0.0
        arrival_rate = birth_rate + death_rate
        U = rand() / (RAND_MAX + 1.0)
        time_to_arrival = -log(1-U) / arrival_rate
        s += time_to_arrival
        # stop and return when exceeds target time
        if s > t:
            return state
        # update
        U = rand() / (RAND_MAX + 1.0)
        if U < (death_rate / arrival_rate):
            state -= 1
        else:
            state += 1

cpdef long discrete_laguerre_one_path(double t, long x0) nogil:
    """
    simulate a birth-death proecss X at time t.

    :param t: float, terminal time
    :param x0: initial value of X
    :return: in-place
    """
    cdef:
        double s = 0, time_to_arrival, birth_rate, death_rate, arrival_rate, U
        long state = x0

    while True:
        birth_rate = state + 1.0
        death_rate = state * 2.0
        arrival_rate = birth_rate + death_rate
        U = rand() / (RAND_MAX + 1.0)
        time_to_arrival = -log(1 - U) / arrival_rate
        s += time_to_arrival
        # stop and return when exceeds target time
        if s > t:
            return state
        # update
        U = rand() / (RAND_MAX + 1.0)
        if U < (death_rate / arrival_rate):
            state -= 1
        else:
            state += 1


cpdef void discrete_bessel_sim(double t, long[:] x0, long num_path, long[:] output, int num_threads=4):
    cdef:
        Py_ssize_t iPath

    with nogil, parallel(num_threads=num_threads):
        for iPath in prange(num_path, schedule='dynamic'):
            output[iPath] = discrete_bessel_one_path(t, x0[iPath])

    return


cpdef void discrete_laguerre_sim(double t, long[:] x0, long num_path, long[:] output, int num_threads=4):
    cdef:
        Py_ssize_t iPath

    with nogil, parallel(num_threads=num_threads):
        for iPath in prange(num_path, schedule='dynamic'):
            output[iPath] = discrete_laguerre_one_path(t, x0[iPath])

    return


cpdef long meixner(long n, long m) nogil:
    """
    Mexiner polynomial Mn(m) = \sum_{k=0}^m (-1)^k {m \choose k} {n \choose k}
    :param n: int
    :param m: int
    """
    cdef:
        long summ = 1, summand = 1, k = 1
        long n1, m1
    n1 = n + 1
    m1 = m + 1
    
    while k <= min(m,n):
        summand *= - (m1 - k) * (n1 - k)
        summand //= k * k
        summ += summand
        k += 1
    
    return summ


cpdef void cmeixner(long n, long[:] m, long num_path, long[:] output):
    """
    Evalutaiton of Mexiner polynomial Mn at ms = (m1, ..., ml)
    :param n: int
    :param m: array
    """
    cdef:
        Py_ssize_t iPath
    
    with nogil, parallel(num_threads=4):
        for iPath in prange(num_path, schedule='dynamic'):
            output[iPath] = meixner(n, m[iPath])
    
    return

