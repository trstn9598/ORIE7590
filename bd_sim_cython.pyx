import numpy as np
import pandas as pd
cimport cython
from libcpp cimport bool
from libc.math cimport isnan, log, exp, NAN
from libc.stdlib cimport rand, RAND_MAX
from libcpp.vector cimport vector
from cython.parallel import prange, parallel


cpdef long bd_one_path(double t, long x0) nogil:
    """
    simulate a birth-death process X at time t.
    
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


cpdef void bd_sim(double t, long[:] x0, long num_path, long[:] output, int num_threads=4):
    cdef:
        Py_ssize_t iPath

    with nogil, parallel(num_threads=num_threads):
        for iPath in prange(num_path, schedule='dynamic'):
            output[iPath] = bd_one_path(t, x0[iPath])

    return
