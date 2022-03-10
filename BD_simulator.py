# ORIE 7590
import numpy as np
from bd_sim_cython import discrete_bessel_sim, discrete_laguerre_sim


def bd_simulator(t, x0, num_paths, method='bessel', num_threads=4):
    """
    :param t: terminal time, double
    :param x0: initial state, callable or int
    :param num_paths: number of paths
    :param method: method of simulating birth-death chain, currently support 'bessel' and 'laguerre'
    :param num_threads: number of threads for multiprocessing
    :return: ndarray of simulated result at terminal time
    """
    if isinstance(x0, int):
        x0_array = np.array([x0]*num_paths, dtype=np.int64)
    else:
        x0_array = np.array([x0() for _ in range(num_paths)], dtype=np.int64)

    output = np.zeros(dtype=np.int64, shape=num_paths)
    if method == 'bessel':
        discrete_bessel_sim(t, x0_array, num_paths, output, int(num_threads))
    else:
        discrete_laguerre_sim(t, x0_array, num_paths, output, int(num_threads))

    return output


# exp = np.random.exponential
# def bd_one_path(t, x0):
#     """
#     simulate a birth-death proecss X at time t.
#
#     :param t: float, terminal time
#     :param x0: initial value of X
#     :return: one realization of X_t
#     """
#
#     s = 0
#     state = x0
#
#     while True:
#         birth_rate = state + 1
#         death_rate = state
#         arrival_rate = birth_rate + death_rate
#         time_to_arrival = exp(1/arrival_rate)
#         s += time_to_arrival
#         # stop and return when exceeds target time
#         if s > t:
#             return state
#         # update
#         if np.random.rand() < death_rate / arrival_rate:
#             state -= 1
#         else:
#             state += 1
#
#
# def bd_simulator(t, x0):
#     """
#     :param t: terminal time
#     :param x0: list of initial values from certain distribution
#     :return: list of simulated X_t
#     """
#
#     num_iter = len(x0)
#     result = np.zeros(num_iter, dtype = np.int64)
#
#     for i in range(num_iter):
#         result[i] = bd_one_path(t, x0[i])
#
#     return result
