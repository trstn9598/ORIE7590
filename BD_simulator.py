# ORIE 7590
import numpy as np
from bd_sim_cython import bd_sim
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


def bd_simulator(t, x0, k, num_threads=4):
    if isinstance(x0, int):
        x0_array = np.array([x0]*k, dtype=np.int64)
    else:
        x0_array = np.array([x0() for _ in range(k)], dtype=np.int64)

    output = np.zeros(dtype=np.int64, shape=k)

    bd_sim(t, x0_array, k, output, int(num_threads))

    return output
