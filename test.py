from BD_simulator import MC_BESQ_gateway, MC_BESQviaLaguerre_gateway, MC_BESQ_hankel
import numpy as np
from scipy.special import jv, laguerre, poch
from scipy.integrate import quad
from math import comb, factorial, exp, sqrt, log
import hankel

def exact_BESQ(t = 0, x0 = 0):
    return exp(-t)*jv(0, 2*np.sqrt(x0))


testno = 0
# TEST: (reparametrized) Bessel functions
# Methods: dBESQ simulation, dLaguerre simulation, exact BESQ
testno += 1
num_paths = 10**2
x0_array = range(10)
times = [0, 0.2, 0.5, 1, 2, 5]
dBESQ_estimates = [[MC_BESQ_gateway(N = num_paths, t = t, x0 = x0, test = 'bessel') for x0 in x0_array] for t in times]
dLaguerre_estimates = [[MC_BESQviaLaguerre_gateway(N = num_paths, t = t, x0 = x0, test = 'bessel') for x0 in x0_array] for t in times]
BESQ_values = [[exact_BESQ(t = t, x0 = x0) for x0 in x0_array] for t in times]
print('Test ', testno, ': Bessel functions')
print('Initial values: ', x0_array)
print('Times: ', times)
print('Estimates from dBESQ simulation:')
print(dBESQ_estimates)
print('Estimates from dLaguerre simulation:')
print(dLaguerre_estimates)
print('Exact BESQ computation:')
print(BESQ_values)




# x0 = 1
# coef = [0, 1]
# t = 0.1
# # print(MC_BESQ_gateway(N = 10**4, t = t, x0 = x0, test = 'bessel'))
# # print(MC_BESQviaLaguerre_gateway(N = 10**4, t = t, x0 = x0, test = 'bessel')
# print(exact_BESQ(t = t, x0 = x0))
# print(MC_BESQ_hankel(N = 10**3, t = t, x0 = x0, test = 'poly', args = [coef]))
# # print(hankel_modified(np.random.exponential(t), lambda x : np.sqrt(x)))







