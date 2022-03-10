import BD_simulator, BirthDeathBessel
import numpy as np
# J: Bessel function
from scipy.special import jv as J
# L: Laguerre polynomial
from scipy.special import  eval_laguerre as L

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


# TEST: polynomials
# Methods: dBESQ simulation, dLaguerre simulation
nrounds = 10
degree = 10
for i in range(nrounds):
    coeff = np.random.standard_normal(degree+1)
    dBESQ_estimates_poly = [[MC_BESQ_gateway(N = num_paths, t = t, x0 = x0, test = 'poly', args = [coeff]) for x0 in x0_array] for t in times]
    dLaguerre_estimates = [[MC_BESQviaLaguerre_gateway(N = num_paths, t = t, x0 = x0, test = 'poly', args = [coeff]) for x0 in x0_array] for t in times]
print('Initial values: ', x0_array)
print('Times: ', times)
print('Estimates from dBESQ simulation:')
print(dBESQ_estimates_poly)
print('Estimates from dLaguerre simulation:')
print(dLaguerre_estimates_poly)
    

# x0 = 1
# coef = [0, 1]
# t = 0.1
# # print(MC_BESQ_gateway(N = 10**4, t = t, x0 = x0, test = 'bessel'))
# # print(MC_BESQviaLaguerre_gateway(N = 10**4, t = t, x0 = x0, test = 'bessel')
# print(exact_BESQ(t = t, x0 = x0))
# print(MC_BESQ_hankel(N = 10**3, t = t, x0 = x0, test = 'poly', args = [coef]))
# # print(hankel_modified(np.random.exponential(t), lambda x : np.sqrt(x)))




