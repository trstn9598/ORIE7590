import BD_simulator, BirthDeathBessel
import numpy as np
<<<<<<< Updated upstream
# J: Bessel function
from scipy.special import jv as J
# L: Laguerre polynomial
from scipy.special import  eval_laguerre as L


def test_bessel_bd(N = 10**6, T = 0, x0 = [0]):
	"""
	Test Bessel BD simulator against exact formulas for Bessel functions

	:param N: int, Number of simulations
	:param T: positive float, Simulation horizon
	:param x0: initial values of X, positive double
	"""
	# x0 = np.array(x0)
	# if x0.ndim == 0:
	# 	x0 = np.array([x0])
	# y0_poisson = np.array([np.random.poisson(x, N) for x in x0]).flatten()  # Map x0 by Poisson kernel
	# bd = BirthDeathBessel.BirthDeathBessel(x0 = y0_poisson)
	# bd.simulate(dt = T)
	# last_states = np.array(bd.states_at_time(T)).reshape(len(x0), N)
	# last_states = BD_simulator.bd_simulator(T, y0_poisson).reshape(len(x0), N)

	def poisson_x0():
		return np.random.poisson(x0)

	last_states = BD_simulator.bd_simulator(T, x0=poisson_x0, num_paths=N, num_threads=1)
	j_actual = np.exp(-T) * J(0, 2*np.sqrt(x0))
	j_estimate = np.mean(L(last_states, 1))

	return j_actual, j_estimate
	

# def laguerre_poly(n, x):
# 	"""
# 	Compute value of Laguerre polynomials
#
# 	:param n: non-negative int, index of Lagerre polynomial,
# 	:param x: positive float
# 	"""
#
# 	return sum([((-x)**k)*comb(n, k)/factorial(k) for k in range(n + 1)])

# print(bd_test_bessel(N = 10**4, T = 1, x0 = range(100)))
=======


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





>>>>>>> Stashed changes


