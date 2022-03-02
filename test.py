import BD_simulator, BirthDeathBessel
import numpy as np
from scipy.special import jv
from math import comb, factorial, exp, sqrt

def bd_test_bessel(N = 10**6, T = 0, x0 = [0]):
	"""
	Test Bessel BD simulator against exact formulas for Bessel functions

	:param N: int, Number of simulations
	:param T: positive float, Simulation horizon
	:param x0: list of positive float, initial values of X
	"""
	x0 = np.array(x0)
	if x0.ndim == 0:
		x0 = np.array([x0])
	y0_poisson = np.array([np.random.poisson(x, N) for x in x0]).flatten()  # Map x0 by Poisson kernel
	# bd = BirthDeathBessel.BirthDeathBessel(x0 = y0_poisson)
	# bd.simulate(dt = T)
	# last_states = np.array(bd.states_at_time(T)).reshape(len(x0), N)
	last_states = BD_simulator.bd_simulator(T, y0_poisson).reshape(len(x0), N)
	j_estimate = exp(T)*np.mean(np.vectorize(laguerre_poly)(last_states, 1), axis = 1)
	j_actual = np.vectorize(jv)(0, 2*np.sqrt(x0))
	return j_actual, j_estimate
	

def laguerre_poly(n, x):
	"""
	Compute value of Laguerre polynomials
	
	:param n: non-negative int, index of Lagerre polynomial, 
	:param x: positive float 
	"""

	return sum([((-x)**k)*comb(n, k)/factorial(k) for k in range(n + 1)])



print(bd_test_bessel(N = 10**4, T = 1, x0 = range(100)))


