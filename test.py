import BD_simulator
import numpy as np
from scipy.special import jv, laguerre, poch
from math import comb, factorial, exp, sqrt, log

def MC_BESQ_gateway(N = 10**6, t = 0, x0 = 0, test = 'Bessel', args = []):
	"""
	Monte Carlo estimator of expected BESQ against test functions
	:param N: int, Number of simulations
	:param T: positive float, Simulation horizon
	:param x0: initial value of X
	:param test: defines test function
	:args: arguments to define test function
	"""
	if test == 'Bessel':
		f = lambda n : laguerre(n)(1)
	elif test == 'poly':
		if len(args) < 1:
			print('No coefficients provided')
			coef = []
		else: 
			coef = args[0]
		f = lambda n : discrete_poly(n, coef)
	else:
		f = lambda n : 0

	def poisson_x0():
		return np.random.poisson(x0)
	xt_array = BD_simulator.bd_simulator(t, x0=poisson_x0, num_paths=N, method = 'Bessel', num_threads=4)
	return np.mean(np.vectorize(f)(xt_array))

def MC_BESQviaLaguerre_gateway(N = 10**6, t = 0, x0 = 0, test = 'Bessel', args = []):
	"""
	Test Bessel BD simulator against exact formulas for Bessel functions

	:param N: int, Number of simulations
	:param T: positive float, Simulation horizon
	:param x0: initial value of X
	"""
	if test == 'Bessel':
		f = lambda n : laguerre(n)(1 + t)
	elif test == 'poly':
		if len(args) < 1:
			print('No coefficients provided')
			coef = []
		else: 
			coef = [args[0][i]*((1 + t)**i) for i in range(len(args[0]))]
		f = lambda n : discrete_poly(n, coef)
	else:
		f = lambda n : 0

	def poisson_x0():
		return np.random.poisson(x0)
	xt_array = BD_simulator.bd_simulator(t = log(t + 1), x0=poisson_x0, num_paths=N, method = 'Laguerre', num_threads=4)
	return np.mean(np.vectorize(f)(xt_array))


def estimate_SqBessel(t = 0, x0 = 0, test = 'Bessel', args = []):
	# DO NOT USE YET
	if test == 'Bessel':
		f = lambda x : exp(-t)*jv(0, 2*np.sqrt(x))
	elif test == 'poly':
		if len(args) < 1:
			print('No coefficients provided')
			coef = []
		else: 
			coef = args[0]
		f = lambda x : np.polyval(coef, x)
	else:
		f = lambda x : 0
	estimate = f(x0)
	return estimate

def discrete_poly(n, coef):
	return sum([coef[i]*poch(n - i + 1, i) for i in range(len(coef)) if n >= i])


x0 = 1
coef = [1]
t = 1
print(estimate_SqBesselBD(N = 10, t = t, x0 = x0, test = 'Bessel', args = []))
print(estimate_SqBessel(t = t, x0 = x0))

