import BD_simulator
import numpy as np
from scipy.special import jv, laguerre, poch
from scipy.integrate import quad
from math import comb, factorial, exp, sqrt, log
import hankel

def MC_BESQ_gateway(N = 10**6, t = 0, x0 = 0, test = 'bessel', args = []):
	"""
	Monte Carlo estimator of expected BESQ using dBESQ simulation
	:param N: int, Number of simulations
	:param T: positive float, Simulation horizon
	:param x0: initial value of X
	:param test: defines test function
	:args: arguments to define test function
	"""
	if test == 'bessel':
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
	xt_array = BD_simulator.bd_simulator(t, x0=poisson_x0, num_paths=N, method = 'bessel', num_threads=4)
	return np.mean(np.vectorize(f)(xt_array))

def MC_BESQviaLaguerre_gateway(N = 10**6, t = 0, x0 = 0, test = 'bessel', args = []):
	"""
	Monte Carlo estimator of expected BESQ using dLaguerre simulation
	:param N: int, Number of simulations
	:param T: positive float, Simulation horizon
	:param x0: initial value of X
	:param test: defines test function
	:args: arguments to define test function
	"""
	if test == 'bessel':
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
	xt_array = BD_simulator.bd_simulator(t = log(t + 1), x0=poisson_x0, num_paths=N, method = 'laguerre', num_threads=4)
	return np.mean(np.vectorize(f)(xt_array))

def MC_BESQ_hankel(N = 10**6, t = 0, x0 = 0, test = 'custom', function = lambda x : 0, args = []):
	"""
	Monte Carlo estimator of expected BESQ using Hankel transform and Exponential r.v.
	:param N: int, Number of simulations
	:param T: positive float, Simulation horizon
	:param x0: initial value of X
	:param test: defines test function
	:param function: custom test function
	:args: arguments to define test function
	"""
	j0 = lambda x : jv(0, 2*np.sqrt(x))
	if test == 'bessel':
		f = j0
	elif test == 'poly':
		if len(args) < 1:
			print('No coefficients provided')
			coef = []
		else: 
			coef = args[0]
		f = lambda x : np.polyval(coef, x)
	else:
		f = function

	estimates = np.zeros(N)
	for n in range(N):
		Z = np.random.exponential(t)
		estimates[n] = j0(x0*Z)*hankel_reparam(Z, f)/t
	return np.mean(estimates)


def estimate_SqBessel(t = 0, x0 = 0, test = 'bessel', args = []):
	# DO NOT USE YET
	if test == 'bessel':
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

def hankel_reparam(z, f):
	"""
	Monte Carlo estimator of expected BESQ using Hankel transform and Exponential r.v.
	Based on S. G. Murray and F. J. Poulin, “hankel: A Python library for performing simple and accurate Hankel transformations”, Journal of Open Source Software, 4(37), 1397, https://doi.org/10.21105/joss.01397
	:param z: positive float
	:param f: function in L^2(R_+)
	"""
	ht = hankel.HankelTransform(
    	nu= 0,     # The order of the bessel function
    	N = 120,   # Number of steps in the integration
    	h = 0.03   # Proxy for "size" of steps in integration
	)
	return 2*ht.transform(lambda x: f(x**2), 2*np.sqrt(z), ret_err = False)


x0 = 1
coef = [0, 1]
t = 0.1
# print(MC_BESQ_gateway(N = 10**4, t = t, x0 = x0, test = 'bessel'))
# print(MC_BESQviaLaguerre_gateway(N = 10**4, t = t, x0 = x0, test = 'bessel')
print(estimate_SqBessel(t = t, x0 = x0))
print(MC_BESQ_hankel(N = 10**3, t = t, x0 = x0, test = 'poly', args = [coef]))
# print(hankel_modified(np.random.exponential(t), lambda x : np.sqrt(x)))
