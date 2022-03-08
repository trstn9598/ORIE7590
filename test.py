import BD_simulator, BirthDeathBessel
import numpy as np
from scipy.special import jv, laguerre, poch
from math import comb, factorial, exp, sqrt

def estimate_SqBesselBD(N = 10**6, t = 0, x0 = [0], test = 'Bessel', args = []):
	"""
	Test Bessel BD simulator against exact formulas for Bessel functions

	:param N: int, Number of simulations
	:param T: positive float, Simulation horizon
	:param x0: list of positive float, initial values of X
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


	x0 = np.array(x0)
	if x0.ndim == 0:
		x0 = np.array([x0])
	y0_poisson = np.array([np.random.poisson(x, N) for x in x0]).flatten()  # Map x0 by Poisson kernel
	# last_states = BD_simulator.bd_simulator(t, y0_poisson, len(y0_poisson)).reshape(len(x0), N)
	last_states = np.array([np.random.binomial(n, 0.5) for n in y0_poisson]).reshape(len(x0), N) #placeholder for simulation
	estimate = exp(t)*np.mean(np.vectorize(f)(last_states), axis = 1)
	return estimate

def estimate_SqBessel(t = 0, x0 = [0], test = 'Bessel', args = []):
	if test == 'Bessel':
		f = jv
	elif test == 'poly':
		if len(args) < 1:
			print('No coefficients provided')
			coef = []
		else: 
			coef = args[0]
		f = lambda x : np.polyval(coef, x)
	else:
		f = lambda x : 0

	x0 = np.array(x0)
	if x0.ndim == 0:
		x0 = np.array([x0])
	estimate = np.vectorize()(0, 2*np.sqrt(x0))
	return estimate

def discrete_poly(n, coef):
	return sum([coef[i]*poch(n - i + 1, i) for i in range(len(coef)) if n >= i])

x0 = range(10)
coef = [1]
print(estimate_SqBesselBD(N = 10, t = 1, x0 = x0, test = 'poly', args = [coef]))
# print(estimate_SqBessel(t = 1, x0 = x0))

