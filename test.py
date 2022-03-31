from BD_simulator import MC_BESQ_gateway, MC_BESQviaLaguerre_gateway, MC_BESQ_hankel, exact_BESQ, hankel_reparam
import numpy as np
from math import exp
import hankel
from scipy.special import  eval_laguerre 


testno = 0
if False:
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

if False:
	# TEST: BESQ simulation using (reparametrized) Hankel transform
	testno += 1
	num_paths = 10**2
	x0_array = range(10)
	times = [0.2, 0.5, 1, 2, 5]
	print('Test ', testno, ': Hankel simulation')
	print('Initial values: ', x0_array)
	print('Times: ', times)
	if True:
		test_fn1 = lambda x : 1/np.sqrt(x)
		print('Test function: sqrt(x)')
		hankel_estimates =  np.array([[MC_BESQ_hankel(N = num_paths, t = t, x0 = x0, test = 'custom', function = test_fn1) for x0 in x0_array] for t in times])
		print('Estimates from Hankel simulation')
		print(hankel_estimates)
	if False:
		test_fn2 = lambda x : np.exp(-x)
		print('Test function: exp(-x)')
		hankel_estimates =  np.array([[MC_BESQ_hankel(N = num_paths, t = t, x0 = x0, test = 'custom', function = test_fn2) for x0 in x0_array] for t in times])
		print('Estimates from Hankel simulation')
		print(hankel_estimates)


for x in range(10):
	f = lambda x : eval_laguerre(2, 2*x)*np.exp(-x)
	print('x = ', x)
	print(f(x + 1))
	print(hankel_reparam(x + 1, f))








