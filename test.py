from BD_simulator import MC_BESQ_gateway, MC_BESQviaLaguerre_gateway, MC_BESQ_hankel, exact_BESQ
import numpy as np


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

if True:
	# TEST: BESQ simulation using (reparametrized) Hankel transform
	testno += 1
	num_paths = 10**2
	x0_array = range(10)
	times = [0.2, 0.5, 1, 2, 5]
	print('Test ', testno, ': Hankel simulation')
	print('Initial values: ', x0_array)
	print('Times: ', times)
	test_fn1 = lambda x : np.sqrt(x)
	print('Test function: sqrt(x)')
	hankel_estimates =  np.array([[MC_BESQ_hankel(N = num_paths, t = t, x0 = x0, test = 'custom', function = test_fn1) for x0 in x0_array] for t in times])
	print('Estimates from Hankel simulation')
	print(hankel_estimates)







