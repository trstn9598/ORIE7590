# ORIE 7590
import numpy as np
from bd_sim_cython import discrete_bessel_sim, discrete_laguerre_sim
from scipy.special import jv, laguerre, poch, eval_laguerre, j0
from scipy.integrate import quad
from math import comb, factorial, exp, sqrt, log
import hankel


def bd_simulator(t, x0, num_paths, method='bessel', num_threads=4):
    """
    :param t: terminal time, double
    :param x0: initial state, callable or int
    :param num_paths: number of paths
    :param method: method of simulating birth-death chain, currently support 'bessel' and 'laguerre'
    :param num_threads: number of threads for multiprocessing
    :return: ndarray of simulated result at terminal time
    """
    if isinstance(x0, int):
        x0_array = np.array([x0]*num_paths, dtype=np.int64)
    else:
        x0_array = np.array([x0() for _ in range(num_paths)], dtype=np.int64)

    output = np.zeros(dtype=np.int64, shape=num_paths)
    if method == 'bessel':
        discrete_bessel_sim(t, x0_array, num_paths, output, int(num_threads))
    else:
        discrete_laguerre_sim(t, x0_array, num_paths, output, int(num_threads))

    return output

def MC_BESQ_gateway(N = 10**6, t = 0, x0 = 0, test = 'bessel', method = 'bessel', args = [], num_decimal = 4):
    """
    Monte Carlo estimator of expected BESQ using dBESQ simulation or dLaguerre simulation
    :param N: int, Number of simulations
    :param T: positive float, Simulation horizon
    :param x0: initial value of X
    :param method: simulation method, currently support {'bessel', 'laguerre', 'bessel-delay', 'laguerre-delay'}
    :param test: defines test function
    :args: arguments to define test function
    """
    if method == 'bessel':
        if test == 'bessel':
            f = lambda n : eval_laguerre(n, 1)
            s = t
    elif method == 'laguerre':
        if test == 'bessel':
            f = lambda n : eval_laguerre(n, 1+t)
            s = log(t + 1)
    elif method == 'bessel-delay':
        method = 'bessel'
        if test == 'bessel':
            f = lambda n : j0(2*np.sqrt(np.random.gamma(n+1)))
            s = t - 1
    elif method == 'laguerre-delay':
        method = 'laguerre'
        if test == 'bessel':
            f = lambda n : j0(2*np.sqrt(np.random.gamma(n+1) * (t/2 + 1/2)))
            s = log(t/2 + 1/2)
    
    def poisson_x0():
        return np.random.poisson(x0)
    xt_array = bd_simulator(s, x0=poisson_x0, num_paths=N, method=method, num_threads=4)
    return np.mean(f(xt_array)).round(num_decimal)

def MC_Laguerre_gateway(N = 10**6, t = 0, x0 = 0, test = 'laguerre', method = 'laguerre', args = [], num_decimal = 4):
    """
    Monte Carlo estimator of expected Laguerre using dLaguerre simulation or dLaguerre simulation
    :param N: int, Number of simulations
    :param T: positive float, Simulation horizon
    :param x0: initial value of X
    :param method: simulation method, currently support {'laguerre', 'laguerre-delay'}
    :param test: defines test function
    :args: arguments to define test function
    """
    if method == 'laguerre':
        if test == 'laguerre':
            f = lambda m : eval_meixner(args['n'], m)
            s = t
    elif method == 'laguerre-delay':
        if test == 'laguerre':
            f = lambda m : eval_laguerre(args['n'], np.random.gamma(m+1)/2)
            s = t - log(2)
    
    def poisson_x0():
        return np.random.poisson(x0)
    xt_array = bd_simulator(s, x0=poisson_x0, num_paths=N, method='laguerre', num_threads=4)
    return np.mean(f(xt_array)).round(num_decimal)

def MC_BESQ_hankel(N = 10**6, t = 0, x0 = 0, test = 'custom', function = lambda x : 0, args = [], num_decimal = 4):
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
    return np.mean(estimates).round(num_decimal)

def discrete_poly(n, coef):
    return sum([coef[i]*poch(n - i + 1, i) for i in range(len(coef)) if n >= i])

def exact_BESQ(t = 0, x0 = 0, num_decimal = 4):
    return (exp(-t)*jv(0, 2*np.sqrt(x0))).round(num_decimal)

def exact_Laguerre(t = 0, x0 = 0, n = 0, num_decimal = 4):
    return (exp(-t*n)*eval_laguerre(n, x0)).round(num_decimal)

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

def meixner(n, m):
    """
    Mexiner polynomial Mn(m) = \sum_{k=0}^m (-1)^k {m \choose k} {n \choose k}
    :param n: int
    :param m: int
    """
    summ, summand = 1, 1
    n1, m1 = n+1, m+1
    for k in range(1, min(m,n)+1):
        summand *= - (m1/k - 1) * (n1/k - 1)
        summ += summand
    return summ

def eval_meixner(n, m):
    """
    Evalutaiton of Mexiner polynomial Mn at ms = (m1, ..., ml)
    :param n: int
    :param m: array
    """
    return np.asarray([meixner(n, mi) for mi in m])
