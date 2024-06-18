import numba
import numpy as np
import pylab as pl
import scipy.stats as stats
import statsmodels.api as sm
from numpy import vectorize
from numba import njit
from scipy.optimize import root_scalar

"""                                                                                                                                                                  
https://www.jstor.org/stable/2532104?seq=2                                                                                                                            
"""

# np.random.seed(420)

def nu_sum_log_core(alpha, yi_max):
    result = np.zeros(yi_max)

    for nu in range(0, yi_max):
        result[nu] = np.log(1. + alpha * nu)

    return result

# @njit
def nu_sum_core(alpha, yi_max):
    result = np.zeros(yi_max)

    for nu in range(0, yi_max):
        result[nu] = nu / (1. + alpha * nu)

    return result

def log_like(samples, alpha, mu):
    nn = len(samples)
    yis, cnts = np.unique(samples, return_counts=True)
    mean = (yis * cnts).sum() / nn

    max_yi = yis.max()
    zeros = np.zeros_like(yis)

    mu = mean
    idx = np.maximum(zeros, (yis - 1)).tolist()

    nu_sums_complete = nu_sum_log_core(alpha, max_yi)
    nu_sums = nu_sums_complete[idx]

    result = (cnts * nu_sums).sum() / nn
    result += mean * np.log(mu)
    result -= (mean + 1. / alpha)*np.log(1. + alpha * mu)

    return result
        
def dispersion_minimas(samples, max_factor=10_000.):
    nn = len(samples)
    yis, cnts = np.unique(samples, return_counts=True)
    mean = (yis * cnts).sum() / nn
    std = np.std(yis)
    
    max_yi = yis.max()
    zeros = np.zeros_like(yis)

    mu = mean
    idx = np.maximum(zeros, (yis - 1)).tolist()
    
    def grad_func(alpha):
        nu_sums_complete = nu_sum_core(alpha, max_yi)
        nu_sums = nu_sums_complete[idx]
        
        result = (cnts * nu_sums).sum() / nn
        result += np.log(1. + alpha * mu) / alpha / alpha
        result -= mu * (mean + 1. / alpha) / (1. + alpha * mu)
        
        return result

    # bracket=[1.e-6, 10. * std],
    # root_scalar(grad_func, method="newton", x0=1.)
    return grad_func
    
    
# Parameters for the negative binomial distribution
mu, var, size = 10, 25., 10  # num. successes, prob. success., num_samples.

p = mu / var
r = mu * mu / (var - mu)

# print(p, r)

alpha = (var - mu) / mu / mu

# Sample from the negative binomial distribution
samples = stats.nbinom.rvs(r, p, size=size)

"""
grad_func = dispersion_minimas(samples)

alphas = 1.e-6 + np.arange(-10., 20., 0.2)
result = vectorize(grad_func)(alphas)

pl.plot(alphas, result)
pl.show()
"""

like = np.exp(log_like(samples, alpha, mu))

print(like)
print(np.exp(np.sum(stats.nbinom.logpmf(samples, r, p))))
