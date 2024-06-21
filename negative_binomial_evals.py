import io
import pstats
import cProfile
import numpy as np

from math import gamma
from scipy.stats import nbinom
from numba import njit, prange
from profile_context import ProfileContext

@njit(cache=True, parallel=False)
def factorial_scalar(nn, log=False):
    result = 0 if log else 1

    if nn <= 1:
        return result

    if log:
        for ii in prange(2, nn + 1):
            result += np.log(ii)
        
    else:
        for ii in prange(2, nn + 1):
            result *= ii

    return result

@njit(cache=True, parallel=False)
def factorial_core(unique_fac, unique_nn, log=False):
    for ii, un in enumerate(unique_nn):
        unique_fac[ii] = factorial_scalar(un, log=log)

    return unique_fac

# NB no njit as return_inverse kwarg.
def factorial(nn, log=False):
    result = np.ones_like(nn)
    unique_nn, idx = np.unique(nn, return_inverse=True)
    unique_fac = np.zeros_like(unique_nn)
    unique_fac = factorial_core(unique_fac, unique_nn, log=log)
    return unique_fac[idx]

def stirlings(arg, max_arg=None):
    """
    return approx. to log(n!)
    """
    arg = np.atleast_1d(arg)
    result = np.zeros_like(arg)

    if max_arg is not None:
        isin = (arg > max_arg)
    else:
        isin = np.zeros_like(result, dtype=bool) 
    
    result[isin] = arg[isin] * np.log(arg[isin]) - arg[isin]  # + O(ln arg)
    result[~isin] = factorial(arg[~isin], log=True)
    
    return result

@njit(cache=True, parallel=False)
def nu_log_sum_core(yi, inv_alpha):
    result = 0.0

    for nu in prange(0, yi):
        result += np.log(inv_alpha + nu)

    return result

@njit(cache=True, parallel=False)
def nu_log_sum_core_vec(result, yis, inv_alphas):
    for ii, yi in enumerate(yis):
        result[ii] = nu_log_sum_core(yi, inv_alphas[ii])

    return result

def nu_log_sum(yis, inv_alphas):
    yis = np.atleast_1d(yis)
    result = np.zeros_like(yis, dtype=float)

    if isinstance(inv_alphas, (int, float)):
        inv_alphas = inv_alphas * np.ones_like(yis)

    return nu_log_sum_core_vec(result, yis, inv_alphas)

def log_like_v1(kk, nn, pp):
    """                                                                                                                                                                                                                                              
    Standard scipy, cannot be jitted.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    NB n is the number of successes, p is the probability of success, k is # failures                                                                                                                                                                
    """
    return nbinom.logpmf(kk, nn, pp)

def log_like_v2(kk, nn, pp):
    result = nn * np.log(pp) + kk * np.log(1.0 - pp)
    result -= stirlings(kk)
    
    result += np.log(gamma(kk + nn))
    result -= np.log(gamma(nn))

    return result

def log_like_v3(kk, nn, pp):
    result = nn * np.log(pp) + kk * np.log(1.0 - pp)
    result -= stirlings(kk)
    
    result += nu_log_sum(kk, nn)
    
    return result

def log_like(kk, nn, pp, version="v1"):
    if version == "v1":
        return log_like_v1(kk, nn, pp)
    elif version == "v2":
        return log_like_v2(kk, nn, pp)
    else:
        return log_like_v3(kk, nn, pp)

if __name__ == "__main__":
    nrepeat, version = 1_000, "v1"

    # kk, nn, pp = 20, 30, 0.25
    kk, nn, pp = 8, 10, 0.41118372
    # kk, nn, pp = np.array([kk, kk]), np.array([nn, nn]), np.array([pp, pp])

    exp = np.log(gamma(kk + nn) / gamma(nn))
    result = nu_log_sum(kk, nn)

    # print(exp, result[0])
    
    with ProfileContext() as context:
        for ii in range(nrepeat):
            result = log_like(kk, nn, pp, version=version)

        print(version, result)
