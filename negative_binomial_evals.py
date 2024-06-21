import io
import pstats
import cProfile
import numpy as np

from math import gamma
from scipy.stats import nbinom
from numba import njit, prange
from profile_context import ProfileContext

def log_like_v1(kk, nn, pp):
    """
    Standard scipy, cannot be jitted.
    """
    return nbinom.logpmf(kk, nn, pp)

# @njit(cache=True, parallel=True)
def factorial(nn):
    result = 1

    if nn <= 1:
        return result
    
    for ii in prange(2, nn + 1):
        result *= ii

    return result

# @njit(cache=True)
def stirlings(arg, max_arg=1_000):
    """
    return approx. to log(n!)
    """
    if arg > max_arg:
        return arg * np.log(arg) - arg # + O(ln arg)
    else:
        return np.log(factorial(arg))    
    
@njit(cache=True, parallel=False)
def nu_log_sum_core(yi, inv_alpha):
    result = 0.0

    for nu in prange(0, yi):
        result += np.log(inv_alpha + nu)
        
    return result

# @njit(cache=True)
def log_like_v2(kk, nn, pp):    
    result = nn * np.log(pp) + kk * np.log(1. - pp)
    result += np.log(gamma(kk + nn))
    result -= np.log(gamma(nn))
    result -= stirlings(kk)
    
    return result

# @njit(cache=True)
def log_like_v3(kk, nn, pp):
    result = nn * np.log(pp) + kk * np.log(1. - pp)
    result += nu_log_sum_core(yi=kk, inv_alpha=nn)
    result -= stirlings(kk)
    return result

def log_like(kk, nn, pp, version="v1"):
    if version == "v1":
        return log_like_v1(kk, nn, pp)
    elif version == "v2":
        return log_like_v2(kk, nn, pp)
    else:
        return log_like_v3(kk, nn, pp)
    
if __name__ == "__main__":
    nrepeat, version = 100_000, "v2"
    kk, nn, pp = 20, 30, 0.25

    # exp = np.log(gamma(kk + nn) / gamma(nn))
    # result = nu_log_sum_core(kk, nn)

    with ProfileContext() as context:                                                                                                                                                                                                                
        for ii in range(nrepeat):                                                                                                                                                                                                                    
            result = log_like(kk, nn, pp, version=version)
                                                                                                                                                                                                                                                     
        print(version, result)

    
    
