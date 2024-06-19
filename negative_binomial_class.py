import numba
import numpy as np
import pylab as pl
import scipy.stats as stats
import statsmodels.api as sm
from numpy import vectorize
from numba import njit
from scipy.optimize import root_scalar
from calicost.utils_distribution_fitting import Weighted_NegativeBinomial

"""                                                                                                                                                                                                                 
https://www.jstor.org/stable/2532104?seq=2                                                                                                                                                                           
"""

np.random.seed(420)

def nu_sum_log_core(alpha, yi_max):
    result = np.zeros(yi_max)

    for nu in range(0, yi_max):
        result[nu] = np.log(1. + alpha * nu)

    return np.cumsum(result)

# @njit                                                                                                                                                                                                              
def nu_sum_core(alpha, yi_max):
    result = np.zeros(yi_max)

    for nu in range(0, yi_max):
        result[nu] = nu / (1. + alpha * nu)

    return np.cumsum(result)

class Weighted_NegativeBinomial_Fast():
    """
    Negative Binomial model endog ~ NB(exposure * exp(exog @ params[:-1]), params[-1]), where exog is the design matrix, and params[-1] is 1 / overdispersion.
    This function fits the NB params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    Attributes
    ----------
    endog : array, (n_samples,)
        Y values.

    exog : array, (n_samples, n_features)
        Design matrix.

    weights : array, (n_samples,)
        Sample weights.

    exposure : array, (n_samples,)
        Multiplication constant outside the exponential term. In scRNA-seq or SRT data, this term is the total UMI count per cell/spot.
    """
    def __init__(self, endog, exog, weights, exposure, seed=0, **kwds):
        super(Weighted_NegativeBinomial, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.seed = seed
    #
    def nloglikeobs(self, params):
        nb_mean = np.exp(self.exog @ params[:-1]) * self.exposure
        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)
        n, p = convert_params(nb_mean, nb_std)
        llf = scipy.stats.nbinom.logpmf(self.endog, n, p)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    #
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append('alpha')
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)

        return super(Weighted_NegativeBinomial, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)
