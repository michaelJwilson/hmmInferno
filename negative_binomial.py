import cProfile
import io
import pstats
import timeit
import warnings

import numba
import numpy as np
import pylab as pl
import scipy
import scipy.stats as stats
import statsmodels.api as sm
from calicost.utils_distribution_fitting import (
    Weighted_NegativeBinomial,
    convert_params,
)
from numba import config, njit, prange, threading_layer
from numpy import vectorize
from scipy.optimize import root_scalar
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tools.sm_exceptions import ValueWarning
from profile_context import ProfileContext
from negative_binomial_evals import log_like

"""                                                                                                                                                                  
https://www.jstor.org/stable/2532104?seq=2                                                                                                                            
"""

np.random.seed(314)

warnings.filterwarnings(
    "ignore", message="more exog_names than parameters", category=ValueWarning
)
warnings.filterwarnings(
    "ignore", message="df_resid differs from nobs - k_params", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="df_model + k_constant + k_extra differs from k_params",
    category=UserWarning,
)


@njit(cache=True, parallel=True)
def nu_sum_core(alpha, yi_max):
    result = np.zeros(yi_max)

    for nu in prange(0, yi_max):
        result[nu] = nu / (1.0 + alpha * nu)

    return np.cumsum(result)

@njit(cache=True, parallel=True)
def nu_log_sum_core(alpha, yi_max):
    result = np.zeros(yi_max)

    for nu in prange(0, yi_max):
        result[nu] = np.log(1. + alpha * nu)

    return np.cumsum(result)

def piegorsch_rootfunc(samples, weights=None):
    """
    TODO exog mus etc. 
    """
    if weights is None:
        weights = np.ones_like(samples)

    yis, cnts = np.unique(samples, return_counts=True)

    # NB Max. like. mean is the (weighted) sample mean.
    mu = (samples * weights).sum() / weights.sum()

    # NB sets upper limit on fitted alpha only.
    var = (weights * ((samples - mu)) ** 2.0).sum() / weights.sum()
    
    max_yi = yis.max()
    zeros = np.zeros_like(samples)
    idx = np.maximum(zeros, (samples - 1)).tolist()

    def grad_func(alpha):
        # NB sum over nu.
        nu_sums_complete = nu_sum_core(alpha, max_yi)
        nu_sums = nu_sums_complete[idx]

        result = (weights * nu_sums).sum() / weights.sum()
        result += np.log(1.0 + alpha * mu) / alpha / alpha
        result -= mu * (mu + 1.0 / alpha) / (1.0 + alpha * mu)

        return result

    return mu, np.sqrt(var), grad_func

def log_like_piegorsch(samples, alpha, mu, weights=None):
    if weights is None:
        weights = np.ones_like(samples)

    yis, cnts = np.unique(samples, return_counts=True)

    # NB Max. like. mean is the (weighted) sample mean.                                                                                                                   
    mu = (samples * weights).sum() / weights.sum()

    # NB sets upper limit on fitted alpha only.                                                                                                                           
    var = (weights * ((samples - mu)) ** 2.0).sum() / weights.sum()

    max_yi = yis.max()
    zeros = np.zeros_like(samples)
    idx = np.maximum(zeros, (samples - 1)).tolist()

    nu_log_sums_complete = nu_log_sum_core(alpha, max_yi)
    nu_sums = nu_log_sums_complete[idx]

    result = (weights * nu_sums).sum() / weights.sum()
    result += mean * np.log(mu)
    result -= (mean + 1. / alpha) * np.log(1. + alpha * mu)

    return result

class Weighted_NegativeBinomial_v2(GenericLikelihoodModel):
    """
    Negative Binomial model endog ~ NB(exposure * exp(exog @ params[:-1]), params[-1]), where exog is the design matrix, and params[-1] is 1 / overdispersion.
    This function fits the NB params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    See https://www.jstor.org/stable/2532104?seq=1

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
        super().__init__(endog, exog, **kwds)

        self.weights = weights
        self.exposure = exposure
        self.seed = seed

    def nloglikeobs(self, params):
        # NB params == (mus, overdispersion)
        nb_mean = np.exp(self.exog @ params[:-1]) * self.exposure
        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)

        n, p = convert_params(nb_mean, nb_std)

        # NB https://github.com/scipy/scipy/blob/v1.12.0/scipy/stats/_discrete_distns.py#L264-L370
        llf = scipy.stats.nbinom.logpmf(self.endog, n, p)

        # assert -llf.dot(self.weights) == -log_like(self.endog, n, p).dot(self.weights)

        llf = log_like(self.endog, n, p, version="v3")
        
        return -llf.dot(self.weights)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append("alpha")

        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)

        # super(Weighted_NegativeBinomial_Piegorsch, self)
        return super().fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )

    def fit_piegorsch(self, max_factor=10.0):
        dalpha = 1.0e-2
        mu, std, grad_func = piegorsch_rootfunc(self.endog, weights=self.weights)

        alpha = root_scalar(grad_func, bracket=(dalpha, max_factor * std)).root

        return (mu / self.exposure), alpha


if __name__ == "__main__":
    # NB num. successes, prob. success., num_samples.
    num_states, mu, var, size, nrepeat = 1, 10, 20.0, 200, 100
    alpha = (var - mu) / mu / mu

    print(mu, alpha)

    (r, p) = convert_params(mu, np.sqrt(var))
    samples = stats.nbinom.rvs(r, p, size=size)

    mean, std, grad_func = piegorsch_rootfunc(samples)

    exog = np.zeros((len(samples), num_states))
    exog[:, 0] = 1.0

    fitter = Weighted_NegativeBinomial_v2(
        endog=samples,
        exog=exog,
        weights=np.ones_like(samples),
        exposure=np.ones(num_states),
    )
    
    """
    with ProfileContext() as context:
        for ii in range(nrepeat):
            params = fitter.fit_piegorsch()

        print(params)
    """
    # title = r"Truth $(\alpha, \mu)$=" + f"({mu:.2f}, {alpha:.2f})"
    #
    # dalpha = 1.0e-2
    # alphas = dalpha + np.arange(0.0, 20.0, dalpha)
    #
    # pl.plot(alphas, vectorize(grad_func)(alphas))
    # pl.axhline(0.0, c="k", lw=0.5, label=r"$\hat \mu=$" + f"{est_mu:.4f}")
    # pl.axvline(est_alpha, c="k", lw=0.5, label=r"$\hat \alpha=$" + f"{est_alpha:.4f}")
    # pl.legend(frameon=False)
    # pl.title(title)
    # pl.show()
    
    start_params = mu + np.sqrt(var) * np.random.normal(size=num_states)
    start_params = np.concatenate((np.log(start_params), np.array([alpha])))

    start_log_like = fitter.nloglikeobs(start_params)

    with ProfileContext() as context:
        for ii in range(nrepeat):
            # NB disp controls output; method="bfgs"
            result = fitter.fit(
                start_params=start_params, disp=0, maxiter=1_500, xtol=1e-6, ftol=1e-6,
            )

        print(
            np.exp(result.params[:-1]),
            result.params[-1],
            fitter.nloglikeobs(result.params),
        )

    print("\n\nDone.\n\n")
