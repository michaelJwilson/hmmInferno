import torch
from torch.distributions import negative_binomial

class NegativeBinomial():
    """
    Models a Negative Binomial distribution, i.e. distribution of the number of successful
    independent and identical Bernoulli trials before total_count failures are achieved.

    total_count (float or Tensor) – non-negative number of negative Bernoulli trials until stop.
    probs (Tensor) – Event probabilities of success in the half-open interval [0, 1).
    logits (Tensor) – Event log-odds for probabilities of success, probs = 1 / (1 + exp(-logits)).

    See:
        https://pytorch.org/docs/stable/distributions.html
        https://en.wikipedia.org/wiki/Negative_binomial_distribution
    """
    def __init__(self, total_count, probs):
        self.dist = negative_binomial.NegativeBinomial(
            total_count, probs=probs, validate_args=True
        )

    @property
    def mean(self):
        return self.dist.mean

    @property
    def mode(self):
        return self.dist.mode

    @property
    def variance(self):
        return self.dist.variance
    
    def log_prob(self, value):
        return self.dist.log_prob(value)

    def sample(self):
        return self.dist.sample()

    
if __name__ == "__main__":
    num_fail, success_prob = 15, 0.25
    dist = NegativeBinomial(num_fail, success_prob)
    
    fail_prob = 1. - success_prob
    mean_trials = num_fail / fail_prob
    mean_success = mean_trials - num_fail

    assert torch.allclose(dist.mean, torch.tensor([mean_success]))
    
    print(dist.mean, mean_success)
    
    
