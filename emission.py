import torch
import logging
from utils import get_device, get_log_probs_precision
from dist import NegativeBinomial

LOG_PROBS_PRECISION = get_log_probs_precision()

logger = logging.getLogger(__name__)

class CategoricalEmission(torch.nn.Module):
    """
    Categoical emission from a bookend + n_states hidden states, (0, 1, .., n_states),
    to n_obvs emission classes.
    """

    def __init__(self, n_states, n_obvs, diag=False, device=None):
        super(CategoricalEmission, self).__init__()

        self.device = get_device() if device is None else device

        # NB we assume a bookend state to transition in/out.
        self.n_states = 1 + n_states

        # NB number of possible observable classes + hidden observation emitted by bookend state.
        self.n_obvs = 1 + n_obvs

        logger.info(
            f"Creating CategoricalEmission (diag={diag}) with {n_states} hidden states & {n_obvs} observed classes on device={self.device}"
        )

        self.log_em = self.init_emission(diag=diag)

        # NB emissions to/from bookend state should not be trained.
        self.log_em_grad_mask = torch.ones(
            (self.n_states, self.n_obvs), requires_grad=False, device=self.device
        )

        self.log_em_grad_mask[0, :] = 0
        self.log_em_grad_mask[:, 0] = 0

    def init_emission(self, log_probs_precision=LOG_PROBS_PRECISION, diag=False):
        # NB simple Markov model, where the hidden state is emitted.
        if diag:
            log_em = (
                torch.eye(self.n_states, self.n_obvs, device=self.device)
                .log()
                .clip(min=log_probs_precision, max=-log_probs_precision)
            )
        else:
            log_em = torch.randn(self.n_states, self.n_obvs, device=self.device)

        # NB nn.Parameter marks this to be optimised via torch.
        log_em = torch.nn.Parameter(log_em)
        log_em.data = CategoricalEmission.normalize_emission(
            log_em.data, log_probs_precision=log_probs_precision
        )

        return log_em

    @classmethod
    def normalize_emission(cls, log_em, log_probs_precision=LOG_PROBS_PRECISION):
        # NB emit only a bookend token from the bookend state.
        log_em[0, :] = log_probs_precision
        log_em[0, 0] = 0.0

        # NB only the bookend state emits a bookend token, to machine precision.
        log_em[1:, 0] = log_probs_precision

        # NB see https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
        return log_em.log_softmax(dim=1)

    def log_emission(self, state, obs):
        """
        Getter for log_em with broadcasting.
        """
        if state is None:
            return self.log_em[:, obs]
        elif obs is None:
            return self.log_em[state, :]
        else:
            return self.log_em[state, obs]

    def forward(self, obs):
        # NB equivalent to normalized self.emission(None, obs) bar bookend row.
        return self.log_emission(None, obs).log_softmax(dim=0)

    def validate(self):
        logger.info(f"Emission log probability matrix:\n{self.log_em}\n")

    def to_device(self, device):
        self.device = device
        self.log_em = self.log_em.to(device)
        self.log_em_grad_mask = self.log_em_grad_mask.to(device)

        return self

    @property
    def parameters_dict(self):
        """
        Dict with named torch parameters.
        """
        return {"log_em": self.log_em}

class TranscriptEmission(torch.nn.Module):
    """
    Emission model for spatial transcripts, with a negative binomial distribution.
    """
    def __init__(self, total_genome_transcipts, baseline_exp, copy_state, eff_read_depth, phi, device=None):
        super(TranscriptEmission, self).__init__()

        self.device = get_device() if device is None else device

        logger.info(
            f"Creating TranscriptEmission with total_count={num_fail:.4f} & probs={prob_success:.4f} on device={self.device}"
        )
        
        num_success = total_genome_transcipts * baseline_exp * (agm + bgm) / eff_read_depth
        prob_success = phi

        exp_trials = torch.round(torch.tensor(num_success / prob_success))

        # TODO CHECK
        num_fail = exp_trials - num_success

        self.dist = NegativeBinomial(num_fail, probs=prob_success)

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

    def log_emission(self, state, obs):
        """
        Getter for log_em with broadcasting
        """
        raise NotImplementedError()

    def forward(self, obs):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()

    def to_device(self, device):
        self.device = device
        self.dist.to_device(device)

        return self

    @property
    def parameters_dict(self):
        """
        Dict with named torch parameters.
        """
        return {"total_count": self.dist.total_count, "probs": self.dist.probs}

    
if __name__ == "__main__":
    num_fail, prob_success, device = 15, 0.25, "cpu" 
    emitter = TranscriptEmission(num_fail, prob_success, device=device)

    assert torch.allclose(emitter.mean, torch.tensor([5.0]))

    print("Done.")
