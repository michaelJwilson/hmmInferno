import sys
import torch
import logging
from torch.distributions import NegativeBinomial
import torch.nn.functional as F
from emission import BookendDist
from utils import get_log_probs_precision, get_device, get_scalars, bookend_sequence
from torch.optim import Adam

LOG_PROBS_PRECISION = get_log_probs_precision()

logger = logging.getLogger(__name__)


class TranscriptEmission(torch.nn.Module):
    """
    Emission model for spatial transcripts, with a negative binomial distribution.
    """

    def __init__(
        self,
        n_states,
        spots_total_transcripts,
        baseline_exp,
        total_exp_read_depth=25,
        device=None,
        name="TranscriptEmission"
    ):
        super(TranscriptEmission, self).__init__()

        self.n_states = 1 + n_states
        self.device = get_device() if device is None else device
        self.name = name
        
        logger.warning(f"Assuming a total exp. read depth of {total_exp_read_depth}.")

        self.state_means, self.state_phis = self.init_emission(
            total_exp_read_depth=total_exp_read_depth
        )

        logger.info(f"Initialized {self.name} with means and phis:\n{self.state_means}\n{self.state_phis}")

        # NB baseline exp. per genomic segment, g e (1, .., G).                                                                                                                                                                                                          
        self.baseline_exp = baseline_exp

        # NB total genomic transcripts per spot, n e (1, .., N).                                                                                                                                                                                                         
        self.spots_total_transcripts = spots_total_transcripts
        
        # NB bookend is not to be optimized.
        self.state_grad_mask = torch.ones(
            self.n_states, requires_grad=False, device=self.device
        )
        self.state_grad_mask[0] = 0.0

    def init_emission(
        self, total_exp_read_depth=25, log_probs_precision=LOG_PROBS_PRECISION
    ):
        # NB generator for normal(0., 1.); state_means == Tn * Lg * mu
        state_means = total_exp_read_depth * torch.rand(
            self.n_states, device=self.device, requires_grad=True
        )
        state_means = torch.nn.Parameter(state_means)

        # NB initialise at high precision
        state_phis = torch.rand(self.n_states, device=self.device, requires_grad=True)
        state_phis = torch.nn.Parameter(state_phis)

        return state_means, state_phis

    def mask_grad(self):
        self.state_means.grad *= self.state_grad_mask
        self.state_phis.grad *= self.state_grad_mask
        return self
    
    def sample(self, states):
        # NB one hot requires int64
        hot_states = F.one_hot(states.to(torch.int64), num_classes=self.n_states).float()
        
        means = hot_states @ self.state_means
        phis = hot_states @ self.state_phis
        
        # TODO inefficient                                                                                                                                                                                                                                               
        result = torch.stack(
            [
                NegativeBinomial(
                    total_count=means[i],
                    probs=means[i] / (means[i] + phis[i]),
                ).sample()
                for i in range(len(states))
            ],
            dim=0,
        )

        result[states == 0] = 0.
        return result

    def forward(self, obs, states=None):
        if states is None:
            states = torch.tensor(range(self.n_states), dtype=torch.int32, device=self.device)
        
        # NB forward is to be used for training only.
        hot_states = F.one_hot(states.to(torch.int64), num_classes=self.n_states).float()
        
        means = hot_states @ self.state_means
        phis = hot_states @ self.state_phis

        obs = torch.atleast_1d(obs)
        
        # TODO inefficient
        result = torch.stack(
            [
                NegativeBinomial(
                    total_count=means[i],
                    probs=means[i] / (means[i] + phis[i]),
                ).log_prob(obs[i])
                for i in range(len(obs))
            ],
            dim=0,
        )

        # TODO HACK
        # result[states == 0] = 1.
            
        return result.sum()

    def finalize_training(self):
        # NB no finalization steps required
        return self

    def log_emission(self, state, obs):
        """
        Getter for log_em with broadcasting
        """
        if state is None:
            if obs is None:
                raise NotImplementedError()
            else:
                state = self.state_dists.keys()
                return self.forward(obs, state)
        elif obs is None:
            raise NotImplementedError()
        else:
            return self.forward(obs, state)

    def torch_training(self, states, obvs, optimizer=None, n_epochs=3_000, lr=1.0e-2):
        # NB weight_decay=1.0e-5
        optimizer = Adam(self.parameters(), lr=lr)

        # NB set model to training mode - important for batch normalization & dropout -
        #    unnecessaary here, but best practice.
        self.train()

        for key in self.parameters_dict:
            logger.info(
                f"Ready to train {key} parameter with torch, initialised to:\n{self.parameters_dict[key]}"
            )
            
        # TODO weight scheduler.
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            loss = -self.forward(obvs, states)
            loss.backward()

            self = self.mask_grad()

            optimizer.step()

            if epoch % 10 == 0:
                logger.info(
                    f"Torch training epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}"
                )

        # NB evaluation, not training, mode.
        self.eval()
        self.finalize_training()

        loss = -self.forward(obvs, states)
        
        for key in self.parameters_dict:
            logger.info(
                f"Found optimised parameters for {key} to be:\n{self.parameters_dict[key]}"
            )

        logger.info(
            f"After training with torch for {n_epochs} epochs, found the log evidence to be {loss:.4f} by the forward method."
        )

        return n_epochs, loss

    def to_device(self, device):
        self.device = device
        self.baseline_exp = self.baseline_exp.to(device)
        self.spots_total_transcripts = self.spots_total_transcripts.to(device)
        self.state_means = self.state_means.to(device)
        self.state_phis = self.state_phis.to(device)
        self.state_grad_mask = self.state_grad_mask.to(device)
        
        return self

    def validate(self):
        logger.info(
            f"Negative Binomial emission with means and overdispersion:\n{self.state_means}\n{self.state_phis}\n"
        )

    @property
    def parameters_dict(self):
        """
        Dict with named torch parameters.
        """
        return {"state_means": self.state_means, "state_phis": self.state_phis}


if __name__ == "__main__":
    formatter = logging.Formatter(
        "%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    # TODO set seed for cuda / mps
    torch.manual_seed(314)

    train = True

    # NB K states with N spots, G segments on device.
    n_seq, K, N, G, device = 25, 8, 100, 25, "cpu"
    total_exp_read_depth = 25

    # NB
    spots_total_transcripts = torch.randn(N, device=device)
    baseline_exp = torch.randn(G, device=device)

    genEmitter = TranscriptEmission(
        K, spots_total_transcripts, baseline_exp, device=device, name="genEmitter"
    )
    
    states = torch.randint(low=1, high=1 + K, size=(n_seq,), device=device)
    states = bookend_sequence(states, device=device)
    
    obvs = genEmitter.sample(states)
    
    modelEmitter = TranscriptEmission(
        K, spots_total_transcripts, baseline_exp, device=device, name="modelEmitter"
    )

    result = modelEmitter.forward(obvs, states)

    if train:
        modelEmitter.torch_training(states, obvs)
        
    logger.info("Done.")
