import sys
import torch
import logging
import numpy as np
from torch.distributions import NegativeBinomial
import torch.nn.functional as F
from emission import BookendDist
from utils import (
    get_log_probs_precision,
    get_device,
    set_scalars,
    bookend_sequence,
    no_grad,
)
from torch.optim import Adam
from dist import BookendDist

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
        name="TranscriptEmission",
    ):
        super(TranscriptEmission, self).__init__()

        self.n_states = 1 + n_states
        self.device = get_device() if device is None else device
        self.name = name

        logger.warning(f"Assuming a total exp. read depth of {total_exp_read_depth}.")

        state_log_means, state_log_frac_std = self.init_emission(
            total_exp_read_depth=total_exp_read_depth
        )

        self.state_log_means = torch.nn.Parameter(state_log_means.requires_grad_(True))
        self.state_log_frac_std = torch.nn.Parameter(state_log_frac_std.requires_grad_(True))

        logger.info(
            f"Initialized {self.name} with log means and log frac. std:\n{self.state_log_means}\n{self.state_log_frac_std}"
        )

        # NB baseline exp. per genomic segment, g e (1, .., G).
        self.baseline_exp = baseline_exp

        # NB total genomic transcripts per spot, n e (1, .., N).
        self.spots_total_transcripts = spots_total_transcripts

        self.bookend = BookendDist(device=self.device)

    @no_grad
    def init_emission(
        self,
        total_exp_read_depth=25,
        log_probs_precision=LOG_PROBS_PRECISION,
    ):
        """
        # NB generator for normal(0., 1.); state_means == Tn * Lg * mu
        state_means = total_exp_read_depth * torch.rand(
            self.n_states, device=self.device, requires_grad=True
        )
        """
        state_means = total_exp_read_depth * (
            1
            + torch.arange(
                self.n_states - 1,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )
        )

        # NB initialise at high precision
        state_frac_std = total_exp_read_depth * (
            1
            + torch.arange(
                self.n_states - 1,
                device=self.device,
                requires_grad=False,
                dtype=torch.float32,
            )
        )

        state_frac_std = state_frac_std.sqrt() / state_frac_std

        return state_means.log(), state_frac_std.log()

    @property
    def parameters_dict(self):
        """
        Dict with named torch parameters.
        """
        return {
            "state_log_means": self.state_log_means.clone(),
            "state_log_frac_std": self.state_log_frac_std.clone(),
        }

    @no_grad
    def mask_grad(self):
        return self

    @no_grad
    def sample(self, states):
        # NB expect bookends
        assert states[0] == states[-1] == 0

        states = -1 + states[1:-1].clone()

        # NB one hot requires int64
        hot_states = F.one_hot(
            states.to(torch.int64), num_classes=self.n_states - 1
        ).float()

        means = (hot_states @ self.state_log_means.clone()).exp()
        frac_std = (hot_states @ self.state_log_frac_std.clone()).exp()

        var = means + (means * frac_std) ** 2.0

        # NB prob. success
        ps = means / var

        # NB number successes
        rs = means * means / (var - means)

        # NB < number trials >
        ts = rs / ps

        # NB real-valued Polya distribution.
        fs = ts - rs

        # TODO inefficient
        result = torch.stack(
            [
                NegativeBinomial(
                    total_count=fs[ii],
                    probs=ps[ii],
                ).sample()
                for ii in states
            ],
            dim=0,
        )

        bookend = self.bookend.sample()

        return torch.cat((bookend, result, bookend))

    def forward(self, obs, states=None):
        obs = torch.atleast_1d(obs.clone())

        # NB expect bookends
        assert obs[0] == obs[-1] == 0

        obs = obs[1:-1]

        if states is None:
            states = torch.arange(
                self.n_states - 1, dtype=torch.int32, device=self.device
            )
        else:
            assert states[0] == states[-1] == 0
            states = -1 + states[1:-1].clone()

        # NB forward is to be used for training only.
        hot_states = F.one_hot(
            states.to(torch.int64), num_classes=self.n_states - 1
        ).float()

        means = (hot_states @ self.state_log_means).exp()
        frac_std = (hot_states @ self.state_log_frac_std).exp()
        var = means + (means * frac_std) ** 2.0

        # NB prob. success
        ps = means / var

        # NB number successes
        rs = means * means / (var - means)

        # NB < number trials >
        ts = rs / ps

        # NB < number fails >
        fs = ts - rs

        if len(states) != len(obs):
            result = [
                NegativeBinomial(total_count=fs[ii], probs=ps[ii]).log_prob(obs)
                for ii in states
            ]
        else:
            result = [
                NegativeBinomial(total_count=fs[ii], probs=ps[ii]).log_prob(obs[ii])
                for ii in states
            ]

        return torch.stack(result, dim=0)

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

    def torch_training(self, states, obvs, optimizer=None, n_epochs=300, lr=1.e-1):
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

            loss = -self.forward(obvs, states).sum()
            loss.backward()
            
            self = self.mask_grad()
            """
            # TODO HACK
            with torch.no_grad():
                self.state_log_means -= lr * self.state_log_means.grad
                self.state_log_frac_std -= lr * self.state_log_frac_std.grad
            """
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(
                    f"Torch training epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f} with log means: {self.state_log_means.detach().cpu().numpy()}"
                )

        # NB evaluation, not training, mode.
        self.eval()
        self.finalize_training()

        loss = -self.forward(obvs, states).sum()

        for key in self.parameters_dict:
            logger.info(
                f"Found optimised parameters for {key} to be:\n{self.parameters_dict[key]}"
            )

        logger.info(
            f"After training with torch for {n_epochs} epochs, found the log evidence to be {loss:.4f} by the forward method."
        )

        return n_epochs, loss

    def finalize_training(self):
        # NB no finalization steps required
        return self

    def to_device(self, device):
        self.device = device
        self.baseline_exp = self.baseline_exp.to(device)
        self.spots_total_transcripts = self.spots_total_transcripts.to(device)
        self.state_log_means = self.state_log_means.to(device)
        self.state_log_frac_std = self.state_log_frac_std.to(device)
        self.state_grad_mask = self.state_grad_mask.to(device)

        return self

    def validate(self):
        logger.info(
            f"Negative Binomial emission with log means and log frac. std:\n{self.state_log_means}\n{self.state_log_frac_std}\n"
        )


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
    n_seq, K, N, G, device = 250, 8, 100, 25, "cpu"
    total_exp_read_depth = 25

    # TODO
    spots_total_transcripts = torch.randn(N, device=device)
    baseline_exp = torch.randn(G, device=device)

    genEmitter = TranscriptEmission(
        K,
        spots_total_transcripts,
        baseline_exp,
        device=device,
        name="genEmitter",
        total_exp_read_depth=100,
    )

    states = torch.randint(low=1, high=K + 1, size=(n_seq,), device=device)
    states = bookend_sequence(states, device=device)

    obvs = genEmitter.sample(states)

    logger.info(f"Generated hidden sequence:\n{states}")
    logger.info(f"Generated observed sequence:\n{obvs}")

    modelEmitter = TranscriptEmission(
        K, spots_total_transcripts, baseline_exp, device=device, name="modelEmitter",
    )
    
    # result = modelEmitter.forward(obvs, None) 
    # result = modelEmitter.forward(obvs, states)

    if train:
        modelEmitter.torch_training(states, obvs)

    logger.info("Done.")