import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from torch.optim import Adam, AdamW

from dist import BookendDist
from emission import BookendDist
from utils import (bookend_sequence, get_bookend_token, get_device,
                   get_log_probs_precision, no_grad, set_scalars)

logger = logging.getLogger(__name__)

LOG_PROBS_PRECISION = get_log_probs_precision()
BOOKEND_TOKEN = get_bookend_token()

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

        state_log_means, state_log_frac_std = self.init_emission(
            total_exp_read_depth=total_exp_read_depth
        )

        self.state_log_means = torch.nn.Parameter(state_log_means.requires_grad_(True))
        self.state_log_frac_std = torch.nn.Parameter(
            state_log_frac_std.requires_grad_(True)
        )

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
        logger.warning(f"Assuming a total exp. read depth of {total_exp_read_depth}.")

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
        state_frac_std = (
            2
            * total_exp_read_depth
            * (
                1
                + torch.arange(
                    self.n_states - 1,
                    device=self.device,
                    requires_grad=False,
                    dtype=torch.float32,
                )
            )
        )

        state_frac_std = state_frac_std.sqrt() / state_frac_std

        return state_means.log(), state_frac_std.log()

    @no_grad
    def get_parameters_dict(self):
        """
        Dict with named torch parameters.
        """
        # TODO .detach()
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
        assert states[0] == states[-1] == BOOKEND_TOKEN

        observable_states = -1 + states[1:-1].clone()

        # NB one hot requires int64
        hot_states = F.one_hot(
            observable_states.to(torch.int64), num_classes=(self.n_states - 1)
        ).float()

        # TODO remove clone()
        means = (hot_states @ self.state_log_means.clone()).exp()
        frac_std = (hot_states @ self.state_log_frac_std.clone()).exp()

        # logger.debug(f"Inferred state emission means:\n{means}")

        var = means + (means * frac_std) ** 2.0

        # logger.debug(f"Inferred state emission var:\n{var}")

        # NB prob. success
        ps = means / var

        logger.debug(f"Inferred prob. of success for {self.name}:\n{ps}")

        # NB number successes
        rs = means * means / (var - means)

        # logger.debug(f"Inferred number of successes:\n{rs}")

        # NB < number trials >
        ts = rs / ps

        # logger.debug(f"Inferred number of trials:\n{ts}")

        # NB real-valued Polya distribution.
        fs = ts - rs

        # logger.debug(f"Inferred number of failures:\n{ts}")

        # TODO inefficient
        result = torch.stack(
            [
                NegativeBinomial(
                    total_count=fs[ii],
                    probs=ps[ii],
                ).sample()
                for ii in observable_states
            ],
            dim=0,
        )

        bookend = self.bookend.sample()

        # TODO requires_grad(False)
        return torch.cat((bookend, result, bookend))

    def forward(self, obs, states=None):
        # TODO drop clone?
        obs = torch.atleast_1d(obs.clone())

        # NB expect bookends
        assert obs[0] == obs[-1] == BOOKEND_TOKEN

        # TODO ugh
        broadcast = False

        if states is None:
            observable_states = torch.arange(
                (self.n_states - 1), dtype=torch.int32, device=self.device
            )

            broadcast = True
        else:
            assert states[0] == states[-1] == BOOKEND_TOKEN
            observable_states = -1 + states[1:-1].clone()

        # NB forward is to be used for training only.
        hot_states = F.one_hot(
            observable_states.to(torch.int64), num_classes=(self.n_states - 1)
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

        # NB broadcast to all potential states, including bookend.
        if broadcast:
            result = [
                NegativeBinomial(total_count=fs[ii], probs=ps[ii]).log_prob(obs)
                for ii in observable_states
            ]

            result = torch.stack(result, dim=0)
            result[:, 0] = LOG_PROBS_PRECISION
            result[:, -1] = LOG_PROBS_PRECISION

            bookend_row = LOG_PROBS_PRECISION * torch.ones(len(obs), device=self.device)
            bookend_row[0] = 0.0
            bookend_row[-1] = 0.0

            return torch.cat((bookend_row.unsqueeze(0), result), dim=0)

        else:
            result = [
                NegativeBinomial(total_count=ff, probs=pp).log_prob(oo)
                for ff, pp, ss, oo in zip(fs, ps, observable_states, obs[1:-1])
            ]

            # NB log_prob for all obs. with the as defined hidden state in each case.
            result = torch.stack(result, dim=0)
            return bookend_sequence(
                result, device=self.device, dtype=torch.float32, token=0.0
            )

    def log_emission(self, state, obs):
        """
        Getter for log_em with broadcasting
        """
        # TODO
        if state is None:
            if obs is None:
                raise NotImplementedError()
            else:
                return self.forward(obs, state)
        elif obs is None:
            raise NotImplementedError()
        else:
            return self.forward(obs, state)

    def torch_training(self, states, obvs, optimizer=None, n_epochs=50, lr=1.0e-1):
        torch.autograd.set_detect_anomaly(True)
        
        # NB weight_decay=1.0e-5
        optimizer = AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[350, 700], gamma=0.1
        )

        # NB set model to training mode - important for batch normalization & dropout -
        #    unnecessaary here, but best practice.
        self.train()

        for key in self.get_parameters_dict():
            logger.info(
                f"Ready to train {key} parameter with torch, initialised to:\n{self.get_parameters_dict()[key]}"
            )

        # TODO weight scheduler.
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            forward = self.forward(obvs, states)

            loss = -forward.sum()
            loss.backward()

            optimizer.step()
            scheduler.step()

            if epoch % 10 == 0:
                logger.info(
                    f"Torch training epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f} for log means={self.state_log_means.detach().cpu().numpy()} and log frac stds={self.state_log_frac_std.detach().cpu().numpy()}"
                )

        # NB evaluation, not training, mode.
        self.eval()
        self.finalize_training()

        loss = -self.forward(obvs, states).sum()

        for key in self.get_parameters_dict():
            logger.info(
                f"Found optimised parameters for {key} to be:\n{self.get_parameters_dict()[key]}"
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
    n_seq, K, N, G, device = 600, 4, 100, 25, "cpu"
    total_exp_read_depth = 25

    # TODO
    assert n_seq not in [K, K + 1]

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
        K,
        spots_total_transcripts,
        baseline_exp,
        device=device,
        name="modelEmitter",
    )

    # samples = modelEmitter.sample(states)
    # result = modelEmitter.forward(obvs, None)
    # result = modelEmitter.forward(obvs, states)

    if train:
        modelEmitter.torch_training(states, obvs)

    logger.info("Done.")
