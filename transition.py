import torch
import logging
import numpy as np
from utils import get_device, get_log_probs_precision

LOG_PROBS_PRECISION = get_log_probs_precision()

logger = logging.getLogger(__name__)

class MarkovTransition(torch.nn.Module):
    def __init__(
        self,
        n_states,
        log_trans=None,
        device=None,
        diag_rate=0.95,
        log_probs_precision=LOG_PROBS_PRECISION,
        fixed=True,
    ):
        super(MarkovTransition, self).__init__()

        self.device = get_device() if device is None else device

        # NB we assume a bookend state to transition in/out.
        self.n_states = 1 + n_states

        self.log_probs_precision = log_probs_precision

        if log_trans is None:
            self.log_trans = MarkovTransition.init_diag_transitions(
                self.n_states, diag_rate=diag_rate, device=self.device
            )
        else:
            assert isinstance(
                log_trans, torch.Tensor
            ), "log_trans must be a torch.tensor instance"

            assert log_trans.shape == (
                self.n_states,
                self.n_states,
            ), "log_trans must be defined for the bookend state."

            # self.log_trans = log_trans
            self.log_trans = self.normalize_transitions(
                log_trans, log_probs_precision=log_probs_precision
            )

        if not fixed:
            self.log_trans = torch.nn.Parameter(self.log_trans)

        # NB transitions to/from bookend state should not be trained.
        self.trans_grad_mask = torch.ones(
            (self.n_states, self.n_states),
            requires_grad=False,
            device=self.device,
        )
        
        self.trans_grad_mask[0, :] = 0
        self.trans_grad_mask[:, 0] = 0

    def log_transition(self, state, second_state):
        """
        Getter for transition matrix with broadcasting.
        """
        if state is None:
            if second_state is None:
                return self.log_trans
            else:
                return self.log_trans[:, second_state]
        elif second_state is None:
            return self.log_trans[state, :]
        else:
            return self.log_trans[state, second_state]

    @classmethod
    def init_diag_transitions(
        cls,
        n_states,
        diag_rate=0.95,
        device=None,
        log_probs_precision=LOG_PROBS_PRECISION,
    ):
        """
        Defaults to a diagonal transition matrix.
        """
        if device is None:
            device = get_device()

        # NB (nstates - 2) to account for diagonal and bookend.
        off_diag_rate = (1.0 - diag_rate) / (n_states - 2.0)

        logger.info(
            f"Initialising MarkovTransition with (diag_rate, off_diag_rate) = ({diag_rate:.4f}, {off_diag_rate:.4f})"
        )

        eye = torch.eye(n_states, device=device)

        log_trans = diag_rate * eye.clone()
        log_trans += off_diag_rate * (
            torch.ones(n_states, n_states, device=device) - eye.clone()
        )

        log_trans[0, 0] = torch.tensor(log_probs_precision, device=device).exp()
        log_trans[0, 1:] = 1.0 / (n_states - 1.0)

        log_trans = log_trans.log()

        log_trans = MarkovTransition.normalize_transitions(
            log_trans, log_probs_precision=log_probs_precision
        )

        return log_trans

    @classmethod
    def normalize_transitions(cls, log_trans, log_probs_precision=LOG_PROBS_PRECISION):
        # NB i) log_probs to transition to the bookend is small (at machine log_probs_precision).
        #    ii)  exact value is irrelevant, beyond shifting log probs. by a constant
        #    iii) but should be equal amongst states - includes bookend to bookend, with log probs ~ -inf.
        #    iv) practical requirement on log_probs_precision is generating samples.
        #    v) we freeze the bookend parameters during training with a mask.
        log_trans[:, 0] = log_probs_precision
        log_trans[0, :] = log_probs_precision

        # NB see https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
        log_trans = log_trans.log_softmax(dim=1)

        # NB reconstruct the first row disrupted by softmax.
        log_trans[0, 0] = torch.tensor(log_probs_precision, device=device)
        log_trans[0, 1:] = torch.tensor(
            (1.0 - torch.tensor([log_probs_precision]).exp()) / len(log_trans[0, 1:]), device=device
        ).log()

        return log_trans

    def validate(self):
        logger.info(
            f"Transition log probs matrix for MarkovTransition:\n{self.log_trans}\n"
        )

    def forward(self, log_vs):
        return log_vs.unsqueeze(-1) + self.log_trans.log_softmax(dim=1)

    def mask_grad(self):
        # NB we do not optimise transitions to/from bookend state.                                                                                                  
        self.log_trans.grad *= self.trans_grad_mask
        return self
        
        
    def to_device(self, device):
        self.log_trans.data = self.log_trans.data.to(device)
        self.trans_grad_mask = self.trans_grad_mask.to(device)

        return self

    @property
    def parameters_dict(self):
        """Dict with named torch parameters."""
        return {"log_trans": self.log_trans}
