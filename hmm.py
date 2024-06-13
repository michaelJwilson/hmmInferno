import logging
import logging.config
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from transition import MarkovTransition
from emission import CategoricalEmission
from transcripts import TranscriptEmission
from utils import (
    bookend_sequence,
    get_device,
    no_grad,
    get_log_probs_precision,
    get_log_probs_precision,
    set_scalars
)
# from rich.logging import RichHandler

LOG_PROBS_PRECISION = get_log_probs_precision()

formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(message)s"
)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
# handler = RichHandler(rich_tracebacks=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class MarkovTransition(torch.nn.Module):
    def __init__(
        self,
        n_states,
        log_trans=None,
        device=None,
        diag_rate=0.95,
        log_probs_precision=LOG_PROBS_PRECISION,
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

        Returns torch.Tensor([..])
        """
        if state is None:
            if second_state is None:
                return self.log_trans
            else:
                return self.log_trans[:, second_state]
        elif second_state is None:
            return self.log_trans[state, :]
        else:
            return set_scalars(self.log_trans[state, second_state], device=self.device)

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
            (1.0 - np.exp(log_probs_precision)) / len(log_trans[0, 1:]), device=device
        ).log()

        return log_trans

    def validate(self):
        logger.info(
            f"Transition log probs matrix for MarkovTransition:\n{self.log_trans}\n"
        )

    def forward(self, log_vs):
        return log_vs.unsqueeze(-1) + self.log_trans.log_softmax(dim=1)

    def to_device(self, device):
        self.log_trans.data = self.log_trans.data.to(device)
        self.trans_grad_mask = self.trans_grad_mask.to(device)

        return self

    @property
    def parameters_dict(self):
        """Dict with named torch parameters."""
        return {"log_trans": self.log_trans}


class HMM(torch.nn.Module):
    def __init__(
        self,
        n_states,
        emission_model,
        transition_model=None,
        device=None,
        log_probs_precision=LOG_PROBS_PRECISION,
        name="HMM",
    ):
        super(HMM, self).__init__()

        self.name = name
        self.device = get_device() if device is None else device

        logger.info(
            f"Creating {name} with {n_states} hidden states, bookended by 0 at both ends."
        )

        # NB we assume a bookend state to transition in/out.
        self.n_states = 1 + n_states

        assert (
            self.n_states == emission_model.n_states
        ), f"State mismatch between HMM and Emission, found {self.n_states} but expected {emission_model.n_states}."

        # NB We start (and end) in the bookend state.  No gradient required.
        self.log_pi = log_probs_precision * torch.ones(
            self.n_states, requires_grad=False, device=self.device
        )
        self.log_pi[0] = 0.0

        if transition_model is None:
            transition_model = MarkovTransition(n_states)

        self.transition_model = transition_model
        self.emission_model = emission_model

        self.to_device(device)
        self.validate()

    def log_transition(self, state, second_state):
        # NB should be guranteed to return a torch.Tensor with length > 1
        return self.transition_model.log_transition(state, second_state)

    def log_emission(self, state, obs):
        # NB should be guranteed to return a torch.Tensor with length > 1
        return self.emission_model.log_emission(state, obs)

    def sample_hidden(self, n_seq, bookend=True):
        last_state = set_scalars(0, device=self.device)
        sequence = [last_state.item()]

        # TODO avoid bookend intra? samples.
        for _ in range(n_seq):
            probs = self.log_transition(last_state, None).clone().exp()
            probs[0, 0] = 0.0

            # NB sample a new state
            state = Categorical(probs).sample()
            sequence.append(state.item())

            last_state = state

        sequence = torch.tensor(sequence, dtype=torch.int32, device=self.device)

        if bookend:
            sequence = bookend_sequence(sequence, device=self.device)

        return sequence[1:]

    def sample(self, n_seq, hidden=None, bookend=True):
        """
        TODO assumes categorical - move to emission.
        """
        if hidden is None:
            hidden = sample_hidden(self, n_seq, bookend=bookend)

        if bookend:
            assert hidden[0].item() == hidden[-1].item() == 0

        return self.emission_model.sample(hidden)

    def log_like(self, obvs, states, bookend=False):
        """
        Eqn. (3.6) of Durbin.
        """
        # NB we must start in a bookend state with a bookend obs.
        assert states[0] == states[-1] == 0
        assert obvs[0] == obvs[-1] == 0

        # NB we start in the bookend state with unit probability.
        log_like = self.log_pi[0].clone().unsqueeze(0)
        last_state = states[0].clone()
        
        for obs, state in zip(obvs[1:-1], states[1:-1]):
            log_like += self.log_transition(last_state, state)
            log_like += self.log_emission(state, obs)
                
            last_state = state
            
        return log_like

    def viterbi(self, obvs, traced=False):
        """
        Decoding the obs. sequence to a hidden sequence with max. likelihood.
        NB no need to calculate the traceback when training.

        See value algebra on pg. 56 of Durbin.
        """
        # NB we must start in a bookend state with a bookend obs.
        assert obvs[0] == obvs[-1] == 0

        if traced:
            trace_table = torch.zeros(
                len(obvs), self.n_states, dtype=torch.int32, device=self.device
            )
            trace_table[0, 0] = 1

        # NB relies on transition from 0 to any state to be equal.
        log_vs = self.log_pi.clone()

        for ii, obs in enumerate(obvs[1:]):            
            interim = self.transition_model.forward(log_vs)

            log_vs, max_states = torch.max(interim, dim=0)
            log_vs += self.log_emission(max_states, obs)

            if traced:
                trace_table[1 + ii, :] = max_states

        # NB finally forced transition into the book end state.  Relies on transition
        #    from any state to 0 to be equal.
        log_vs += self.log_transition(None, 0)
        log_joint_prob, penultimate_state = torch.max(log_vs, dim=0)

        if traced:
            return log_joint_prob, penultimate_state, trace_table
        else:
            return log_joint_prob, penultimate_state

    @staticmethod
    def viterbi_traceback(trace_table, penultimate_state):
        """
        Viterbi traceback.

        See pointer algebra on pg. 56 of Durbin.
        """
        reversed_trace_table = torch.flip(trace_table.clone(), dims=[0])
        decoded_states = torch.zeros(len(trace_table) - 2, dtype=torch.int32)

        last_state = penultimate_state

        for ii, row in enumerate(reversed_trace_table[2:]):
            idx = len(decoded_states) - 1 - ii
            decoded_states[idx] = last_state

            last_state = row[last_state]

        return decoded_states

    def log_forward_scan(self, obvs, as_prior=False):
        """
        Log evidence (marginalised over latent) by the forward method.

        See termination step after Eqn. (3.11) of Durbin.
        """
        assert obvs[0] == obvs[-1] == 0
        
        log_fs = self.log_pi.clone()

        states = torch.arange(self.n_states, dtype=torch.int32, device=self.device)
        
        for ii, obs in enumerate(obvs[1:-1]):
            interim = self.transition_model.forward(log_fs)
            log_fs = torch.logsumexp(interim, dim=0)

            if not as_prior:
                log_fs += self.emission_model.forward(obs, states=states).squeeze(-1)
            
        # NB final transition into the book end state; note coefficient is not trained.
        log_fs += self.log_transition(None, 0)

        return torch.logsumexp(log_fs, dim=0)

    @no_grad
    def log_forward(self, obvs):
        """
        Log evidence (marginalised over latent) by the forward method,
        returning forward array.
        See termination step after Eqn. (3.11) of Durbin.
        """
        # NB we must start in a bookend state with a bookend obs.
        assert obvs[0] == obvs[-1] == 0

        log_fs = torch.zeros(len(obvs), self.n_states, device=self.device)
        log_fs[0] = self.log_pi.clone()

        for ii, obv in enumerate(obvs[1:-1]):
            interim = self.transition_model.forward(log_fs[ii])

            # TODO HACK squeeze
            log_fs[ii + 1] = self.emission_model.forward(obv).squeeze(-1)
            log_fs[ii + 1] += torch.logsumexp(interim, dim=0)
            
        # NB final transition into the book end state.
        log_fs[-1] = log_fs[-2] + self.log_transition(None, 0)

        return torch.logsumexp(log_fs[-1], dim=0), log_fs

    @no_grad
    def log_backward_scan(self, obvs):
        """
        Log evidence (marginalised over latent) by the forward method,

        See termination step before Eqn. (3.14) of Durbin.
        """
        assert obvs[0] == obvs[-1] == 0

        rev_obvs = torch.flip(obvs, dims=[0])
        log_bs = self.log_transition(None, 0).clone()
        
        # NB no bookend states.
        for ii, obv in enumerate(rev_obvs[1:-2]):            
            interim = log_bs.unsqueeze(0) + self.log_transition(None, None)            
            interim += self.emission_model.forward(obv)

            log_bs = torch.logsumexp(interim, dim=1)
            
        log_bs += self.log_transition(0, None)

        # TODO HACK squeeze.
        log_bs += self.log_emission(None, rev_obvs[-2]).squeeze(-1)
        
        log_evidence = torch.logsumexp(log_bs, dim=0)

        return log_evidence

    @no_grad
    def log_backward(self, obvs):
        """
        Log evidence (marginalised over latent) by the forward method,
        returning backward array.

        See termination step before Eqn. (3.14) of Durbin.
        """
        assert obvs[0] == obvs[-1] == 0

        rev_obvs = torch.flip(obvs, dims=[0])

        log_bs = torch.zeros(len(obvs) - 1, self.n_states, device=self.device)
        log_bs[0] = self.log_transition(None, 0).clone()

        for ii, obv in enumerate(rev_obvs[1:-2]):
            interim = log_bs[ii, :].unsqueeze(0) + self.log_transition(None, None)

            # TODO HACK squeeze
            interim += self.emission_model.forward(obv) # .unsqueeze(0)

            log_bs[ii + 1] = torch.logsumexp(interim, dim=1)

        # TODO HACK squeeze
        log_bs[-1] = (
            log_bs[-2]
            + self.log_transition(0, None)
            + self.log_emission(None, rev_obvs[-2]).squeeze(-1)
        )

        log_evidence = torch.logsumexp(log_bs[-1], dim=0)

        return log_evidence, torch.flip(log_bs, dims=[0])

    @no_grad
    def log_state_posterior(self, obvs):
        """
        Eqn. (3.14) of Durbin.
        """
        assert obvs[0] == obvs[-1] == 0

        log_evidence_forward, log_forward_array = self.log_forward(obvs)
        _, log_backward_array = self.log_backward(obvs)

        log_state_posterior = (
            log_forward_array[1:-1] + log_backward_array[1:] - log_evidence_forward
        )

        return log_state_posterior

    @no_grad
    def max_posterior_decoding(self, obvs):
        """
        Eqn. (3.15) of Durbin.
        """
        assert obvs[0] == obvs[-1] == 0

        # NB no bookend states
        log_state_posterior = self.log_state_posterior(obvs)

        return torch.argmax(self.log_state_posterior(obvs), dim=1).to(torch.int32)

    @no_grad
    def log_transition_posterior(self, obvs):
        """
        Eqn. (3.19) of Durbin.
        """
        assert obvs[0] == obvs[-1] == 0

        log_evidence_forward, log_forward_array = self.log_forward(obvs)
        log_evidence_backward, log_backward_array = self.log_backward(obvs)

        # NB we need x_(i+1)
        obvs = obvs.clone()[2:-1]

        # NB limit to observed states with i < L; i.e. no transition to bookend.
        log_forward_array = log_forward_array[1:-2].unsqueeze(2)
        log_backward_array = log_backward_array[2:].unsqueeze(1)

        # NB indexed by (i == time, k, l)
        log_transition_posterior = log_forward_array + log_backward_array
        log_transition_posterior += self.log_transition(None, None).unsqueeze(0)
        log_transition_posterior += self.log_emission(None, obvs).T.unsqueeze(2)
        log_transition_posterior -= log_evidence_forward

        return log_transition_posterior

    @no_grad
    def exp_transition_counts(
        self, obvs, log_transition_posterior=None, pseudo_counts=None
    ):
        assert obvs[0] == obvs[-1] == 0

        if log_transition_posterior is None:
            # NB only defined for observable states, not bookends.
            log_transition_posterior = self.log_transition_posterior(obvs)

        exp_transition_counts = torch.logsumexp(log_transition_posterior, dim=0).exp()

        if pseudo_counts is not None:
            exp_transition_counts += pseudo_counts

        # NB TODO exp. for bookend state transition?
        return exp_transition_counts

    # TODO emission model specific, i.e. assumes Categorical.  Move to Emission class.
    def exp_emission_counts(self, obvs, pseudo_counts=None):
        """ """
        assert obvs[0] == obvs[-1] == 0

        if pseudo_counts is None:
            exp_emission_counts = torch.zeros(
                (self.n_states, self.emission_model.n_obvs), device=self.device
            )
        else:
            # TODO assert shape etc.
            exp_emission_counts = pseduo_counts

        log_evidence_forward, log_forward_array = self.log_forward(obvs)
        log_backward_evidence, log_backward_array = self.log_backward(obvs)

        # NB no bookend states
        obvs, log_forward_array, log_backward_array = (
            obvs[1:-1],
            log_forward_array[1:-1],
            log_backward_array[1:],
        )

        interim = log_forward_array + log_backward_array - log_evidence_forward

        for obv in torch.unique(obvs):
            mask = obvs == obv
            exp_emission_counts[:, obv] += torch.logsumexp(interim[mask], dim=0).exp()

        return exp_emission_counts

    def baum_welch_update(
        self, obvs, transition_pseudo_counts=None, emission_pseudo_counts=None
    ):
        """ """
        exp_transition_counts = self.exp_transition_counts(
            obvs, transition_pseudo_counts
        )
        exp_transition_counts /= torch.sum(exp_transition_counts, dim=1).unsqueeze(-1)

        # TODO emission model specific, i.e. assumes Categorical.  Move to Emission class.
        exp_emission_counts = self.exp_emission_counts(obvs, emission_pseudo_counts)
        exp_emission_counts /= torch.sum(exp_emission_counts, dim=1).unsqueeze(-1)

        log_trans = exp_transition_counts.log()
        log_trans[0, :] = self.log_transition(0, None)

        log_emit = exp_emission_counts.log()
        log_emit[0, :] = self.log_emission(0, None)

        return log_trans, log_emit

    def baum_welch_training(self):
        raise NotImplementedError()

    def torch_training(self, obvs, optimizer=None, n_epochs=10, lr=1.0e-2):
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

            loss = -self.log_forward_scan(obvs)
            loss.backward()

            self.transition_model = self.transition_model.mask_grad()            
            self.emission_model = self.emission_model.mask_grad()

            optimizer.step()

            if epoch % 10 == 0:
                logger.info(
                    f"Torch training epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}"
                )

        # NB evaluation, not training, mode.
        self.eval()

        self.transition_model.log_trans.data = MarkovTransition.normalize_transitions(
            self.transition_model.log_trans.data
        )
        
        self.emission_model.finalize_training()
        
        for key in self.parameters_dict:
            logger.info(
                f"Found optimised parameters for {key} to be:\n{self.parameters_dict[key]}"
            )

        logger.info(
            f"After training with torch for {n_epochs} epochs, found the log evidence to be {self.log_forward_scan(obvs):.4f} by the forward method."
        )

        return n_epochs, self.log_forward_scan(obvs)

    def validate(self):
        """
        # NB log probs.
        assert torch.allclose(
            self.log_trans.logsumexp(dim=1),
            torch.zeros(self.log_trans.size(0)).to(device),
            rtol=1e-05,
            atol=1e-06,
        )
        """
        logger.info(
            f"Initialised HMM starting in bookend state with log probability matrix:\n{self.log_pi}"
        )

        self.transition_model.validate()
        self.emission_model.validate()

    @property
    def parameters_dict(self):
        """Dict with named torch parameters."""
        # TODO name clash.
        return (
            self.transition_model.parameters_dict | self.emission_model.parameters_dict
        )

    def to_device(self, device):
        self.log_pi = self.log_pi.to(device)
        self.transition_model = self.transition_model.to_device(device)
        self.emission_model = self.emission_model.to_device(device)

        return self

    
if __name__ == "__main__":
    # TODO set seed for cuda / mps
    torch.manual_seed(314)

    # TODO BUG? must be even?
    n_states, n_seq, diag, device, train = 4, 2_000, True, "cpu", True
    n_spots, n_segments = 100, 100
    
    start = time.time()

    transition_model = MarkovTransition(n_states=n_states, diag_rate=0.5, device=device)
    transition_model.validate()

    # TODO
    spots_total_transcripts = torch.randn(n_spots, device=device)
    baseline_exp = torch.randn(n_segments, device=device)

    # casino = Casino(device=device)
    # categorical = CategoricalEmission(n_states=n_states, diag=diag, n_obvs=n_states, device=device)
    genTranscripts = TranscriptEmission(
        n_states, spots_total_transcripts, baseline_exp, device=device, name="genTranscripts"
    )
    
    modelTranscripts = TranscriptEmission(
        n_states, spots_total_transcripts, baseline_exp, device=device, name = "modelTranscripts"
    )

    # NB (n_states * n_obvs) action space.
    genHMM = HMM(
        n_states=n_states,
        emission_model=genTranscripts,
        transition_model=transition_model,
        device=device,
        name="genHMM",
    )

    # NB hidden states matched to observed time steps.
    hidden_states = genHMM.sample_hidden(n_seq, bookend=True)
    obvs = genHMM.sample(n_seq, hidden_states)

    logger.info(f"Generated hidden sequence:\n{hidden_states}")
    logger.info(f"Generated observed sequence:\n{obvs}")

    # NB defaults to a diagonal transition matrix.
    modelHMM = HMM(
        n_states=n_states,
        emission_model=modelTranscripts,
        transition_model=None,
        device=device,
        name="modelHMM",
    )

    if train:
        torch_n_epochs, torch_log_evidence_forward = modelHMM.torch_training(obvs)

    log_like = modelHMM.log_like(obvs, hidden_states).item()

    logger.info(f"Found a log likelihood= {log_like:.4f} for generated hidden states")

    # NB P(x, pi) with tracing for most probably state sequence
    log_joint_prob, penultimate_state, trace_table = modelHMM.viterbi(obvs, traced=True)

    logger.info(
        f"Found a joint probability P(x, pi)={log_joint_prob:.4f} with trace:\n{trace_table}"
    )

    # NB Most probable state sequence
    viterbi_decoded_states = modelHMM.viterbi_traceback(trace_table, penultimate_state)

    logger.info(
        f"Found the penultimate state to be {penultimate_state} with a Viterbi decoding of:\n{viterbi_decoded_states}"
    )

    # NB P(x) marginalised over hidden states by forward & backward scan - no array traceback.
    log_evidence_forward_scan = modelHMM.log_forward_scan(obvs)
    log_evidence_backward_scan = modelHMM.log_backward_scan(obvs)
    
    # NB P(x) marginalised over hidden states by forward & backward method - array traceback.
    log_evidence_forward, log_forward_array = modelHMM.log_forward(obvs)
    log_evidence_backward, log_backward_array = modelHMM.log_backward(obvs)

    logger.info(
        f"Found the evidence to be {log_evidence_forward.item():.4f}, {log_evidence_forward_scan.item():.4f} by the forward method and scan."
    )

    logger.info(
        f"Found the evidence to be {log_evidence_backward.item():.4f}, {log_evidence_backward_scan.item():.4f} by the backward method and scan."
    )

    assert torch.allclose(
        log_evidence_forward_scan, log_evidence_forward
    ), f"Inconsistent log evidence by forward scanning and forward method: {log_evidence_forward_scan:.4f} and {log_evidence_forward:.4f}"

    # TODO tolerance
    assert torch.allclose(
        log_evidence_forward_scan,
        log_evidence_backward_scan,
        rtol=1.0e-2,
    ), f"Inconsistent log evidence by forward and backward scan: {log_evidence_forward_scan:.4f} and {log_evidence_backward_scan:.4f}"

    # TODO tolerance
    assert torch.allclose(
        log_evidence_forward,
        log_evidence_backward,
        atol=1.0,
    ), f"Inconsistent log evidence by forward and backward methods: {log_evidence_forward:.4f} and {log_evidence_backward:.4f}"

    logger.info(f"Found the log forward array to be:\n{log_forward_array}")
    logger.info(f"Found the log backward array to be:\n{log_backward_array}")

    # NB P(pi_i = k | x) for all i.
    log_state_posteriors = modelHMM.log_state_posterior(obvs)

    logger.info(f"Found the state posteriors to be:\n{log_state_posteriors}")

    assert len(log_state_posteriors) == n_seq

    # NB argmax_k P(pi_i = k | x) for all i.
    posterior_decoded_states = modelHMM.max_posterior_decoding(obvs)
    posterior_decoded_states = bookend_sequence(posterior_decoded_states)

    logger.info(
        f"Found a state decoding (max. disjoint posterior):\n{posterior_decoded_states}"
    )

    # NB satisfying! in the case of genHMM != modelHMM, this matches? because .. diag emission?
    assert torch.allclose(
        hidden_states, posterior_decoded_states
    ), f"State decoding and truth inconsistent:\n{hidden_states}\n{posterior_decoded_states}."


    log_transition_posteriors = modelHMM.log_transition_posterior(obvs)

    # NB the last transition is (i, i + 1) == (L - 1, L).
    assert len(log_transition_posteriors) == (n_seq - 1)

    logger.info(
        f"Found the log transition posteriors to be:\n{log_transition_posteriors}"
    )

    exp_transition_counts = modelHMM.exp_transition_counts(obvs)

    logger.info(
        f"Found the exp transition counts (no bookends) to be:\n{exp_transition_counts}"
    )

    # NB specific to Categorical emission
    exp_emission_counts = modelHMM.exp_emission_counts(obvs)

    logger.info(
        f"Found the exp emission counts (no bookends) to be:\n{exp_emission_counts}"
    )

    baum_welch_transitions, baum_welch_emissions = modelHMM.baum_welch_update(obvs)

    logger.info(
        f"Found the transitions Baum-Welch update to be:\n{baum_welch_transitions}"
    )

    logger.info(f"Found the emissions Baum-Welch update to be:\n{baum_welch_emissions}")

    logger.info(f"Done (in {time.time() - start:.1f}s).\n\n")
