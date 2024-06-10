import logging
import logging.config
import sys

import numpy as np
import torch
from torch.distributions import Categorical, negative_binomial
from torch.optim import Adam

from casino import Casino
from utils import bookend_sequence, get_device

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class CategoricalEmission(torch.nn.Module):
    """
    Categoical emission from a bookend + n_states hidden states, (0, 1, .., n_states),
    to n_obvs emission classes.
    """

    def __init__(self, n_states, n_obvs, device=None):
        super(CategoricalEmission, self).__init__()

        self.device = get_device() if device is None else device

        # NB we assume a bookend state to transition in/out.
        self.n_states = 1 + n_states

        # NB number of possible observable classes + hidden observation emitted by bookend state.
        self.n_obvs = 1 + n_obvs

        logger.info(
            f"Creating CategoricalEmission with {n_states} hidden states & {n_obvs} observed classes on device={self.device}"
        )

        # TODO HACK
        # self.log_em = torch.randn(self.n_states, self.n_obvs)
        self.log_em = (
            torch.eye(self.n_states, self.n_obvs).log().clip(min=-99.0, max=99.0)
        )

        # NB nn.Parameter marks this to be optimised via torch.
        self.log_em = torch.nn.Parameter(self.log_em)

        # TODO HACK
        # NB emit a bookend token from the bookend state.
        self.log_em.data[0, :] = -99.0
        self.log_em.data[0, 0] = 0.0

        # NB rows sums to zero, as prob. to emit to any obs. is unity.
        self.log_em.data = self.log_em.data.log_softmax(dim=1)

    def emission(self, state, obs):
        if state is None:
            return self.log_em[:, obs]
        elif obs is None:
            return self.log_em[state, :]
        else:
            return self.log_em[state, obs]

    def validate(self):
        logger.info(f"Emission log probability matrix:\n{self.log_em}\n")

    def to_device(self, device):
        self.device = device
        self.log_em = self.log_em.to(device)

        return self
        
    @property
    def parameters_dict(self):
        """
        Dict with named torch parameters.
        """
        return {"log_em": self.log_em}


class NegativeBinomial:
    """
    Models the # of failures in a sequence of IID Bernoulli trials
    before a specified (non-random) # of successes, r.

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

    def emission(self, state, obs):
        raise NotImplementedError()


class HMM(torch.nn.Module):
    def __init__(self, n_states, emission_model, log_trans=None, device=None):
        super(HMM, self).__init__()

        self.device = get_device() if device is None else device

        logger.info(
            f"Creating HMM with {n_states} hidden states, bookended by 0 at both ends."
        )

        # NB we assume a bookend state to transition in/out.
        self.n_states = 1 + n_states

        assert (
            self.n_states == emission_model.n_states
        ), f"State mismatch between HMM and Emission, found {self.n_states} but expected {emission_model.n_states}."

        # NB number of possible observable states, as opposed to sequence length.
        self.n_obvs = emission_model.n_obvs

        # NB We start (and end) in the bookend state.  No gradient required.
        self.log_pi = -99.0 * torch.ones(
            self.n_states, requires_grad=False, device=self.device
        )
        self.log_pi[0] = 0.0

        if log_trans is None:
            self_trans = 0.5
            other_trans = (1.0 - self_trans) / (self.n_states)

            eye = torch.eye(self.n_states, device=self.device)

            self.log_trans = self_trans * eye
            self.log_trans += other_trans * (
                torch.ones(self.n_states, self.n_states, device=self.device) - eye
            )
            self.log_trans = self.log_trans.log()

        else:
            assert isinstance(
                log_trans, torch.Tensor
            ), "log_trans must be a torch.tensor instance"

            assert log_trans.shape == (
                self.n_states,
                self.n_states,
            ), "log_trans must include bookend states."

            self.log_trans = log_trans

        # NB the probability for any state to transition to the bookend is small (at machine precision).
        #    the exact value should not matter, beyond shifting log probs. by a constant, but it should
        #    be equal amongst states.
        self.log_trans.data[:, 0] = -99.0

        self.log_trans = torch.nn.Parameter(self.log_trans)

        # NB rows sums to zero, as prob. to transition to any state is unity.
        self.log_trans.data = self.log_trans.data.log_softmax(dim=1)

        # self.log_trans.data[0][1:] = self.log_trans.data[0][1:].log_softmax(dim=0)
        # self.log_trans.data[0,0] = -torch.inf

        self.emission_model = emission_model
        self.to_device(device)
        self.validate()

    def emission(self, state, obs):
        return self.emission_model.emission(state, obs)

    def sample_hidden(self, n_seq, bookend=False):
        last_state = torch.tensor([0], device=self.device)
        sequence = [last_state.item()]

        for _ in range(n_seq):
            probs = self.log_trans[last_state, :].clone().exp()
            probs[0, 0] = 0.0

            # NB sample a new state
            state = Categorical(probs).sample()
            sequence.append(state.item())

            last_state = state

        sequence = torch.tensor(sequence, dtype=torch.int32, device=self.device)

        if bookend:
            sequence = bookend_sequence(sequence, device=self.device)

        return sequence[1:]

    def sample_obvs(self, n_seq, hidden=None, bookend=False):
        """
        TODO assumes categorical - move to emission.
        """
        if hidden is None:
            hidden = sample_hidden(self, n_seq, bookend=bookend)

        if bookend:
            assert hidden[0].item() == hidden[-1].item() == 0
            
        obvs = []

        for state in hidden:
            probs = self.emission(state, None).exp()
            emit = Categorical(probs).sample()

            obvs.append(emit.item())

        return torch.tensor(obvs, dtype=torch.int32, device=self.device)

    def log_like(self, obvs, states, bookend=False):
        """
        Eqn. (3.6) of Durbin.
        """
        # NB we must start in a bookend state with a bookend obs.
        assert states[0] == states[-1] == 0
        assert obvs[0] == obvs[-1] == 0

        # NB we start in the bookend state with unit probability.
        log_like = self.log_pi[0].clone()
        last_state = states[0].clone()

        for obs, state in zip(obvs[1:-1], states[1:-1]):
            log_like += self.log_trans[last_state, state]
            log_like += self.emission(state, obs)

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
            interim = log_vs.unsqueeze(-1) + self.log_trans.clone()

            log_vs, max_states = torch.max(interim, dim=0)
            log_vs += self.emission(max_states, obs)

            if traced:
                trace_table[1 + ii, :] = max_states

        # NB finally forced transition into the book end state.  Relies on transition
        #    from any state to 0 to be equal.
        log_vs += self.log_trans[:, 0]
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

    def log_forward_scan(self, obvs):
        """
        Log evidence (marginalised over latent) by the forward method.

        See termination step after Eqn. (3.11) of Durbin.
        """
        log_fs = self.log_pi.clone()

        for ii, obs in enumerate(obvs):
            log_fs = log_fs.unsqueeze(-1) + self.log_trans.clone()
            log_fs = self.emission(None, obs) + torch.logsumexp(log_fs, dim=0)

        # NB final transition into the book end state.
        return torch.logsumexp(log_fs + self.log_trans[:, 0], dim=0)

    def log_backward_scan(self, obvs):
        """
        Log evidence (marginalised over latent) by the forward method,

        See termination step before Eqn. (3.14) of Durbin.
        """
        log_bs = self.log_trans[:, 0].clone()
        rev_obvs = torch.flip(obvs, dims=[0])

        for ii, obv in enumerate(rev_obvs[:-1]):
            log_bs = (
                self.log_trans.clone()
                + self.emission(None, obv).unsqueeze(0)
                + log_bs.unsqueeze(0)
            )

            log_bs = torch.logsumexp(log_bs, dim=1)

        obv = rev_obvs[-1]
        log_evidence = torch.logsumexp(
            self.log_trans[0, :] + self.emission(None, obv) + log_bs, dim=0
        )

        return log_evidence

    def log_forward(self, obvs):
        """
        Log evidence (marginalised over latent) by the forward method,
        returning forward array.

        See termination step after Eqn. (3.11) of Durbin.
        """
        log_fs_init = self.log_pi.clone()

        log_fs = torch.zeros(1 + len(obvs) + 1, len(log_fs_init)).to(self.device)
        log_fs[0] = log_fs_init

        for ii, obv in enumerate(obvs):
            interim = log_fs[ii].clone().unsqueeze(-1) + self.log_trans.clone()
            log_fs[ii + 1] = self.emission(None, obv) + torch.logsumexp(interim, dim=0)

        # NB final transition into the book end state.
        log_fs[-1] = log_fs[-2] + self.log_trans[:, 0]

        return torch.logsumexp(log_fs[-1], dim=0), log_fs

    def log_backward(self, obvs):
        """
        Log evidence (marginalised over latent) by the forward method,
        returning backward array.

        See termination step before Eqn. (3.14) of Durbin.
        """
        log_bs_init = self.log_trans[:, 0].clone()

        log_bs = torch.zeros(1 + len(obvs), len(log_bs_init)).to(self.device)
        log_bs[-1] = log_bs_init

        rev_obvs = torch.flip(obvs, dims=[0])

        for ii, obv in enumerate(rev_obvs[:-1]):
            interim = (
                self.log_trans.clone()
                + self.emission(None, obv).unsqueeze(0)
                + log_bs[-(ii + 1), :].unsqueeze(0)
            )

            log_bs[-(ii + 2)] = torch.logsumexp(interim, dim=1)

        obv = rev_obvs[-1]
        log_bs[0] = self.log_trans[0, :] + self.emission(None, obv) + log_bs[1]

        log_evidence = torch.logsumexp(log_bs[0], dim=0)

        return log_evidence, log_bs

    def log_state_posterior(self, obvs):
        """
        Eqn. (3.14) of Durbin.
        """
        log_evidence_forward, log_forward_array = self.log_forward(obvs)
        log_evidence_backward, log_backward_array = self.log_backward(obvs)

        # TODO double check indexing
        log_state_posterior = (
            log_forward_array[1:-1] + log_backward_array[:-1] - log_evidence_forward
        )

        return log_state_posterior

    def max_posterior_decoding(self, obvs):
        """
        Eqn. (3.15) of Durbin.
        """
        return torch.argmax(self.log_state_posterior(obvs), dim=1).to(torch.int32)

    def log_transition_posterior(self, obvs):
        """
        Eqn. (3.19) of Durbin.
        """
        # NB we need x_(i+1)
        obvs = torch.roll(obvs.clone(), shifts=-1)
        log_ems = self.emission(None, obvs).T.unsqueeze(2)

        log_evidence_forward, log_forward_array = self.log_forward(obvs)
        _, log_backward_array = self.log_backward(obvs)

        # NB limit to observed states
        log_forward_array = log_forward_array[1:-1].unsqueeze(2)
        log_backward_array = log_backward_array[1:].unsqueeze(1)

        log_transition_posterior = log_forward_array + log_backward_array
        log_transition_posterior += self.log_trans.clone()

        log_transition_posterior += log_ems
        log_transition_posterior -= log_evidence_forward

        # NB the last row in time is invalid
        return log_transition_posterior[:-1]

    def exp_transition_counts(self, obvs, pseudo_counts=None):
        log_transition_posterior = self.log_transition_posterior(obvs)
        exp_transition_counts = torch.logsumexp(log_transition_posterior, dim=0).exp()

        if pseudo_counts is not None:
            exp_transition_counts += pseudo_counts

        return exp_transition_counts

    def exp_emission_counts(self, obvs, pseudo_counts=None):
        """ """
        # TODO emission model specific, i.e. assumes Categorical.  Move to Emission class?
        log_evidence_forward, log_forward_array = self.log_forward(obvs)
        _, log_backward_array = self.log_backward(obvs)

        # NB no bookend states
        log_forward_array, log_backward_array = (
            log_forward_array[1:-1],
            log_backward_array[1:],
        )
        interim = log_forward_array + log_backward_array - log_evidence_forward

        if pseudo_counts is None:
            exp_emission_counts = torch.zeros(
                (self.n_states, self.n_obvs), device=self.device
            )
        else:
            exp_emission_counts = pseduo_counts

        # TODO
        for ii, obv in enumerate(torch.unique(obvs)):
            mask = obvs == obv
            exp_emission_counts[:, ii] = torch.logsumexp(interim[mask], dim=0).exp()

        return exp_emission_counts

    def baum_welch_update(
        self, obvs, transition_pseudo_counts=None, emission_pseudo_counts=None
    ):
        """ """
        exp_transition_counts = self.exp_transition_counts(
            obvs, transition_pseudo_counts
        )
        exp_transition_counts /= torch.sum(exp_transition_counts, dim=1).unsqueeze(1)

        # TODO emission model specific, i.e. assumes Categorical.  Move to Emission class.
        exp_emission_counts = self.exp_emission_counts(obvs, emission_pseudo_counts)
        exp_emission_counts /= torch.sum(exp_emission_counts, dim=1).unsqueeze(1)

        return exp_transition_counts, exp_emission_counts

    def baum_welch_training(self):
        raise NotImplementedError()

    def torch_training(self, obvs, optimizer=None, n_epochs=1, lr=1.0e-2):
        optimizer = Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            loss = -self.log_forward_scan(obvs)
            loss.backward()

            optimizer.step()

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
        logger.info(f"Transition probability matrix:\n{self.log_trans.exp()}\n")

        self.emission_model.validate()

    @property
    def parameters_dict(self):
        """Dict with named torch parameters."""
        # TODO name clash.                                                                                                                                                                                                                                                      
        return {"log_trans": self.log_trans} | self.emission_model.parameters_dict

    def to_device(self, device):
        self.log_pi = self.log_pi.to(device)
        self.log_trans = self.log_trans.to(device)
        self.emission_model = self.emission_model.to_device(device)
    

if __name__ == "__main__":
    # TODO set seed for cuda / mps
    torch.manual_seed(123)

    # TODO BUG? must be even?
    n_seq, device = 20, "cpu"

    categorical = CategoricalEmission(n_states=4, n_obvs=4, device=device)
    # casino = Casino(device=device)

    emission_model = categorical
    emission_model.validate()

    # NB (n_states * n_obvs) action space.
    hmm = HMM(
        n_states=emission_model.n_states - 1,
        emission_model=emission_model,
        log_trans=None,
        device=device,
    )

    # NB hidden states matched to observed time steps.
    hidden_states = hmm.sample_hidden(n_seq, bookend=True)
    obvs = hmm.sample_obvs(n_seq, hidden_states)

    logger.info(f"Sampled hidden sequence:\n{hidden_states}")
    logger.info(f"Observed sequence:\n{obvs}")

    log_like = hmm.log_like(obvs, hidden_states)

    logger.info(f"Found a log likelihood= {log_like:.4f} for generated hidden states")

    # NB P(x, pi) with tracing for most probably state sequence
    log_joint_prob, penultimate_state, trace_table = hmm.viterbi(obvs, traced=True)

    logger.info(
        f"Found a joint probability P(x, pi)={log_joint_prob:.4f} with trace:\n{trace_table}"
    )

    # NB Most probable state sequence
    viterbi_decoded_states = hmm.viterbi_traceback(trace_table, penultimate_state)

    logger.info(
        f"Found the penultimate state to be {penultimate_state} with a Viterbi decoding of:\n{viterbi_decoded_states}"
    )

    # NB P(x) marginalised over hidden states by forward & backward scan - no array traceback.
    log_evidence_forward_scan = hmm.log_forward_scan(obvs)
    log_evidence_backward_scan = hmm.log_backward_scan(obvs)

    # NB P(x) marginalised over hidden states by forward & backward method - array traceback.
    log_evidence_forward, log_forward_array = hmm.log_forward(obvs)
    log_evidence_backward, log_backward_array = hmm.log_backward(obvs)

    logger.info(
        f"Found the evidence to be {log_evidence_forward:.4f} by the forward method."
    )
    logger.info(
        f"Found the evidence to be {log_evidence_backward:.4f} by the backward method."
    )

    assert torch.allclose(log_evidence_forward, log_evidence_backward)
    assert torch.allclose(log_evidence_forward_scan, log_evidence_forward)

    logger.info(f"Found the log forward array to be:\n{log_forward_array}")
    logger.info(f"Found the log backward array to be:\n{log_backward_array}")

    # NB P(pi_i = k | x) for all i.
    log_state_posteriors = hmm.log_state_posterior(obvs)

    logger.info(f"Found the state posteriors to be:\n{log_state_posteriors}")

    # NB argmax_k P(pi_i = k | x) for all i.
    decoded_states = hmm.max_posterior_decoding(obvs)

    logger.info(f"Found a state decoding (max. disjoint posterior):\n{decoded_states}")

    # NB Satisfying!
    assert torch.allclose(hidden_states, decoded_states)

    log_transition_posteriors = hmm.log_transition_posterior(obvs)

    logger.info(
        f"Found the log transition posteriors to be:\n{log_transition_posteriors}"
    )

    exp_transition_counts = hmm.exp_transition_counts(obvs)

    logger.info(f"Found the exp transition counts to be:\n{exp_transition_counts}")

    # NB specific to Categorical emission
    exp_emission_counts = hmm.exp_emission_counts(obvs)

    logger.info(f"Found the exp emission counts to be:\n{exp_emission_counts}")

    baum_welch_transitions, baum_welch_emissions = hmm.baum_welch_update(obvs)

    logger.info(
        f"Found the transitions Baum-Welch update to be:\n{baum_welch_transitions}"
    )
    logger.info(f"Found the emissions Baum-Welch update to be:\n{baum_welch_emissions}")

    torch_n_epochs, torch_log_evidence_forward = hmm.torch_training(
        obvs, n_epochs=1_000
    )

    logger.info(
        f"After training with torch for {torch_n_epochs}, found the evidence to be {torch_log_evidence_forward:.4f} by the forward method."
    )

    for key in hmm.parameters_dict:
        logger.info(
            f"Found optimised parameters for {key} to be:\n{list(hmm.parameters_dict[key])}"
        )

    logger.info(f"Done.\n\n")
