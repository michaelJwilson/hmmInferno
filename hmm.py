import torch
from torch.optim import Adam
from torch.distributions import negative_binomial

# TODO set seed for cuda / mps
torch.manual_seed(123)


def get_device():
    # NB mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.gpu.is_available():
        device = torch.device("gpu")
    else:
        device = torch.device("cpu")

    return device


class CategoricalEmission(torch.nn.Module):
    def __init__(self, n_states, n_obvs, device=None):
        super(CategoricalEmission, self).__init__()

        self.device = get_device() if device is None else device
        
        # NB rows sums to zero, as prob. to emit to any obs. is unity.
        #    nn.Parameter marks this to be optimised via backprop.
        self.log_em = torch.randn(n_states, n_obvs)
        self.log_em[0, :] = -99.0
        self.log_em = torch.nn.Parameter(self.log_em)

        self.log_em.data = self.log_em.data.log_softmax(dim=1)

    def to_device(self, device):
        self.device = device
        self.log_em = self.log_em.to(device)

        return self
        
    def emission(self, state, obs):
        if state is None:
            return self.log_em[:, obs]
        elif obs is None:
            return self.log_em[state, :]
        else:
            return self.log_em[state, obs]


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
        self.dist = negative_binomial.NegativeBinomial(total_count, probs=probs, validate_args=True)

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
    def __init__(self, n_states, n_obvs, log_trans=None, device=None):
        super(HMM, self).__init__()

        self.device = get_device() if device is None else device

        # NB we assume a bookend state to transition in/out.
        self.n_states = 1 + n_states

        # NB number of possible observable states, as opposed to sequence length.
        self.n_obvs = n_obvs

        # NB bookends
        self.n_steps = 1 + n_obvs + 1

        # NB We start (and end) in the bookend state.  No gradient required.
        self.log_pi = -99.0 * torch.ones(self.n_states, requires_grad=False)
        self.log_pi[0] = 0.0

        if log_trans is None:
            self.log_trans = torch.randn(self.n_states, self.n_states)
        else:
            assert isinstance(
                log_trans, torch.Tensor
            ), "log_trans must be a torch.tensor instance"

            assert log_trans.shape == (
                self.n_states,
                self.n_states,
            ), "log_trans must include bookend states."

            self.log_trans = log_trans

        self.log_trans = torch.nn.Parameter(self.log_trans)
            
        # NB rows sums to zero, as prob. to transition to any state is unity.
        self.log_trans.data = self.log_trans.data.log_softmax(dim=1)

        self.emission_model = CategoricalEmission(self.n_states, self.n_obvs, device=device)

        self.to_device(device)
        self.validate()

    def emission(self, state, obs):
        return self.emission_model.emission(state, obs)

    def validate(self):
        # NB log probs.
        assert torch.allclose(
            self.log_trans.logsumexp(dim=1),
            torch.zeros(self.log_trans.size(0)).to(device),
            rtol=1e-05,
            atol=1e-06,
        )

        print(f"\n\nInitialised HMM with start log probs.:\n{self.log_pi}")
        print(f"\n\nwith transition log probs.:\n{self.log_trans}\n")
        print(f"\nwith emission log probs.:\n{self.emission_model.log_em}\n")

    def to_device(self, device):
        self.log_pi = self.log_pi.to(device)
        self.log_trans = self.log_trans.to(device)
        self.emission_model = self.emission_model.to_device(device)
        self.device = device

    @staticmethod
    def bookend_states(hidden_states):
        """
        Obs. sequence transitions from and to a book end state.
        """
        return torch.cat((torch.tensor([0]), hidden_states, torch.tensor([0])), dim=0)

    def log_like(self, obvs, states):
        """
        Eqn. (3.6) of Durbin.
        """
        # NB we must start in a bookend state
        assert states[0] == states[-1] == 0

        # NB we start in the bookend state with unit probability.
        log_like = self.log_pi[0].clone()
        last_state = states[0].clone()

        for obs, state in zip(obvs, states[1:-1]):
            log_like += self.emission(state, obs)
            log_like += self.log_trans[last_state, state]

            last_state = state

        return log_like

    def viterbi(self, obvs, traced=False):
        """
        Decoding the obs. sequence to a hidden sequence with max. likelihood.
        NB no need to calculate the traceback when training.

        See value algebra on pg. 56 of Durbin.
        """
        log_vs = self.log_pi.clone()

        if traced:
            trace_table = torch.zeros(
                1 + len(obvs) + 1, self.n_states, dtype=torch.int32
            )

        for ii, obs in enumerate(obvs):
            log_vs = log_vs[:, None] + self.log_trans.clone()
            log_vs, states = torch.max(log_vs, dim=0)
            log_vs += self.emission(states, obs)

            if traced:
                trace_table[1 + ii, :] = states

        # NB finally transition into the book end state.
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
            log_fs = log_fs[:, None] + self.log_trans.clone()
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
                + self.emission(None, obv)[None, :]
                + log_bs[None, :]
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
        return torch.argmax(self.log_state_posterior(obvs), dim=1)

    def log_transition_posterior(self, obvs):
        """
        Eqn. (3.19) of Durbin.
        """
        # NB we need x_(i+1)
        obvs = torch.roll(obvs.clone(), shifts=-1)
        log_ems = self.emission(None, obvs).T.unsqueeze(2)

        log_evidence_forward, log_forward_array = self.log_forward(obvs)
        log_evidence_backward, log_backward_array = self.log_backward(obvs)

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
        """
        TODO emission model specific, i.e. assumes Categorical.  Move to Emission class.
        """
        log_evidence_forward, log_forward_array = self.log_forward(obvs)
        log_evidence_backward, log_backward_array = self.log_backward(obvs)

        # NB no bookend states
        log_forward_array = log_forward_array[1:-1]
        log_backward_array = log_backward_array[1:]

        unique_obvs = torch.unique(obvs)
        
        interim = log_forward_array + log_backward_array - log_evidence_forward

        # TODO int?
        exp_emission_counts = torch.zeros((self.n_states, self.n_obvs))

        # TODO
        for ii, obv in enumerate(unique_obvs):
            mask = obvs == ii
            exp_emission_counts[:, ii] = torch.logsumexp(interim[mask], dim=0).exp()

        if pseudo_counts is not None:
            exp_emission_counts += pseduo_counts

        return exp_emission_counts

    def baum_welch_update(
        self, obvs, transition_pseudo_counts=None, emission_pseudo_counts=None
    ):
        """
        """
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

    
    def backprop_training(model):
        optimizer.zero_grad()

        loss = calc_dkl(logp, logq)
        loss.backward()

        optimizer.step()

        
if __name__ == "__main__":
    device = "cpu"
    n_states, n_obvs, n_seq = 4, 4, 10

    # NB (n_states * n_obvs) action space.
    hmm = HMM(n_states=n_states, n_obvs=n_obvs, device=device)
    
    # NB Normal(0., 1.) for observed sequence.
    obvs = torch.randint(low=0, high=n_states, size=(n_seq,))

    # NB hidden states matched to observed time steps.
    states = torch.randint(low=1, high=(n_states + 1), size=(n_seq,))
    states = hmm.bookend_states(states).to(device)

    log_like = hmm.log_like(obvs, states)

    # NB P(x, pi) with tracing for most probably state sequence
    log_joint_prob, penultimate_state, trace_table = hmm.viterbi(obvs, traced=True)

    # NB Most probable state sequence
    viterbi_decoded_states = hmm.viterbi_traceback(trace_table, penultimate_state)

    # NB P(x) marginalised over hidden states by forward & backward scan - no array traceback.
    log_evidence_forward = hmm.log_forward_scan(obvs)
    log_evidence_backward = hmm.log_backward_scan(obvs)

    assert log_evidence_forward == log_evidence_backward

    # NB P(x) marginalised over hidden states by forward & backward method - array traceback.
    log_evidence_forward, log_forward_array = hmm.log_forward(obvs)
    log_evidence_backward, log_backward_array = hmm.log_backward(obvs)
    
    assert log_evidence_forward == log_evidence_backward

    # NB P(pi_i = k | x) for all i.
    log_state_posteriors = hmm.log_state_posterior(obvs)

    # NB argmax_k P(pi_i = k | x) for all i.
    decoded_states = hmm.max_posterior_decoding(obvs)

    log_transition_posteriors = hmm.log_transition_posterior(obvs)
    exp_transition_counts = hmm.exp_transition_counts(obvs)

    # NB specific to Categorical emission
    exp_emission_counts = hmm.exp_emission_counts(obvs)

    baum_welch_transitions, baum_welch_emissions = hmm.baum_welch_update(obvs)

    # TODO
    # assert torch.allclose(
    #    viterbi_decoded_states.long(),
    #    decoded_states.long(),
    #    
    #)

    print(f"\nObserved sequence: {obvs}")
    print(f"\nAssumed Hidden sequence: {states}")
    print(f"\nTrace table:\n{trace_table}")
    print(
        f"\nFound the penultimate state to be {penultimate_state} with a Viterbi decoding of:\n{decoded_states}"
    )
    print(f"\nFound the evidence to be {log_evidence_forward} by the forward method.")
    print(f"Found the evidence to be {log_evidence_backward} by the backward method.")
    print(f"\nFound the log forward array to be:\n{log_forward_array}")
    print(f"\nFound the log backward array to be:\n{log_backward_array}")
    print(f"\nFound the state posteriors to be:\n{log_state_posteriors}")
    print(f"\nFound decoded states:\n{decoded_states}")
    print(f"\nFound the log transition posteriors to be:\n{log_transition_posteriors}")
    print(f"\nFound the exp transition counts to be:\n{exp_transition_counts}")
    print(f"\nFound the exp emission counts to be:\n{exp_emission_counts}")
    print(f"\nFound the transitions Baum-Welch update to be:\n{baum_welch_transitions}")
    print(f"\nFound the emissions Baum-Welch update to be:\n{baum_welch_transitions}")
    print(f"\n\nDone.\n\n")
