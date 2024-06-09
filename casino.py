import torch
import numpy as np
from utils import get_device


class CasinoEmission(torch.nn.Module):
    def __init__(self, device=None):
        super(CasinoEmission, self).__init__()

        # NB includes a bookend state denoted 0.
        self.n_states, self.n_obvs = 3, 6
        self.device = get_device() if device is None else device

    def to_device(self, device):
        self.device = device
        return self

    def fair_roll(self, n_roll=1):
        # NB uniform probability (of 1/6).
        return torch.randint(low=1, high=7, size=(n_roll,), device=self.device)

    def loaded_roll(self, n_roll=1):
        # NB 50% of a 6, 1/10 otherwise.
        draws = torch.randint(low=1, high=11, size=(n_roll,), device=self.device)
        draws[draws > 6] = 6

        return draws

    def emission(self, state, obs):
        # NB there is no emission from a bookend state.
        if state == 0:
            return -99.0
        elif state == 1:
            return np.log(1.0 / 6.0)
        elif state == 2:
            return np.log(1.0 / 2.0) if obs == 6 else np.log(1.0 / 10.0)
        else:
            raise RuntimeError()

    @property
    def log_trans(self):
        return torch.tensor([[0.0, 0.5, 0.5], [1.0, 0.95, 0.05], [1.0, 0.1, 0.9]]).log()

    def validate(self):
        # TODO
        # logger.warning()
        return


if __name__ == "__main__":
    casino = CasinoEmission()
    fair_rolls = casino.fair_roll(n_roll=10)

    # loaded_rolls = casino.loaded_roll(n_roll=100_000)

    # TODO gpu arithmetic.
    # loaded_weights = list(map(lambda count: (loaded_rolls == count).cpu().numpy().mean(), range(1, 7)))

    log_prob = casino.emission(1, 5)

    print(log_prob)
