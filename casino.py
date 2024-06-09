import torch
import numpy as np
from utils import get_device


class Casino(torch.nn.Module):
    def __init__(self, device=None):
        super(Casino, self).__init__()

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

    def sample(self, n_seq=1):
        is_loaded = torch.randint(low=0, high=2, size=(n_seq,), device=self.device)

        obvs = self.fair_roll(n_roll=n_seq)
        obvs[is_loaded] = self.loaded_roll(n_roll=n_seq)[is_loaded]

        return obvs
        
    def emission(self, state, obvs, log=True):
        if state is None:
            state = torch.tensor(range(self.n_states), device=self.device, dtype=torch.int32)

        n_states = len(state)

        # NB there is no emission from a bookend state
        result = -99. * torch.ones((n_states, self.n_obvs), device=self.device)
        state = state.unsqueeze(-1) * torch.ones_like(result, dtype=torch.int32, device=self.device)
        
        result[(state == 1)] = torch.tensor([1.0 / 6.0], device=self.device)
        result[(state == 2) & (obvs == 6)] = torch.tensor([1.0 / 2.0], device=self.device)
        result[(state == 2) & (obvs != 6)] = torch.tensor([1.0 / 10.0], device=self.device)

        if log:
            result = result.log()
        
        return result
        
    @property
    def log_trans(self):
        return torch.tensor([[0.0, 0.5, 0.5], [1.0, 0.95, 0.05], [1.0, 0.1, 0.9]]).log()

    def validate(self):
        # TODO
        # logger.warning()
        return


if __name__ == "__main__":
    # TODO set seed for cuda / mps                                                                                                                                                                                   
    torch.manual_seed(123)
    
    casino = CasinoEmission()
    fair_rolls = casino.fair_roll(n_roll=10)

    # loaded_rolls = casino.loaded_roll(n_roll=100_000)

    # TODO gpu arithmetic.
    # loaded_weights = list(map(lambda count: (loaded_rolls == count).cpu().numpy().mean(), range(1, 7)))

    states, obvs = None, torch.tensor([1, 2, 3, 4, 5, 6], device=get_device())
    result = casino.emission(None, obvs)

    print(states, obvs)
    print(result)
