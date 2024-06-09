import torch
import numpy as np
from utils import get_device


class CasinoEmission(torch.nn.Module):
    def __init__(self, device=None):
        super(CasinoEmission, self).__init__()

        self.n_states, self.n_obvs = 2, 6
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
        if state is None:
            return self.log_em[:, obs]
        elif obs is None:
            return self.log_em[state, :]
        else:
            return self.log_em[state, obs]


if __name__ == "__main__":
    emitter = CasinoEmission()
    fair_rolls = emitter.fair_roll(n_roll=10)

    loaded_rolls = emitter.loaded_roll(n_roll=100_000)
    loaded_weights = list(map(lambda count: (loaded_rolls == count).cpu().numpy().mean(), range(1, 7)))
    
    print(loaded_weights)
