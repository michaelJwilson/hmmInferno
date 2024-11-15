import torch
import random
import numpy as np

from torch import nn
from torchvision.transforms import ToTensor

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


class Model(nn.Module):
    def __init__(self, nstates, nsteps=10):
        super().__init__()

        self.nsteps = nsteps
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential()
        self.linear_layer = nn.Linear(nstates, nstates, bias=False)

        for _ in range(nsteps):
            self.layers.append(self.linear_layer)

    def forward(self, start):
        raw = self.layers(start)
        return raw / torch.sum(raw)


if __name__ == "__main__":
    device = get_device()
    nstates, nsteps = 7, 100

    model = Model(nstates=nstates, nsteps=nsteps).to(device)

    # NB we must start in a single state.
    start = torch.ones(nstates, dtype=torch.float32, device=device)
    start = start / torch.sum(start)

    current = model(start)

    print(start)
    print(current)

    transfer = model.linear_layer.weight.data.cpu().numpy()
    nstep_transfer = np.linalg.matrix_power(transfer, nsteps)

    result = np.dot(nstep_transfer, start.cpu().numpy())
    result = result / np.sum(result)

    print(result)

    assert np.allclose(current.cpu().detach().numpy(), result)
