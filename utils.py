import torch

def get_device():
    # NB mac                                                                                                                                                                                                         
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.gpu.is_available():
        device = torch.device("gpu")
    else:
        device = torch.device("cpu")

    return device

def bookend_sequence(sequence, device=None):
    """                                                                                                                                                                                                           
    Bookend a sequence with the 0-state.
    """
    return torch.cat(
        (
            torch.tensor([0], dtype=torch.int32, device=device),
            sequence,
            torch.tensor([0], dtype=torch.int32, device=device),
        ),
        dim=0,
    )
