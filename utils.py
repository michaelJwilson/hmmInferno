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
