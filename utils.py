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


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def get_log_probs_precision():
    return -99.0

def get_scalars(scalars):
    if isinstance(scalars, int):
        yield scalars
    elif isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            yield scalar.item()
        else:
            for ss in scalars:
                yield ss.item()
    else:
        raise RuntimeError(
            f"get_scalars() does not support input of type {type(scalar)} and len {len(scalar)}.  Found {scalar}."
        )
