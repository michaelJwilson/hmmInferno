import torch


def get_log_probs_precision():
    return -99.0


def get_bookend_token():
    # NB should be >>> expected transcript count, e.g. 999_999_999    
    return 0 

def get_device():
    # NB mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.backends.gpu.is_available():
        device = torch.device("gpu")
    else:
        device = torch.device("cpu")

    return device


def bookend_sequence(
    sequence, device=None, dtype=torch.int32, token=get_bookend_token()
):
    """
    Bookend a sequence with the 0-state.
    """
    return torch.cat(
        (
            torch.tensor([token], dtype=dtype, device=device),
            sequence,
            torch.tensor([token], dtype=dtype, device=device),
        ),
        dim=0,
    )


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def get_scalars(scalars):
    if isinstance(scalars, int):
        yield scalars
    elif isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            yield scalars.item()
        else:
            for ss in scalars:
                yield ss.item()
    else:
        raise RuntimeError(
            f"get_scalars() does not support input of type {type(scalar)} and len {len(scalar)}.  Found {scalar}."
        )


def set_scalars(scalars, device=None, requires_grad=False, dtype=torch.int32):
    if device is None:
        device = get_device()

    if isinstance(scalars, int):
        return torch.tensor(
            [scalars], dtype=dtype, device=device, requires_grad=requires_grad
        )
    elif isinstance(scalars, torch.Tensor):
        if scalars.dim() == 0:
            return scalars.unsqueeze(0)
        else:
            return scalars
    else:
        raise RuntimeError(f"set_scalars() does not support {type(scalars)} type.")
