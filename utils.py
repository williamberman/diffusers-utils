from safetensors import safe_open
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def load_safetensors_state_dict(filename):
    state_dict = {}

    with safe_open(filename, framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    return state_dict


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def maybe_ddp_dtype(m):
    if isinstance(m, DDP):
        m = m.module
    return m.dtype


def maybe_ddp_module(m):
    if isinstance(m, DDP):
        m = m.module
    return m
