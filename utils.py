from safetensors import safe_open
from torch import nn


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
