from safetensors import safe_open
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from safetensors.torch import load_file


# TODO remove
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

def load_model(model_constructor, state_dict_path, device):
    import load_state_dict_patch

    with torch.device('meta'):
        model = model_constructor()

    state_dict = load_file(state_dict_path, device=device)

    model.load_state_dict(state_dict, assign=True)

    return model
