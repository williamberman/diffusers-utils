from torch.nn.parallel import DistributedDataParallel as DDP


def maybe_ddp_dtype(m):
    if isinstance(m, DDP):
        m = m.module
    return m.dtype


def maybe_ddp_module(m):
    if isinstance(m, DDP):
        m = m.module
    return m


def maybe_ddp_device(m):
    if isinstance(m, DDP):
        m = m.module
    return m.device
