Single Machine, Single GPU

```sh
DIFFUSERS_UTILS_TRAINING_CONFIG="<path to config file>" torchrun --standalone --nproc_per_node=1 training_loop.py
```

Single Machine, Multiple GPUs

```sh
DIFFUSERS_UTILS_TRAINING_CONFIG="<path to config file>" torchrun --standalone --nproc_per_node=<number of gpus> training_loop.py
```

Multiple Machines, Multiple GPUs

TODO