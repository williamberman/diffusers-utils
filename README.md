Single Machine, Single GPU

```sh
CONFIG="<path to config file>" torchrun --standalone --nproc_per_node=1 training_loop.py
```

Single Machine, Multiple GPUs

```sh
CONFIG="<path to config file>" torchrun --standalone --nproc_per_node=8 training_loop.py
```

Multiple Machines, Multiple GPUs

TODO