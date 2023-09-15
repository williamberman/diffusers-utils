Single Machine, Single GPU

```sh
DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE="<optional override, see below>" \
    DIFFUSERS_UTILS_TRAINING_CONFIG="<path to config file>" \
    torchrun \
        --standalone \
        --nproc_per_node=1 \
        training_loop.py
```

Single Machine, Multiple GPUs

```sh
DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE="<optional override, see below>" \
    DIFFUSERS_UTILS_TRAINING_CONFIG="<path to config file>" \
    torchrun \
        --standalone \
        --nproc_per_node=<number of gpus> \
        training_loop.py
```

Multiple Machines, Multiple GPUs

```sh
DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE="<optional override, see below>" \
    DIFFUSERS_UTILS_TRAINING_CONFIG="<path to config file>" \
    sbatch \
        --nodes=<number of nodes> \
        --partition=<production-cluster or dev-cluster> \
        --output=<log file> \
        training_loop.slurm
```


DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE

use to choose a set of configs under the `override` key to override config
in the top level config

i.e. with the yaml file,

```yaml
mixed_precision: "no"
batch_size: 16
learning_rate: 0.00001

override:
    use_fp16_mixed_precision:
        mixed_precision: fp16
        batch_size: 32

    use_bf16_mixed_precision:
        mixed_precision: bf16
        batch_size: 32
```

setting `DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE=use_fp16_mixed_precision` would
set mixed_precision to fp16 and batch_size to 32 while leaving the learning rate as
0.00001
