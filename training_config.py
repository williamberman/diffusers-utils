import os
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import yaml


@dataclass
class Config:
    # required config
    output_dir: str
    training: Literal["sdxl_adapter"]
    train_shards: str

    # set adapter type if `training_config.training == "sdxl_adapter"`
    adapter_type: Optional[Literal["mediapipe_pose"]] = None

    # core training config
    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[torch.dtype] = None
    batch_size: int = 8
    max_train_steps: int = 30_000
    resume_from: Optional[str] = None

    # data config
    resolution: int = 1024
    shuffle_buffer_size: int = 1000

    # validation
    validation_steps: int = 500
    num_validation_images: int = 2
    validation_prompts: Optional[List[str]] = None
    validation_images: Optional[List[str]] = None

    # checkpointing
    checkpointing_steps: int = 1000
    checkpoints_total_limit: int = 5


if "DIFFUSERS_UTILS_TRAINING_CONFIG" not in os.environ:
    raise ValueError(
        "Must set environment variable `DIFFUSERS_UTILS_TRAINING_CONFIG` to path to the yaml config to use for the training run."
    )

with open(os.environ["DIFFUSERS_UTILS_TRAINING_CONFIG"], "r") as f:
    yaml_config = yaml.safe_load(f.read())

if (
    "mixed_precision" not in yaml_config
    or yaml_config["mixed_precision"] is None
    or yaml_config["mixed_precision"] == "no"
):
    yaml_config["mixed_precision"] = None
elif yaml_config["mixed_precision"] == "fp16":
    yaml_config["mixed_precision"] = torch.float16
elif yaml_config["mixed_precision"] == "bf16":
    yaml_config["mixed_precision"] = torch.bfloat16
else:
    assert False

training_config: Config = Config(**yaml_config)

if training_config.training == "sdxl_adapter":
    if training_config.adapter_type is None:
        raise ValueError('must set `adapter_type` if `training` set to "sdxl_adapter"')
