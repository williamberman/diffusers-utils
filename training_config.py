import os
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import yaml


@dataclass
class Config:
    output_dir: str
    training: Literal["sdxl_adapter"]
    train_shards: str

    # training: "sdxl_adapter"
    adapter_type: Optional[Literal["mediapipe_pose"]] = None

    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[torch.dtype] = None
    resume_from: Optional[str] = None
    checkpointing_steps: int = 1000
    checkpoints_total_limit: int = 5
    validation_steps: int = 500
    max_train_steps: int = 30_000
    validation_prompts: Optional[List[str]] = None
    num_validation_images: int = 2
    shuffle_buffer_size: int = 1000
    resolution: int = 1024
    batch_size: int = 8


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
