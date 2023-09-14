import os
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import yaml


@dataclass
class Config:
    # required config
    output_dir: str
    training: Literal["sdxl_adapter", "sdxl_unet", "sdxl_controlnet"]
    train_shards: str

    # `training_config.training == "sdxl_adapter"` specific config
    adapter_type: Optional[Literal["mediapipe_pose", "openpose"]] = None
    adapter_conditioning_scale: float = 1.0
    adapter_conditioning_factor: float = 1.0

    # `training_config.training == "sdxl_controlnet"` specific config
    controlnet_type: Optional[Literal["canny", "inpainting"]] = None

    # core training config
    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[torch.dtype] = None
    batch_size: int = 8
    max_train_steps: int = 30_000
    resume_from: Optional[str] = None
    start_step: int = 0

    # data config
    resolution: int = 1024
    shuffle_buffer_size: int = 1000
    proportion_empty_prompts: float = 0.1

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

if training_config.training == "sdxl_controlnet":
    if training_config.controlnet_type is None:
        raise ValueError(
            'must set `controlnet_type` if `training` set to "sdxl_controlnet"'
        )
