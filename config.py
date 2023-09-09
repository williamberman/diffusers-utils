import os
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import yaml


@dataclass
class Config:
    output_dir: str
    gradient_accumulation_steps: int
    mixed_precision: Optional[torch.dtype] = None
    resume_from: str
    checkpointing_steps: int
    checkpoints_total_limit: int
    validation_steps: int
    max_train_steps: int
    validation_prompts: List[str]
    num_validation_images: int
    train_shards: str
    shuffle_buffer_size: int
    resolution: int
    batch_size: int
    training: Literal["sdxl_adapter"]


yaml_config = yaml.safe_load(os.environ["DIFFUSERS_UTILS_CONFIG"])

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

config: Config = Config(**yaml_config)
