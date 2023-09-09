import os
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import yaml


@dataclass
class Config:
    output_dir: str
    training: Literal["sdxl_adapter"]

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
    train_shards: str = "pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-min512-data/{00000..01210}.tar -"
    shuffle_buffer_size: int = 1000
    resolution: int = 1024
    batch_size: int = 8


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

training_config: Config = Config(**yaml_config)
