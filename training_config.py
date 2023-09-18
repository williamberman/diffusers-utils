import dataclasses
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
import yaml

DIFFUSERS_UTILS_TRAINING_CONFIG = "DIFFUSERS_UTILS_TRAINING_CONFIG"
DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE = "DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE"

if DIFFUSERS_UTILS_TRAINING_CONFIG not in os.environ:
    raise ValueError(f"Must set environment variable `{DIFFUSERS_UTILS_TRAINING_CONFIG}` to path to the yaml config to use for the training run.")


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

    # TODO: bad naming
    # `training_config.training == "sdxl_controlnet"` specific config
    controlnet_type: Optional[Literal["canny", "inpainting"]] = None
    controlnet_variant: Literal["default", "full", "pre_encoded_controlnet_cond"] = "default"
    controlnet_train_base_unet: bool = False

    # core training config
    learning_rate: float = 0.00001
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

    # wandb
    project_name: Optional[str] = None
    training_run_name: Optional[str] = None


# this instance will never be reset, only the values inside it will be overwritten
# this allows other modules to import the value once, and then call `load_training_config`
# to re-read updated values from the config file.
#
# All initial default values will be immediately over written when `load_training_config` is
# called at module initialization
training_config: Config = Config(output_dir="NOT USED", training="sdxl_controlnet", train_shards="NOT USED")


def load_training_config():
    global training_config, training_run_name

    with open(os.environ[DIFFUSERS_UTILS_TRAINING_CONFIG], "r") as f:
        yaml_config: Dict = yaml.safe_load(f.read())

    override_configs = yaml_config.pop("overrides", {})

    if DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE in os.environ:
        override_config_key = os.environ[DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE]

        if override_config_key not in override_configs:
            raise ValueError(f"{override_config_key} is not one of the available overrides {override_configs.keys()}")

        yaml_config.update(override_configs[override_config_key])

    if "mixed_precision" not in yaml_config or yaml_config["mixed_precision"] is None or yaml_config["mixed_precision"] == "no":
        yaml_config["mixed_precision"] = None
    elif yaml_config["mixed_precision"] == "fp16":
        yaml_config["mixed_precision"] = torch.float16
    elif yaml_config["mixed_precision"] == "bf16":
        yaml_config["mixed_precision"] = torch.bfloat16
    else:
        assert False

    training_config_ = Config(**yaml_config)

    if training_config_.training == "sdxl_adapter":
        if training_config_.adapter_type is None:
            raise ValueError('must set `adapter_type` if `training` set to "sdxl_adapter"')

    if training_config_.training == "sdxl_controlnet":
        if training_config_.controlnet_type is None:
            raise ValueError('must set `controlnet_type` if `training` set to "sdxl_controlnet"')

    # dirty set/get attr because dataclasses do not allow setting/getting via strings
    for field in dataclasses.fields(Config):
        setattr(training_config, field.name, getattr(training_config_, field.name))


load_training_config()
