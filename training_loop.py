import dataclasses
import logging
import os
import shutil
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Literal, Optional

import safetensors.torch
import torch
import torch.distributed as dist
import wandb
import yaml
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sdxl import GetSDXLConditioningImages, SDXLTraining, get_sdxl_dataset

DIFFUSERS_UTILS_TRAINING_CONFIG = "DIFFUSERS_UTILS_TRAINING_CONFIG"
DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE = "DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

device = int(os.environ["LOCAL_RANK"])


@dataclass
class Config:
    # required config
    output_dir: str
    training: Literal["sdxl_adapter", "sdxl_unet", "sdxl_controlnet"]
    train_shards: str

    # `training_config.training == "sdxl_adapter"` specific config
    adapter_type: Optional[Literal["openpose"]] = None

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


def training_loop():
    if DIFFUSERS_UTILS_TRAINING_CONFIG not in os.environ:
        raise ValueError(f"Must set environment variable `{DIFFUSERS_UTILS_TRAINING_CONFIG}` to path to the yaml config to use for the training run.")

    training_config_filename = os.environ[DIFFUSERS_UTILS_TRAINING_CONFIG]

    if DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE in os.environ:
        training_config_override_key = os.environ[DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE]
    else:
        training_config_override_key = None

    training_config = load_training_config(training_config_filename, training_config_override_key)

    torch.cuda.set_device(device)

    dist.init_process_group("nccl")

    get_sdxl_conditioning_images = GetSDXLConditioningImages.from_training_config(training_config)

    training = SDXLTraining.from_training_config(device=device, training_config=training_config, get_sdxl_conditioning_images=get_sdxl_conditioning_images)

    dataset = get_sdxl_dataset(
        train_shards=training_config.train_shards,
        shuffle_buffer_size=training_config.shuffle_buffer_size,
        batch_size=training_config.batch_size,
        proportion_empty_prompts=training_config.proportion_empty_prompts,
        get_sdxl_conditioning_images=get_sdxl_conditioning_images,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    if dist.get_rank() == 0:
        os.makedirs(training_config.output_dir, exist_ok=True)

        wandb.init(
            name=training_config.training_run_name,
            project=training_config.project_name,
            config=training_config,
        )

    optimizer = AdamW8bit(training.parameters(), lr=training_config.learning_rate)

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    if training_config.resume_from is not None:
        optimizer.load_state_dict(safetensors.torch.load_file(os.path.join(training_config.resume_from, "optimizer.bin"), device=device))

    global_step = training_config.start_step

    progress_bar = tqdm(
        range(global_step, training_config.max_train_steps),
        disable=dist.get_rank() != 0,
    )

    dataloader = iter(dataloader)

    scaler = GradScaler(enabled=training_config.mixed_precision == torch.float16)

    while True:
        accumulated_loss = None

        nan_loss = False

        for _ in range(training_config.gradient_accumulation_steps):
            batch = next(dataloader)

            loss = training.train_step(batch)

            if torch.isnan(loss):
                logger.error("nan loss, ending training")
                nan_loss = True
                break

            loss = loss / training_config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if accumulated_loss is None:
                accumulated_loss = loss.detach()
            else:
                accumulated_loss += loss.detach()

        if nan_loss:
            break

        scaler.unscale_(optimizer)

        clip_grad_norm_(training.parameters(), 1.0)

        scaler.step(optimizer)

        lr_scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        scaler.update()

        global_step += 1

        if global_step % training_config.checkpointing_steps == 0:
            if dist.get_rank() == 0:
                save_checkpoint(
                    output_dir=training_config.output_dir,
                    checkpoints_total_limit=training_config.checkpoints_total_limit,
                    global_step=global_step,
                    optimizer=optimizer,
                )

            dist.barrier()

        if dist.get_rank() == 0 and global_step % training_config.validation_steps == 0:
            logger.info("Running validation... ")
            training_config.log_validation(
                step=global_step, num_validation_imges=training_config.num_validation_images, validation_prompts=training_config.validation_prompts, validation_images=training_config.validation_images
            )

        if dist.get_rank() == 0:
            logs = {
                "loss": accumulated_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            progress_bar.set_postfix(**logs, refresh=False)
            progress_bar.update(1)

            wandb.log(logs, step=global_step)

        if global_step % 10 == 0:
            training_config = load_training_config()

        if global_step >= training_config.max_train_steps:
            break

    dist.barrier()

    if dist.get_rank() == 0:
        training.save()


def save_checkpoint(output_dir, checkpoints_total_limit, global_step, optimizer, training):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")

    os.makedirs(save_path, exist_ok=True)

    safetensors.torch.save_file(optimizer.state_dict(), os.path.join(save_path, "optimizer.safetensors"))

    training.save(save_path)

    logger.info(f"Saved state to {save_path}")


def load_training_config(training_config_filename, training_config_override_key=None):
    with open(training_config_filename, "r") as f:
        yaml_config: Dict = yaml.safe_load(f.read())

    override_configs = yaml_config.pop("overrides", {})

    if training_config_override_key is not None:
        if training_config_override_key not in override_configs:
            raise ValueError(f"{training_config_override_key} is not one of the available overrides {override_configs.keys()}")

        yaml_config.update(override_configs[training_config_override_key])

    if "mixed_precision" not in yaml_config or yaml_config["mixed_precision"] is None or yaml_config["mixed_precision"] == "no":
        yaml_config["mixed_precision"] = None
    elif yaml_config["mixed_precision"] == "fp16":
        yaml_config["mixed_precision"] = torch.float16
    elif yaml_config["mixed_precision"] == "bf16":
        yaml_config["mixed_precision"] = torch.bfloat16
    else:
        assert False

    training_config = Config(**yaml_config)

    if training_config.training == "sdxl_adapter":
        if training_config.adapter_type is None:
            raise ValueError('must set `adapter_type` if `training` set to "sdxl_adapter"')

    if training_config.training == "sdxl_controlnet":
        if training_config.controlnet_type is None:
            raise ValueError('must set `controlnet_type` if `training` set to "sdxl_controlnet"')

    return training_config


if __name__ == "__main__":
    training_loop()
