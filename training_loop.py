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
import yaml
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

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
class TrainingConfig:
    output_dir: str
    learning_rate: float = 0.00001
    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[torch.dtype] = None
    max_train_steps: int = 30_000
    resume_from: Optional[str] = None
    start_step: int = 0

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


def main():
    from sdxl import (
        SDXLTraining, get_adapter_openpose_conditioning_image,
        get_controlnet_canny_conditioning_image,
        get_controlnet_inpainting_conditioning_image,
        get_controlnet_pre_encoded_controlnet_inpainting_conditioning_image,
        get_sdxl_dataset)
    from sdxl_models import (SDXLAdapter, SDXLControlNet, SDXLControlNetFull,
                             SDXLControlNetPreEncodedControlnetCond)

    torch.cuda.set_device(device)

    dist.init_process_group("nccl")

    @dataclass
    class SDXLTrainingConfig:
        training: Literal["sdxl_adapter", "sdxl_unet", "sdxl_controlnet"]
        adapter_type: Optional[Literal["openpose"]]
        controlnet_type: Optional[Literal["canny", "inpainting"]]
        controlnet_variant: Optional[Literal["default", "full", "pre_encoded_controlnet_cond"]]
        controlnet_train_base_unet: bool = False
        mixed_precision: Optional[torch.dtype]
        resume_from: Optional[str] = None
        train_shards: str
        shuffle_buffer_size: int = 1000
        proportion_empty_prompts: float = 0.1
        batch_size: int = 8

    config = load_config(SDXLTrainingConfig)

    if config.training == "sdxl_adapter":
        if config.adapter_type is None:
            raise ValueError('must set `adapter_type` if `training` set to "sdxl_adapter"')

    if config.training == "sdxl_controlnet":
        if config.controlnet_type is None:
            raise ValueError('must set `controlnet_type` if `training` set to "sdxl_controlnet"')

    if config.adapter_type == "openpose":
        from controlnet_aux import OpenposeDetector

        open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        get_sdxl_conditioning_images = lambda *args, **kwargs: get_adapter_openpose_conditioning_image(*args, **kwargs, open_pose=open_pose)
    elif config.controlnet_type == "canny":
        get_sdxl_conditioning_images = get_controlnet_canny_conditioning_image
    elif config.controlnet_type == "inpainting":
        if config.controlnet_variant == "pre_encoded_controlnet_cond":
            get_sdxl_conditioning_images = get_controlnet_pre_encoded_controlnet_inpainting_conditioning_image
        else:
            get_sdxl_conditioning_images = get_controlnet_inpainting_conditioning_image
    else:
        assert False

    dataset = get_sdxl_dataset(
        train_shards=config.train_shards,
        shuffle_buffer_size=config.shuffle_buffer_size,
        batch_size=config.batch_size,
        proportion_empty_prompts=config.proportion_empty_prompts,
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

    if config.training == "sdxl_controlnet":
        if config.controlnet_variant == "default":
            controlnet_cls = SDXLControlNet
        elif config.controlnet_variant == "full":
            controlnet_cls = SDXLControlNetFull
        elif config.controlnet_variant == "pre_encoded_controlnet_cond":
            controlnet_cls = SDXLControlNetPreEncodedControlnetCond
        else:
            assert False
    else:
        controlnet_cls = None

    if config.training == "sdxl_adapter":
        adapter_cls = SDXLAdapter
    else:
        adapter_cls = None

    if config.training == "sdxl_adapter":
        timestep_sampling = "cubic"
    else:
        timestep_sampling = "uniform"

    if config.training == "sdxl_controlnet" and config.controlnet_type == "inpainting":
        log_validation_input_images_every_time = True
    else:
        log_validation_input_images_every_time = False

    training = SDXLTraining(
        device=device,
        train_unet=config.training == "sdxl_unet",
        train_unet_up_blocks=config.training == "sdxl_controlnet" and config.controlnet_train_base_unet,
        unet_resume_from=config.resume_from is not None and os.path.join(config.resume_from, "unet.safetensors"),
        controlnet_cls=controlnet_cls,
        adapter_cls=adapter_cls,
        adapter_resume_from=config.resume_from is not None and os.path.join(config.resume_from, "adapter.safetensors"),
        timestep_sampling=timestep_sampling,
        log_validation_input_images_every_time=log_validation_input_images_every_time,
        get_sdxl_conditioning_images=get_sdxl_conditioning_images,
        mixed_precision=config.mixed_precision,
    )

    training_loop(training, dataloader)


def training_loop(training, dataloader):
    training_config = load_config(TrainingConfig)

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
            training_config = load_config(TrainingConfig)

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


def load_config(config_cls):
    if "DIFFUSERS_UTILS_TRAINING_CONFIG" not in os.environ:
        raise ValueError(f"Must set environment variable `'DIFFUSERS_UTILS_TRAINING_CONFIG'` to path to the yaml config to use for the training run.")

    with open(os.environ["DIFFUSERS_UTILS_TRAINING_CONFIG"], "r") as f:
        yaml_config: Dict = yaml.safe_load(f.read())

    override_configs = yaml_config.pop("overrides", {})

    training_config_override_key = os.environ.get("DIFFUSERS_UTILS_TRAINING_CONFIG_OVERRIDE", None)

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

    yaml_config_ = {}

    for field in dataclasses.fields(config_cls):
        yaml_config_[field.name] = yaml_config[field.name]

    training_config = config_cls(**yaml_config_)

    return training_config


if __name__ == "__main__":
    training_loop()
