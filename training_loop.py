import logging
import os
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List

import diffusers
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .sdxl_t2i_openpose import SDXLT2IOpenpose

torch.backends.cuda.matmul.allow_tf32 = True

logger = get_logger(__name__)


@dataclass
class Config:
    output_dir: str
    gradient_accumulation_steps: int
    mixed_precision: str
    resume_from: str
    max_grad_norm: float
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
    training_spec_class: object


def main():
    args = ArgumentParser()

    args.add_argument("--config_path")

    args = args.parse_args()

    config = load_config(args.config_path)

    training_loop(config)


def load_config(config_path):
    config: Config = yaml.safe_load(config_path)

    if config.training_spec_class == "SDXLT2IOpenpose":
        config.training_spec_class = SDXLT2IOpenpose
    else:
        raise ValueError(f"unknown training_spec_class: {config.training_spec_class}")

    return config


def training_loop(config):
    os.makedirs(config.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="wandb",
        project_dir=config.output_dir,
        split_batches=True,
    )

    accelerator.init_trackers("t2iadapter")

    accelerator.register_save_state_pre_hook(config.training_spec_class.save_model_hook)
    accelerator.register_load_state_pre_hook(config.training_spec_class.load_model_hook)

    training_spec = config.training_spec_class(config, accelerator)

    optimizer = AdamW(training_spec.training_parameters(), lr=1e-5)

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config.resume_from is not None:
        accelerator.load_state(config.resume_from)

    dataloader = DataLoader(
        training_spec.get_dataset(),
        batch_size=None,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    for batch in dataloader:
        with accelerator.accumulate(training_spec.training_model()):
            loss = training_spec.train_step(batch, accelerator)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    training_spec.training_parameters(), config.max_grad_norm
                )

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % config.checkpointing_steps == 0:
                        checkpoint(
                            output_dir=config.output_dir,
                            checkpoints_total_limit=config.checkpoints_total_limit,
                            global_step=global_step,
                            accelerator=accelerator,
                        )

                    if global_step % config.validation_steps == 0:
                        logger.info("Running validation... ")
                        training_spec.log_validation()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config.max_train_steps:
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        training_spec.save_results()

    accelerator.end_training()


def checkpoint(output_dir, checkpoints_total_limit, global_step, accelerator):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    main()
