import logging
import os
import shutil
from logging import getLogger

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from training_config import training_config

torch.backends.cuda.matmul.allow_tf32 = True

logger = getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main():
    dist.init_process_group("nccl")

    if dist.get_rank() == 0:
        os.makedirs(training_config.output_dir, exist_ok=True)

        wandb.init()

    if training_config.training == "sdxl_unet":
        from sdxl import init_sdxl_unet

        init_sdxl_unet()

        from sdxl import sdxl_log_unet_validation, sdxl_unet_train_step, unet
        from sdxl_dataset import get_sdxl_unet_dataset

        training_parameters = unet.parameters()
        parameters_to_clip = unet.parameters()
        dataset = get_sdxl_unet_dataset()
        log_validation = sdxl_log_unet_validation
        train_step = sdxl_unet_train_step
    elif training_config.training == "sdxl_adapter":
        from sdxl import init_sdxl_adapter

        init_sdxl_adapter()

        from sdxl import (adapter, sdxl_adapter_train_step,
                          sdxl_log_adapter_validation)
        from sdxl_dataset import get_sdxl_adapter_dataset

        training_parameters = adapter.parameters()
        parameters_to_clip = adapter.parameters()
        dataset = get_sdxl_adapter_dataset()
        log_validation = sdxl_log_adapter_validation
        train_step = sdxl_adapter_train_step
    else:
        assert False

    training_loop(
        training_parameters=training_parameters,
        parameters_to_clip=parameters_to_clip,
        dataset=dataset,
        log_validation=log_validation,
        train_step=train_step,
    )

    dist.barrier()

    if dist.get_rank() == 0:
        if training_config.training == "sdxl_unet":
            unet.module.save_pretrained(training_config.output_dir)
        elif training_config.training == "sdxl_adapter":
            adapter.module.save_pretrained(training_config.output_dir)
        else:
            assert False


def training_loop(
    training_parameters, parameters_to_clip, dataset, log_validation, train_step
):
    optimizer = AdamW(training_parameters, lr=1e-5)

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    if training_config.resume_from is not None:
        load_checkpoint(training_config.resume_from, optimizer=optimizer)

    global_step = 0

    progress_bar = tqdm(
        range(0, training_config.max_train_steps),
        disable=dist.get_rank() != 0,
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

    dataloader = iter(dataloader)

    while True:
        accumulated_loss = None

        for _ in range(training_config.gradient_accumulation_steps):
            batch = next(dataloader)

            loss = train_step(batch)

            loss = loss / training_config.gradient_accumulation_steps
            loss.backward()

            if accumulated_loss is None:
                accumulated_loss = loss.detach()
            else:
                accumulated_loss += loss.detach()

        clip_grad_norm_(parameters_to_clip, 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        progress_bar.update(1)
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
            log_validation(global_step)

        logs = {"loss": accumulated_loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        wandb.log(logs, step=global_step)

        if global_step >= training_config.max_train_steps:
            break


def save_checkpoint(output_dir, checkpoints_total_limit, global_step, optimizer):
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

    if training_config.training == "sdxl_adapter":
        from sdxl import adapter

        adapter.module.save_pretrained(output_dir, subfolder="adapter")
    else:
        assert False

    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))

    logger.info(f"Saved state to {save_path}")


def load_checkpoint(resume_from, optimizer):
    optimizer.load(os.path.join(resume_from, "optimizer.bin"))

    if training_config.training == "sdxl_adapter":
        from sdxl import adapter

        adapter_state_dict = torch.load(
            os.path.join(resume_from, "adapter", "pytorch_model.bin"),
            map_location=adapter.device,
        )
        adapter.load_state_dict(adapter_state_dict)
    else:
        assert False


if __name__ == "__main__":
    main()
