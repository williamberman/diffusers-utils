import os
import time
from argparse import ArgumentParser
from logging import getLogger

import numpy as np
from torch.utils.data import DataLoader

import webdataset as wds
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import draw_poses
from controlnet_aux.util import HWC3
from PIL import Image


logger = getLogger(__name__)


def main():
    args = ArgumentParser()
    args.add_argument(
        "--start_shard",
        type=int,
        help="The starting shard to pre-encode.",
        required=True,
    )
    args.add_argument(
        "--end_shard",
        type=int,
        help="The ending shard to pre-encode, inclusive. If not given, defaults to `--start_shard`.",
        required=False,
    )
    args.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "If set, this process is running under a batch of slurm tasks."
            "`--start_shard` and `--end_shard` must be set for the entirety of shards over all slurm tasks."
            " The shards that will be encoded in each instance of the task will be determined via"
            " the env vars `$SLURM_NTASKS` and `$SLURM_PROCID`."
        ),
    )
    args = args.parse_args()

    dataset = "s3://muse-datasets/laion-aesthetic6plus-min512-data"
    upload_to = "s3://muse-datasets/laion-aesthetic6plus-min512-data-openpose"

    logger.warning("********************")
    logger.warning("Pre-encoding dataset")
    logger.warning(f"dataset: {dataset}")
    logger.warning(f"start_shard: {args.start_shard}")
    logger.warning(f"end_shard: {args.end_shard}")
    logger.warning(f"upload_to: {upload_to}")
    logger.warning("********************")

    if args.slurm:
        slurm_procid = int(os.environ["SLURM_PROCID"])
        slurm_ntasks = int(os.environ["SLURM_NTASKS"])

        distributed_shards = distribute_shards(args.start_shard, args.end_shard, slurm_ntasks)

        start_shard_task, end_shard_task = distributed_shards[slurm_procid]

        args.start_shard = start_shard_task
        args.end_shard = end_shard_task

        logger.warning("************")
        logger.warning("Running as slurm task")
        logger.warning(f"SLURM_NTASKS: {slurm_ntasks}")
        logger.warning(f"SLURM_PROCID: {slurm_procid}")
        logger.warning(f"start_shard: {start_shard_task}, end_shard: {end_shard_task}")
        logger.warning("************")
        logger.warning(f"all slurm processes")
        for slurm_proc_id_, (start_shard, end_shard) in enumerate(distributed_shards):
            logger.warning(f"slurm process: {slurm_proc_id_}, start_shard: {start_shard}, end_shard: {end_shard}")
        logger.warning("************")

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose.to(f"cuda")

    for shard in range(args.start_shard, args.end_shard + 1):
        download_command = f"pipe:aws s3 cp {dataset}/{format_shard_number(shard)}.tar -"
        upload_command = f"pipe:aws s3 cp - {upload_to}/{format_shard_number(shard)}.tar"

        logger.warning(f"download {download_command}")
        logger.warning(f"upload {upload_command}")

        src = (
            wds.WebDataset(download_command, handler=wds.warn_and_continue)
            .decode("pil", handler=wds.warn_and_continue)
            .rename(image="jpg;png;jpeg;webp", prompt="text;txt;caption", metadata="json")
            .map(
                lambda dict: {
                    "__key__": dict["__key__"],
                    "__url__": dict["__url__"],
                    "image": dict["image"],
                    "prompt": dict["prompt"],
                    "metadata": dict["metadata"],
                }
            )
            .to_tuple("__key__", "__url__", "image", "prompt", "metadata")
        )
        src = DataLoader(
            src,
            batch_size=None,
            shuffle=False,
            num_workers=0,
        )

        writer = wds.TarWriter(upload_command)

        ctr = 0

        for __key__, __url__, image, prompt, metadata in src:
            t0 = time.perf_counter()

            openpose_image = run_openpose(openpose, image)

            logger.warning(f"shard {shard}: {time.perf_counter() - t0}")

            if openpose_image is None:
                continue

            sample = {
                "__key__": __key__,
                "png": image,
                "openpose.png": openpose_image,
                "txt": prompt,
                "json": metadata,
            }

            writer.write(sample)

            ctr += 1
            print(f"shard: {shard} uploaded: {ctr}")

        writer.close()


def distribute_shards(start_shard_all, end_shard_all, slurm_ntasks):
    total_shards = end_shard_all - start_shard_all + 1
    shards_per_task = total_shards // slurm_ntasks
    shards_per_task = [shards_per_task] * slurm_ntasks

    # to distribute the remainder of tasks for non-evenly divisible number of shards
    left_over_shards = total_shards % slurm_ntasks

    for slurm_procid in range(left_over_shards):
        shards_per_task[slurm_procid] += 1

    assert sum(shards_per_task) == total_shards

    distributed_shards = []

    for slurm_procid in range(len(shards_per_task)):
        if slurm_procid == 0:
            start_shard = start_shard_all
        else:
            start_shard = distributed_shards[slurm_procid - 1][1] + 1

        end_shard = start_shard + shards_per_task[slurm_procid] - 1
        distributed_shards.append((start_shard, end_shard))

    assert sum([end_shard - start_shard + 1 for start_shard, end_shard in distributed_shards]) == total_shards

    return distributed_shards


def run_openpose(openpose, input_image):
    input_image = np.array(input_image, dtype=np.uint8)
    input_image = HWC3(input_image)
    H, W, C = input_image.shape

    poses = openpose.detect_poses(input_image, include_hand=False, include_face=False)

    if len(poses) == 0:
        return None

    canvas = draw_poses(poses, H, W, draw_body=True, draw_hand=False, draw_face=False)

    detected_map = canvas
    detected_map = HWC3(detected_map)
    detected_map = Image.fromarray(detected_map)

    return detected_map


def format_shard_number(shard_n: int):
    return "{:0>{}}".format(shard_n, 5)


if __name__ == "__main__":
    main()
