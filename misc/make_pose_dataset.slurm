#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output /fsx/william/logs/make_pose_dataset.log

set -e

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/diffusers-utils

time srun --wait=60 \
    python misc/make_pose_dataset.py --slurm --start_shard 0 --end_shard 600 --pose_algorithm dwpose
