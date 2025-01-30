#!/usr/bin/env bash

# Slurm configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=output.log
#SBATCH --time=72:00:00
#CUDA_LAUNCH_BLOCKING=1
# Run the program
./train.sh


