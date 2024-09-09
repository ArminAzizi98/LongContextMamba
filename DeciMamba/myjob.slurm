#!/usr/bin/env bash

# Slurm configuration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=50G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=output_16k_SPSA_div2_130m_lr0.005_rows_normal.log
#SBATCH --time=48:00:00
#CUDA_LAUNCH_BLOCKING=1
# Run the program
./eval.sh


