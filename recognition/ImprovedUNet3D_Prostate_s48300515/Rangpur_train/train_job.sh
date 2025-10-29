#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --qos=comp3710
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100
#SBATCH --job-name=prostate_seg
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=s4830051@student.uq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Activate conda environment
conda activate prostate_seg

# Run training with A100-optimized settings using Rangpur dataset - FULL RESOLUTION
python train.py --data_root /home/groups/comp3710/HipMRI_Study_open \
                --batch_size 1 \
                --epochs 150 \
                --learning_rate 0.001 \
                --target_shape 144 288 288 \
                --base_channels 32 \
                --num_workers 4 \
                --output_dir ./outputs_a100_fullres \
                --save_freq 10
