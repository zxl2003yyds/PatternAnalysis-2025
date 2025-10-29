#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --qos=comp3710
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=test_setup
#SBATCH --time=0-00:05:00
#SBATCH --output=test_setup_%j.out
#SBATCH --error=test_setup_%j.err

# Activate conda environment
conda activate prostate_seg

# Run setup test
python test_setup.py
