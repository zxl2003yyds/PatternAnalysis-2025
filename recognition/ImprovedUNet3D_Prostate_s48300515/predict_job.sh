#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --qos=comp3710
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100
#SBATCH --job-name=prostate_pred
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=s4830051@student.uq.edu.au
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err

# Activate conda environment
conda activate prostate_seg

# Run predictions
python predict.py --data_root /home/groups/comp3710/HipMRI_Study_open \
                  --target_shape 144 288 288 \
                  --model_path ./outputs_a100_fullres/best_model.pth \
                  --num_samples 6 \
                  --output_dir ./outputs_a100_fullres/predictions
