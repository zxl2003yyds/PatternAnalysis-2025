# Quick Setup Guide for Running on Rangpur A100

## Step 1: Transfer Files to Rangpur

Run this in PowerShell on your Windows machine:

```powershell
# Navigate to your project directory
cd "d:\zxl\Downloads\COMP3710_Final_Report\recognition"

# Transfer the entire project to Rangpur
scp -r ImprovedUNet3D_Prostate s4830051@rangpur.compute.eait.uq.edu.au:~/
```

## Step 2: SSH into Rangpur and Setup Environment

```powershell
# Connect to Rangpur
ssh s4830051@rangpur.compute.eait.uq.edu.au
```

Then on Rangpur, run:

```bash
# Create conda environment
conda create -n prostate_seg python=3.10 -y

# Activate environment
conda activate prostate_seg

# Install pip
conda install pip -y

# Install PyTorch for A100 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Install other dependencies
pip install nibabel matplotlib tqdm numpy scipy scikit-learn --no-cache-dir

# Navigate to project directory
cd ~/ImprovedUNet3D_Prostate
```

## Step 3: Test Your Setup (Optional but Recommended)

```bash
# Submit test job to verify GPU and environment
sbatch test_job.sh

# Check job status
squeue -u s4830051

# View test results (wait for job to complete)
cat test_setup_*.out
```

## Step 4: Submit Training Job

```bash
# Make sure you're in the project directory
cd ~/ImprovedUNet3D_Prostate

# Submit training job
sbatch train_job.sh

# Monitor job status
squeue -u s4830051

# Watch training progress (replace JOBID with actual number from squeue)
tail -f train_JOBID.out

# Or check GPU usage in real-time
watch -n 5 nvidia-smi
```

## Step 5: Submit Prediction Job (After Training)

```bash
# Once training completes, generate predictions
sbatch predict_job.sh

# Monitor prediction job
squeue -u s4830051
```

## Step 6: Transfer Results Back to Windows

Run this in PowerShell on your Windows machine:

```powershell
# Transfer all A100 outputs back to your local machine
scp -r s4830051@rangpur.compute.eait.uq.edu.au:~/ImprovedUNet3D_Prostate/outputs_a100 "d:\zxl\Downloads\"

# View training history
Start-Process "d:\zxl\Downloads\outputs_a100\training_history.png"

# View predictions
Start-Process "d:\zxl\Downloads\outputs_a100\predictions\prediction_sample_0_single.png"
```

## Useful Commands

### Monitor Jobs
```bash
# Check your job queue
squeue -u s4830051

# Check detailed job info
scontrol show job JOBID

# View past jobs
sacct -u s4830051

# Cancel a job
scancel JOBID
```

### View Outputs
```bash
# List all output files
ls -lh *.out *.err

# View training output
cat train_*.out

# Follow training in real-time
tail -f train_*.out

# Check for errors
cat train_*.err
```

### Check GPU Resources
```bash
# See available GPUs
sinfo -p gpu

# Check GPU usage
nvidia-smi
```

## Expected Results with A100

- **Training Time**: ~1-2 hours (vs 22 minutes on RTX 3060, but MUCH better quality)
- **Resolution**: 64×128×128 (vs 48×96×96)
- **Batch Size**: 4 (vs 1)
- **Expected Prostate Dice**: ~0.84-0.85 (vs 0.83)
- **Visual Quality**: Much smoother, less blocky predictions

## Troubleshooting

### If job fails immediately:
```bash
# Check error file
cat train_*.err

# Check if conda environment exists
conda env list

# Reactivate and test manually
conda activate prostate_seg
python test_setup.py
```

### If out of memory:
Edit `train_job.sh` and reduce batch size:
```bash
vim train_job.sh
# Change --batch_size 4 to --batch_size 2
```

### If dataset not found:
Make sure your dataset is accessible on Rangpur. You may need to transfer it:
```powershell
# From Windows
scp -r "path\to\HipMRI_study_complete_release_v1" s4830051@rangpur.compute.eait.uq.edu.au:~/
```

## Quick Command Summary

```bash
# Full workflow on Rangpur
cd ~/ImprovedUNet3D_Prostate
sbatch test_job.sh          # Test setup (optional)
sbatch train_job.sh         # Train model
squeue -u s4830051          # Check status
tail -f train_*.out         # Monitor training
sbatch predict_job.sh       # Generate predictions (after training)
```

## Notes

- All job scripts are configured with email notifications to s4830051@student.uq.edu.au
- Training output saved to `train_JOBID.out`
- Errors saved to `train_JOBID.err`
- Model checkpoints saved to `./outputs_a100/`
- Predictions saved to `./outputs_a100/predictions/`
