#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Prostate Segmentation A100 Setup${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Create conda environment
echo -e "\n${YELLOW}Step 1: Creating conda environment 'prostate_seg'...${NC}"
conda create -n prostate_seg python=3.10 -y

# Step 2: Activate environment
echo -e "\n${YELLOW}Step 2: Activating environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate prostate_seg

# Step 3: Install pip
echo -e "\n${YELLOW}Step 3: Installing pip...${NC}"
conda install pip -y

# Step 4: Install PyTorch
echo -e "\n${YELLOW}Step 4: Installing PyTorch for A100 (CUDA 11.8)...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Step 5: Install dependencies
echo -e "\n${YELLOW}Step 5: Installing other dependencies...${NC}"
pip install nibabel matplotlib tqdm numpy scipy scikit-learn --no-cache-dir

# Step 6: Navigate to project
echo -e "\n${YELLOW}Step 6: Navigating to project directory...${NC}"
cd ~/ImprovedUNet3D_Prostate

# Verify installation
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${BLUE}To verify setup, run:${NC}"
echo -e "  sbatch test_job.sh"
echo -e "\n${BLUE}To start training, run:${NC}"
echo -e "  sbatch train_job.sh"
echo -e "\n${BLUE}To check job status:${NC}"
echo -e "  squeue -u \$USER"
echo -e "\n${BLUE}To monitor training:${NC}"
echo -e "  tail -f train_*.out"
