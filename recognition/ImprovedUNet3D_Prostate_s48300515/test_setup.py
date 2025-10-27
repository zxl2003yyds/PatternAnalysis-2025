"""
Quick test script to verify the model, dataset, and GPU setup.
Run this before starting full training to check everything works.
"""

import torch
import sys
import os

print("="*60)
print("3D Improved UNet - System Check")
print("="*60)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available. Training will be very slow on CPU!")

# Check imports
print("\nChecking package imports...")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not found")

try:
    import nibabel as nib
    print(f"✓ NiBabel {nib.__version__}")
except ImportError:
    print("✗ NiBabel not found - install with: pip install nibabel")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ Matplotlib not found")

try:
    from tqdm import tqdm
    print(f"✓ tqdm")
except ImportError:
    print("✗ tqdm not found")

# Test model creation
print("\nTesting model creation...")
try:
    from modules import ImprovedUNet3D
    model = ImprovedUNet3D(in_channels=1, out_channels=5, base_channels=16)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  Parameters: {total_params:,}")
    
    # Test forward pass on CPU
    dummy_input = torch.randn(1, 1, 32, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Output shape: {output.shape}")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ GPU forward pass successful")
        
except Exception as e:
    print(f"✗ Model test failed: {e}")

# Test dataset
print("\nTesting dataset...")
try:
    from dataset import ProstateDataset3D
    
    data_root = r'D:\zxl\Downloads\COMP3710_Final_Report\Data\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1'
    
    if os.path.exists(data_root):
        print(f"✓ Data directory found: {data_root}")
        
        # Check for images
        image_dir = os.path.join(data_root, 'semantic_MRs_anon')
        label_dir = os.path.join(data_root, 'semantic_labels_anon')
        
        if os.path.exists(image_dir):
            num_images = len([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
            print(f"✓ Found {num_images} images")
        else:
            print(f"✗ Image directory not found: {image_dir}")
        
        if os.path.exists(label_dir):
            num_labels = len([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])
            print(f"✓ Found {num_labels} labels")
        else:
            print(f"✗ Label directory not found: {label_dir}")
        
        # Try loading dataset
        print("\nTrying to load dataset (this may take a moment)...")
        dataset = ProstateDataset3D(
            data_root=data_root,
            split='train',
            target_shape=(32, 64, 64)  # Small size for quick test
        )
        print(f"✓ Dataset loaded: {len(dataset)} volumes")
        
        # Try loading one sample
        image, label_onehot, label_class = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Image shape: {image.shape}")
        print(f"  Label (one-hot) shape: {label_onehot.shape}")
        print(f"  Label (class) shape: {label_class.shape}")
        print(f"  Unique labels: {torch.unique(label_class).tolist()}")
        
    else:
        print(f"✗ Data directory not found: {data_root}")
        print("  Please update the path in this script")
        
except Exception as e:
    print(f"✗ Dataset test failed: {e}")
    import traceback
    traceback.print_exc()

# Memory check
if torch.cuda.is_available():
    print("\nGPU Memory Check...")
    torch.cuda.empty_cache()
    memory_allocated = torch.cuda.memory_allocated(0) / 1e9
    memory_reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"  Allocated: {memory_allocated:.2f} GB")
    print(f"  Reserved: {memory_reserved:.2f} GB")

print("\n" + "="*60)
print("System check complete!")
print("="*60)

# Recommendations
print("\nRecommendations:")
if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_memory < 8:
        print("⚠ GPU memory < 8 GB: Use --batch_size 1 and --target_shape 48 96 96")
    elif total_memory < 12:
        print("⚠ GPU memory < 12 GB: Use --batch_size 1-2")
    else:
        print("✓ Sufficient GPU memory for default settings")
else:
    print("⚠ No GPU detected. Training will be very slow!")

print("\nTo start training, run:")
print("  python train.py --batch_size 2 --epochs 100")
print("\nTo run quick training test (5 epochs):")
print("  python train.py --batch_size 1 --epochs 5 --target_shape 32 64 64")
