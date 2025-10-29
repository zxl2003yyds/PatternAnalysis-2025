"""
Training script for 3D Improved UNet prostate segmentation.
Supports GPU training with mixed precision and model checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from modules import ImprovedUNet3D, CombinedLoss, dice_coefficient
from dataset import get_data_loaders


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: The neural network model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision training
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Average loss and Dice scores
    """
    model.train()
    total_loss = 0.0
    dice_scores = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels_onehot, labels_class) in enumerate(pbar):
        images = images.to(device)
        labels_onehot = labels_onehot.to(device)
        labels_class = labels_class.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels_onehot, labels_class)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate Dice coefficient
        with torch.no_grad():
            dice = dice_coefficient(outputs, labels_onehot)
            dice_scores.append(dice.cpu().numpy())
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.mean().item():.4f}'
        })
    
    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(dice_scores, axis=0)
    
    return avg_loss, avg_dice


def validate_epoch(model, loader, criterion, device, epoch):
    """
    Validate for one epoch.
    
    Args:
        model: The neural network model
        loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        Average loss and Dice scores
    """
    model.eval()
    total_loss = 0.0
    dice_scores = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for images, labels_onehot, labels_class in pbar:
            images = images.to(device)
            labels_onehot = labels_onehot.to(device)
            labels_class = labels_class.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels_onehot, labels_class)
            
            # Calculate Dice coefficient
            dice = dice_coefficient(outputs, labels_onehot)
            dice_scores.append(dice.cpu().numpy())
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.mean().item():.4f}'
            })
    
    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(dice_scores, axis=0)
    
    return avg_loss, avg_dice


def test_model(model, loader, device, class_names):
    """
    Test the model and calculate per-class Dice scores.
    
    Args:
        model: The neural network model
        loader: Test data loader
        device: Device to test on
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    dice_scores = []
    
    print("\nTesting model...")
    with torch.no_grad():
        for images, labels_onehot, labels_class in tqdm(loader, desc='Testing'):
            images = images.to(device)
            labels_onehot = labels_onehot.to(device)
            
            with autocast():
                outputs = model(images)
            
            # Calculate Dice coefficient
            dice = dice_coefficient(outputs, labels_onehot)
            dice_scores.append(dice.cpu().numpy())
    
    # Average Dice scores across all test samples
    avg_dice = np.mean(dice_scores, axis=0)
    
    # Print results
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    for i, (name, score) in enumerate(zip(class_names, avg_dice)):
        print(f"{name:20s}: {score:.4f}")
    print(f"{'Mean Dice':20s}: {avg_dice.mean():.4f}")
    print("="*50)
    
    return {
        'per_class_dice': avg_dice.tolist(),
        'mean_dice': float(avg_dice.mean()),
        'class_names': class_names
    }


def plot_training_history(history, save_path):
    """
    Plot and save training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Dice coefficient
    axes[1].plot(history['train_dice'], label='Train Dice', marker='o')
    axes[1].plot(history['val_dice'], label='Val Dice', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Training and Validation Dice')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to {save_path}")
    plt.close()


def save_checkpoint(model, optimizer, epoch, history, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        history = checkpoint['history']
        print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
        return epoch, history
    else:
        print(f"No checkpoint found at {filepath}")
        return 0, None


def main(args):
    """Main training function."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Class names
    class_names = ['Background', 'Body', 'Bone', 'Bladder', 'Prostate']
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        target_shape=tuple(args.target_shape),
        num_workers=args.num_workers,
        num_classes=args.num_classes
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = ImprovedUNet3D(
        in_channels=1,
        out_channels=args.num_classes,
        base_channels=args.base_channels,
        use_residual=True,
        deep_supervision=False
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }
    
    # Load checkpoint if resume
    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
        start_epoch, loaded_history = load_checkpoint(model, optimizer, checkpoint_path, device)
        if loaded_history:
            history = loaded_history
    
    # Training loop
    print("\nStarting training...")
    best_val_dice = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch+1
        )
        
        # Validate
        val_loss, val_dice = validate_epoch(
            model, val_loader, criterion, device, epoch+1
        )
        
        # Update learning rate
        scheduler.step(val_dice.mean())
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice.mean())
        history['val_dice'].append(val_dice.mean())
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice.mean():.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice.mean():.4f}")
        
        # Save best model
        if val_dice.mean() > best_val_dice:
            best_val_dice = val_dice.mean()
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Val Dice: {best_val_dice:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch+1, history, checkpoint_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
    save_checkpoint(model, optimizer, args.epochs, history, final_checkpoint_path)
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Test the best model
    print("\nLoading best model for testing...")
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    test_results = test_model(model, test_loader, device, class_names)
    
    # Save test results
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"\nTest results saved to {results_path}")
    
    # Save training config
    config = vars(args)
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Training config saved to {config_path}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3D Improved UNet for prostate segmentation')
    
    # Data parameters
    parser.add_argument('--data_root', type=str,
                        default=r'D:\zxl\Downloads\PatternAnalysis-2025\recognition\ImprovedUNet3D_Prostate_s48300515\Data\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1',
                        help='Root directory of the dataset')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[64, 128, 128],
                        help='Target volume shape (D H W)')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of segmentation classes')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in UNet')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency of saving checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    main(args)
