"""
Prediction script for 3D Improved UNet prostate segmentation.
Loads trained model and performs inference on test data with visualizations.
"""

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

from modules import ImprovedUNet3D
from dataset import ProstateDataset3D, normalize_volume


def load_model(model_path, device, num_classes=5, base_channels=32):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model weights
        device: Device to load model on
        num_classes: Number of output classes
        base_channels: Base number of channels
    
    Returns:
        Loaded model
    """
    model = ImprovedUNet3D(
        in_channels=1,
        out_channels=num_classes,
        base_channels=base_channels,
        use_residual=True,
        deep_supervision=False
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def predict_volume(model, volume, device, target_shape=(64, 128, 128)):
    """
    Predict segmentation for a single volume.
    
    Args:
        model: Trained model
        volume: Input volume (numpy array)
        device: Device to run prediction on
        target_shape: Target shape for model input
    
    Returns:
        Predicted segmentation (numpy array)
    """
    import torch.nn.functional as F
    
    # Normalize volume
    volume = normalize_volume(volume)
    
    # Resize to target shape
    original_shape = volume.shape
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    volume_resized = F.interpolate(volume_tensor, size=target_shape, mode='trilinear', align_corners=False)
    
    # Move to device
    volume_resized = volume_resized.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(volume_resized)
        prediction = torch.argmax(output, dim=1)  # (1, D, H, W)
    
    # Resize back to original shape
    prediction = prediction.unsqueeze(1).float()  # (1, 1, D, H, W)
    prediction = F.interpolate(prediction, size=original_shape, mode='nearest')
    prediction = prediction.squeeze().cpu().numpy().astype(np.uint8)
    
    return prediction


def visualize_slice(image, label, prediction, slice_idx, class_names, save_path=None):
    """
    Visualize a single slice with image, ground truth, and prediction.
    
    Args:
        image: Input image volume
        label: Ground truth segmentation
        prediction: Predicted segmentation
        slice_idx: Index of slice to visualize
        class_names: List of class names
        save_path: Path to save visualization
    """
    # Define colors for each class
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan']
    cmap = ListedColormap(colors[:len(class_names)])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image[:, :, slice_idx], cmap='gray')
    axes[0].set_title('Input MRI')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image[:, :, slice_idx], cmap='gray')
    axes[1].imshow(label[:, :, slice_idx], cmap=cmap, alpha=0.5, vmin=0, vmax=len(class_names)-1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(image[:, :, slice_idx], cmap='gray')
    axes[2].imshow(prediction[:, :, slice_idx], cmap=cmap, alpha=0.5, vmin=0, vmax=len(class_names)-1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=class_names[i]) 
                      for i in range(len(class_names))]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05), ncol=len(class_names))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_3d_comparison(image, label, prediction, slice_indices, class_names, save_path=None):
    """
    Visualize multiple slices in a grid.
    
    Args:
        image: Input image volume
        label: Ground truth segmentation
        prediction: Predicted segmentation
        slice_indices: List of slice indices to visualize
        class_names: List of class names
        save_path: Path to save visualization
    """
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan']
    cmap = ListedColormap(colors[:len(class_names)])
    
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(3, n_slices, figsize=(4*n_slices, 12))
    
    if n_slices == 1:
        axes = axes.reshape(-1, 1)
    
    for i, slice_idx in enumerate(slice_indices):
        # Original image
        axes[0, i].imshow(image[:, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f'Slice {slice_idx}')
        axes[0, i].axis('off')
        
        # Ground truth
        axes[1, i].imshow(image[:, :, slice_idx], cmap='gray')
        axes[1, i].imshow(label[:, :, slice_idx], cmap=cmap, alpha=0.5, vmin=0, vmax=len(class_names)-1)
        axes[1, i].axis('off')
        
        # Prediction
        axes[2, i].imshow(image[:, :, slice_idx], cmap='gray')
        axes[2, i].imshow(prediction[:, :, slice_idx], cmap=cmap, alpha=0.5, vmin=0, vmax=len(class_names)-1)
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Input', fontsize=14, rotation=0, labelpad=40, va='center')
    axes[1, 0].set_ylabel('Ground Truth', fontsize=14, rotation=0, labelpad=40, va='center')
    axes[2, 0].set_ylabel('Prediction', fontsize=14, rotation=0, labelpad=40, va='center')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=class_names[i]) 
                      for i in range(len(class_names))]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.02), ncol=len(class_names))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_dice_per_class(prediction, label, num_classes):
    """
    Calculate Dice coefficient for each class.
    
    Args:
        prediction: Predicted segmentation
        label: Ground truth segmentation
        num_classes: Number of classes
    
    Returns:
        Dice scores per class
    """
    dice_scores = []
    
    for c in range(num_classes):
        pred_c = (prediction == c)
        label_c = (label == c)
        
        intersection = np.logical_and(pred_c, label_c).sum()
        union = pred_c.sum() + label_c.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / union
        else:
            dice = 1.0  # Perfect score if both are empty
        
        dice_scores.append(dice)
    
    return np.array(dice_scores)


def main(args):
    """Main prediction function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Class names
    class_names = ['Background', 'Body', 'Bone', 'Bladder', 'Prostate']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path, device, args.num_classes, args.base_channels)
    
    # Load dataset
    print("\nLoading test data...")
    test_dataset = ProstateDataset3D(
        data_root=args.data_root,
        split='test',
        target_shape=tuple(args.target_shape),
        num_classes=args.num_classes
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Select samples to visualize
    num_samples = min(args.num_samples, len(test_dataset))
    sample_indices = np.linspace(0, len(test_dataset)-1, num_samples, dtype=int)
    
    print(f"\nGenerating predictions for {num_samples} samples...")
    
    all_dice_scores = []
    
    for i, idx in enumerate(sample_indices):
        print(f"\nProcessing sample {i+1}/{num_samples} (index {idx})...")
        
        # Load sample
        image, label_onehot, label_class = test_dataset[idx]
        
        # Move to device and add batch dimension
        image_input = image.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_input)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Convert to numpy for visualization
        image_np = image.squeeze().cpu().numpy()
        label_np = label_class.cpu().numpy()
        
        # Calculate Dice scores
        dice_scores = calculate_dice_per_class(prediction, label_np, args.num_classes)
        all_dice_scores.append(dice_scores)
        
        print("Dice scores:")
        for name, score in zip(class_names, dice_scores):
            print(f"  {name:20s}: {score:.4f}")
        print(f"  {'Mean':20s}: {dice_scores.mean():.4f}")
        
        # Select slices to visualize (uniformly spaced through volume)
        depth = image_np.shape[2]
        slice_indices = [depth//4, depth//2, 3*depth//4]
        
        # Visualize multiple slices
        save_path = os.path.join(args.output_dir, f'prediction_sample_{idx}_grid.png')
        visualize_3d_comparison(image_np, label_np, prediction, slice_indices, class_names, save_path)
        
        # Also save individual slice
        save_path_single = os.path.join(args.output_dir, f'prediction_sample_{idx}_single.png')
        visualize_slice(image_np, label_np, prediction, depth//2, class_names, save_path_single)
    
    # Calculate average Dice scores across all samples
    avg_dice = np.mean(all_dice_scores, axis=0)
    
    print("\n" + "="*60)
    print("Average Dice Scores Across All Samples:")
    print("="*60)
    for name, score in zip(class_names, avg_dice):
        print(f"{name:20s}: {score:.4f}")
    print(f"{'Mean':20s}: {avg_dice.mean():.4f}")
    print("="*60)
    
    # Save results
    results_file = os.path.join(args.output_dir, 'prediction_results.txt')
    with open(results_file, 'w') as f:
        f.write("Average Dice Scores:\n")
        f.write("="*60 + "\n")
        for name, score in zip(class_names, avg_dice):
            f.write(f"{name:20s}: {score:.4f}\n")
        f.write(f"{'Mean':20s}: {avg_dice.mean():.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"\nResults saved to {results_file}")
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using trained 3D Improved UNet')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='./outputs/best_model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--data_root', type=str,
                        default=r'D:\zxl\Downloads\COMP3710_Final_Report\Data\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1',
                        help='Root directory of the dataset')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[64, 128, 128],
                        help='Target volume shape (D H W)')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of segmentation classes')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in UNet')
    
    # Prediction parameters
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save predictions')
    
    args = parser.parse_args()
    
    main(args)
