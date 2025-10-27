"""
Dataset loader for 3D prostate MRI segmentation.
Loads NIfTI files and prepares them for training.
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob


def to_channels(arr, num_classes=5, dtype=np.uint8):
    """
    Convert label array to one-hot encoded format.
    
    Args:
        arr: Input array with class labels
        num_classes: Number of classes
        dtype: Data type for output
    
    Returns:
        One-hot encoded array of shape (..., num_classes)
    """
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in range(num_classes):
        res[..., c] = (arr == c).astype(dtype)
    return res


def normalize_volume(volume):
    """
    Normalize 3D volume using z-score normalization.
    
    Args:
        volume: Input 3D volume
    
    Returns:
        Normalized volume
    """
    mean = volume.mean()
    std = volume.std()
    if std > 0:
        volume = (volume - mean) / std
    return volume


def load_nifti_volume(filepath, normalize=True):
    """
    Load a single NIfTI file.
    
    Args:
        filepath: Path to .nii.gz file
        normalize: Whether to normalize the volume
    
    Returns:
        Numpy array of the volume
    """
    nifti_img = nib.load(filepath)
    volume = nifti_img.get_fdata(caching='unchanged')
    
    # Remove extra dimensions if present
    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]
    
    if normalize:
        volume = normalize_volume(volume)
    
    return volume, nifti_img.affine


def pad_or_crop_volume(volume, target_shape):
    """
    Pad or crop volume to target shape.
    
    Args:
        volume: Input volume
        target_shape: Target shape (D, H, W)
    
    Returns:
        Processed volume
    """
    current_shape = volume.shape
    
    # Calculate padding/cropping for each dimension
    processed = volume
    for i in range(3):
        if current_shape[i] < target_shape[i]:
            # Pad
            pad_total = target_shape[i] - current_shape[i]
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            
            pad_width = [(0, 0)] * 3
            pad_width[i] = (pad_before, pad_after)
            processed = np.pad(processed, pad_width, mode='constant', constant_values=0)
        elif current_shape[i] > target_shape[i]:
            # Crop
            crop_total = current_shape[i] - target_shape[i]
            crop_before = crop_total // 2
            crop_after = crop_before + target_shape[i]
            
            if i == 0:
                processed = processed[crop_before:crop_after, :, :]
            elif i == 1:
                processed = processed[:, crop_before:crop_after, :]
            else:
                processed = processed[:, :, crop_before:crop_after]
    
    return processed


class ProstateDataset3D(Dataset):
    """
    PyTorch Dataset for 3D prostate MRI segmentation.
    
    The dataset expects the following structure:
    - Images in semantic_MRs_anon/ folder
    - Labels in semantic_labels_anon/ folder
    
    Labels:
    0 - Background
    1 - Body outline
    2 - Bone
    3 - Bladder
    4 - Rectum
    5 - Prostate (we'll remap to 4 in the 5-class version)
    """
    
    def __init__(self, data_root, split='train', target_shape=(64, 128, 128), 
                 transform=None, num_classes=5):
        """
        Args:
            data_root: Root directory containing the data
            split: 'train', 'val', or 'test'
            target_shape: Target volume shape (D, H, W) for resizing
            transform: Optional transforms to apply
            num_classes: Number of segmentation classes
        """
        self.data_root = data_root
        self.split = split
        self.target_shape = target_shape
        self.transform = transform
        self.num_classes = num_classes
        
        # Paths to image and label directories
        self.image_dir = os.path.join(data_root, 'semantic_MRs_anon')
        self.label_dir = os.path.join(data_root, 'semantic_labels_anon')
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.nii.gz')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # Split data into train/val/test
        self._split_data()
        
        print(f"Loaded {len(self.image_files)} volumes for {split} split")
        
    def _split_data(self):
        """Split data into train/val/test sets."""
        np.random.seed(42)
        
        # Filter to only include images that have corresponding labels
        # Labels have _SEMANTIC inserted before _LFOV
        valid_files = []
        for img_path in self.image_files:
            filename = os.path.basename(img_path)
            label_filename = filename.replace('_LFOV.nii.gz', '_SEMANTIC_LFOV.nii.gz')
            label_path = os.path.join(self.label_dir, label_filename)
            if os.path.exists(label_path):
                valid_files.append(img_path)
        
        print(f"Found {len(valid_files)} images with corresponding labels out of {len(self.image_files)} total images")
        
        # Get unique cases from valid files only
        cases = {}
        for img_path in valid_files:
            filename = os.path.basename(img_path)
            case_id = filename.split('_Week')[0]  # Extract case ID
            if case_id not in cases:
                cases[case_id] = []
            cases[case_id].append(img_path)
        
        case_ids = sorted(cases.keys())
        n_cases = len(case_ids)
        
        # Split: 70% train, 15% val, 15% test
        n_train = int(0.7 * n_cases)
        n_val = int(0.15 * n_cases)
        
        train_cases = case_ids[:n_train]
        val_cases = case_ids[n_train:n_train + n_val]
        test_cases = case_ids[n_train + n_val:]
        
        # Collect files for this split
        if self.split == 'train':
            selected_cases = train_cases
        elif self.split == 'val':
            selected_cases = val_cases
        else:  # test
            selected_cases = test_cases
        
        selected_files = []
        for case_id in selected_cases:
            selected_files.extend(cases[case_id])
        
        self.image_files = selected_files
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Load and return a single volume and its label.
        
        Returns:
            image: (1, D, H, W) tensor
            label_onehot: (C, D, H, W) tensor - one-hot encoded
            label_class: (D, H, W) tensor - class indices
        """
        # Load image
        img_path = self.image_files[idx]
        image, _ = load_nifti_volume(img_path, normalize=True)
        
        # Load corresponding label
        # Labels have _SEMANTIC inserted before _LFOV
        filename = os.path.basename(img_path)
        label_filename = filename.replace('_LFOV.nii.gz', '_SEMANTIC_LFOV.nii.gz')
        label_path = os.path.join(self.label_dir, label_filename)
        
        if os.path.exists(label_path):
            label, _ = load_nifti_volume(label_path, normalize=False)
        else:
            # If label doesn't exist, skip this sample
            print(f"Warning: Label not found for {filename}, skipping...")
            return self.__getitem__((idx + 1) % len(self.image_files))
        
        # Resize to target shape
        image = self._resize_volume(image)
        label = self._resize_volume(label, is_label=True)
        
        # Ensure label values are in valid range
        label = np.clip(label, 0, self.num_classes - 1)
        
        # Convert to tensors
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim (1, D, H, W)
        label_class = torch.from_numpy(label).long()  # (D, H, W)
        
        # Create one-hot encoded label
        label_onehot = torch.zeros((self.num_classes,) + label_class.shape, dtype=torch.float32)
        for c in range(self.num_classes):
            label_onehot[c] = (label_class == c).float()
        
        # Apply transforms if any
        if self.transform:
            image, label_onehot, label_class = self.transform(image, label_onehot, label_class)
        
        return image, label_onehot, label_class
    
    def _resize_volume(self, volume, is_label=False):
        """
        Resize volume to target shape.
        
        Args:
            volume: Input volume
            is_label: Whether this is a label (use nearest neighbor interpolation)
        
        Returns:
            Resized volume
        """
        import torch.nn.functional as F
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        
        # Resize
        mode = 'nearest' if is_label else 'trilinear'
        resized = F.interpolate(volume_tensor, size=self.target_shape, mode=mode, align_corners=False if mode == 'trilinear' else None)
        
        # Convert back to numpy
        resized = resized.squeeze().numpy()
        
        return resized


def get_data_loaders(data_root, batch_size=2, target_shape=(64, 128, 128), 
                     num_workers=4, num_classes=5):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_root: Root directory containing the data
        batch_size: Batch size for training
        target_shape: Target volume shape (D, H, W)
        num_workers: Number of worker processes for data loading
        num_classes: Number of segmentation classes
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = ProstateDataset3D(
        data_root=data_root,
        split='train',
        target_shape=target_shape,
        num_classes=num_classes
    )
    
    val_dataset = ProstateDataset3D(
        data_root=data_root,
        split='val',
        target_shape=target_shape,
        num_classes=num_classes
    )
    
    test_dataset = ProstateDataset3D(
        data_root=data_root,
        split='test',
        target_shape=target_shape,
        num_classes=num_classes
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    data_root = r'D:\zxl\Downloads\COMP3710_Final_Report\Data\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1'
    
    dataset = ProstateDataset3D(data_root=data_root, split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Load one sample
    image, label_onehot, label_class = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label (one-hot) shape: {label_onehot.shape}")
    print(f"Label (class) shape: {label_class.shape}")
    print(f"Label unique values: {torch.unique(label_class)}")
