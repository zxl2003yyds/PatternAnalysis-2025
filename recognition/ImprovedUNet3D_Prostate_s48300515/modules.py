"""
3D Improved UNet modules for prostate segmentation.
Implementation based on the 3D UNet architecture with improvements including:
- Instance normalization
- Residual connections
- Deep supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    3D Convolutional block with two conv layers, instance normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


class ResidualConvBlock3D(nn.Module):
    """
    3D Residual convolutional block with skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualConvBlock3D, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels, kernel_size, padding)
        
        # Skip connection
        self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
            
        out = out + residual
        return out


class EncoderBlock3D(nn.Module):
    """
    Encoder block for 3D UNet with downsampling.
    """
    def __init__(self, in_channels, out_channels, use_residual=True):
        super(EncoderBlock3D, self).__init__()
        if use_residual:
            self.conv_block = ResidualConvBlock3D(in_channels, out_channels)
        else:
            self.conv_block = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv_block(x)
        pooled = self.pool(x)
        return x, pooled


class DecoderBlock3D(nn.Module):
    """
    Decoder block for 3D UNet with upsampling and skip connections.
    """
    def __init__(self, in_channels, out_channels, use_residual=True):
        super(DecoderBlock3D, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        if use_residual:
            self.conv_block = ResidualConvBlock3D(in_channels, out_channels)
        else:
            self.conv_block = ConvBlock3D(in_channels, out_channels)
        
    def forward(self, x, skip_connection):
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape != skip_connection.shape:
            x = self._pad_or_crop(x, skip_connection)
            
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_block(x)
        return x
    
    def _pad_or_crop(self, x, target):
        """Pad or crop x to match target size."""
        diff_d = target.size(2) - x.size(2)
        diff_h = target.size(3) - x.size(3)
        diff_w = target.size(4) - x.size(4)
        
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                     diff_h // 2, diff_h - diff_h // 2,
                     diff_d // 2, diff_d - diff_d // 2])
        return x


class ImprovedUNet3D(nn.Module):
    """
    3D Improved UNet for medical image segmentation.
    
    Features:
    - Instance normalization for better training stability
    - Residual connections for improved gradient flow
    - Deep supervision for better feature learning
    - Multi-scale feature extraction
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale MRI)
        out_channels: Number of output segmentation classes
        base_channels: Base number of channels (default: 32)
        use_residual: Use residual connections (default: True)
        deep_supervision: Use deep supervision (default: False)
    """
    def __init__(self, in_channels=1, out_channels=5, base_channels=32, 
                 use_residual=True, deep_supervision=False):
        super(ImprovedUNet3D, self).__init__()
        
        self.deep_supervision = deep_supervision
        
        # Encoder path
        self.encoder1 = EncoderBlock3D(in_channels, base_channels, use_residual)
        self.encoder2 = EncoderBlock3D(base_channels, base_channels * 2, use_residual)
        self.encoder3 = EncoderBlock3D(base_channels * 2, base_channels * 4, use_residual)
        self.encoder4 = EncoderBlock3D(base_channels * 4, base_channels * 8, use_residual)
        
        # Bottleneck
        if use_residual:
            self.bottleneck = ResidualConvBlock3D(base_channels * 8, base_channels * 16)
        else:
            self.bottleneck = ConvBlock3D(base_channels * 8, base_channels * 16)
        
        # Decoder path
        self.decoder4 = DecoderBlock3D(base_channels * 16, base_channels * 8, use_residual)
        self.decoder3 = DecoderBlock3D(base_channels * 8, base_channels * 4, use_residual)
        self.decoder2 = DecoderBlock3D(base_channels * 4, base_channels * 2, use_residual)
        self.decoder1 = DecoderBlock3D(base_channels * 2, base_channels, use_residual)
        
        # Final output layer
        self.output = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
        # Deep supervision outputs
        if deep_supervision:
            self.ds_output4 = nn.Conv3d(base_channels * 8, out_channels, kernel_size=1)
            self.ds_output3 = nn.Conv3d(base_channels * 4, out_channels, kernel_size=1)
            self.ds_output2 = nn.Conv3d(base_channels * 2, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.decoder4(x, skip4)
        if self.deep_supervision:
            ds4 = self.ds_output4(x)
            
        x = self.decoder3(x, skip3)
        if self.deep_supervision:
            ds3 = self.ds_output3(x)
            
        x = self.decoder2(x, skip2)
        if self.deep_supervision:
            ds2 = self.ds_output2(x)
            
        x = self.decoder1(x, skip1)
        
        # Final output
        output = self.output(x)
        
        if self.deep_supervision and self.training:
            return output, [ds4, ds3, ds2]
        else:
            return output


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    def __init__(self, smooth=1.0, weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = weights
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, D, H, W) - predicted logits
            targets: (B, C, D, H, W) - one-hot encoded targets
        """
        predictions = torch.softmax(predictions, dim=1)
        
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        # Calculate Dice coefficient for each class
        intersection = (predictions * targets).sum(dim=2)
        union = predictions.sum(dim=2) + targets.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights if provided
        if self.weights is not None:
            dice = dice * self.weights.to(dice.device)
        
        # Return 1 - mean Dice (loss)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice and Cross-Entropy loss.
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(weights=class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, predictions, targets_onehot, targets_class):
        """
        Args:
            predictions: (B, C, D, H, W) - predicted logits
            targets_onehot: (B, C, D, H, W) - one-hot encoded targets for Dice
            targets_class: (B, D, H, W) - class indices for CE
        """
        dice = self.dice_loss(predictions, targets_onehot)
        ce = self.ce_loss(predictions, targets_class)
        
        return self.dice_weight * dice + self.ce_weight * ce


def dice_coefficient(predictions, targets, smooth=1.0):
    """
    Calculate Dice coefficient for evaluation.
    
    Args:
        predictions: (B, C, D, H, W) - predicted probabilities or logits
        targets: (B, C, D, H, W) - one-hot encoded targets
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient per class
    """
    if predictions.dim() == 5 and predictions.size(1) > 1:
        predictions = torch.softmax(predictions, dim=1)
    
    predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
    targets = targets.view(targets.size(0), targets.size(1), -1)
    
    intersection = (predictions * targets).sum(dim=2)
    union = predictions.sum(dim=2) + targets.sum(dim=2)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.mean(dim=0)  # Average over batch, return per class
