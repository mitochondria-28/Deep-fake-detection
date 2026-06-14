# dataloader/transforms.py
"""
Frame-level transforms applied to each (112, 112, 3) face frame.
Training: normalise + mild augmentation
Inference: normalise only
"""

import numpy as np
from torchvision import transforms

# ImageNet stats — required because ResNeXt50 was pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    """
    Augmentations applied per-frame during training only.
    Kept mild — aggressive augmentation on face crops hurts detection.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),                        # (H,W,C) uint8 → (C,H,W) float [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    No augmentation at validation/inference time — deterministic output.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])