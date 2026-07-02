"""
dataset.py

Loads dataset using torchvision ImageFolder
and creates train/validation DataLoaders.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ===========================
# Configuration
# ===========================

DATA_DIR = Path("data/raw")

IMAGE_SIZE = 300

BATCH_SIZE = 16

TRAIN_SPLIT = 0.85

SEED = 42


# ===========================
# Data Augmentation
# ===========================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),
    transforms.RandomAutocontrast(p=0.3),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.RandomAdjustSharpness(
    sharpness_factor=2,
    p=0.3
    ),

    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ===========================
# Validation Transform
# ===========================

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ===========================
# Dataset
# ===========================

full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform
)


# Split sizes

train_size = int(TRAIN_SPLIT * len(full_dataset))

val_size = len(full_dataset) - train_size


generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(

    full_dataset,

    [train_size, val_size],

    generator=generator

)


# Validation shouldn't use augmentation

val_dataset.dataset.transform = val_transform


# ===========================
# DataLoaders
# ===========================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ===========================
# Class Names
# ===========================

class_names = full_dataset.classes


# ===========================
# Test
# ===========================

if __name__ == "__main__":

    print("=" * 60)

    print("Dataset Loaded Successfully")

    print("=" * 60)

    print(f"Classes      : {class_names}")

    print(f"Train Images : {len(train_dataset)}")

    print(f"Val Images   : {len(val_dataset)}")

    print()

    images, labels = next(iter(train_loader))

    print("Batch Shape :", images.shape)

    print("Labels      :", labels)