import torch
import numpy as np
from torchvision import transforms
import random


def add_gaussian_noise(image, mean=0.0, std=0.05):
    """Add Gaussian noise to the image"""
    if isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        image = torch.tensor(image)

    noise = torch.randn_like(image) * std + mean
    image = image + noise
    return torch.clamp(image, 0.0, 1.0)


def improved_specaugment(image, freq_mask_param=30, time_mask_param=30, num_masks=2):
    """
    Improved SpecAugment implementation for audio spectrograms
    """
    if isinstance(image, torch.Tensor):
        image = image.clone()
    else:
        image = torch.tensor(image)

    C, H, W = image.shape

    for _ in range(num_masks):
        f = random.randint(1, min(freq_mask_param, H // 10))
        f0 = random.randint(0, H - f)
        image[:, f0:f0 + f, :] = 0

    # Time masking (horizontal bands)
    for _ in range(num_masks):
        t = random.randint(1, min(time_mask_param, W // 10))
        t0 = random.randint(0, W - t)
        image[:, :, t0:t0 + t] = 0

    return image


train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: add_gaussian_noise(x) if np.random.rand() < 0.3 else x),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Lambda(lambda x: improved_specaugment(x) if np.random.rand() < 0.5 else x)
])

test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])