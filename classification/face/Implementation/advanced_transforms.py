import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import random

class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection."""
    def __init__(self, csv_file, transform=None, preprocess_mode='standard'):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.preprocess_mode = preprocess_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = 0 if self.data.iloc[idx]['label'] == 'real' else 1
        image = Image.open(img_path).convert('RGB')

        if self.preprocess_mode == 'frequency':
            image = self.apply_frequency_preprocessing(image)
        elif self.preprocess_mode == 'edge':
            image = self.apply_edge_preprocessing(image)

        if self.transform:
            image = self.transform(image)

        return image, float(label), img_path

    def apply_frequency_preprocessing(self, image):
        """Apply frequency domain preprocessing."""
        img_array = np.array(image)
        for i in range(3):
            channel = img_array[:, :, i]
            f_transform = np.fft.fft2(channel)
            f_shift = np.fft.fftshift(f_transform)
            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            r = 30
            center = (crow, ccol)
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 0
            f_shift = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min()) * 255
            img_array[:, :, i] = 0.7 * channel + 0.3 * img_back
        return Image.fromarray(img_array.astype(np.uint8))

    def apply_edge_preprocessing(self, image):
        """Apply edge detection preprocessing."""
        gray = np.array(image.convert('L'))
        edges = cv2.Canny(gray, 50, 150)
        edge_rgb = np.stack([edges, edges, edges], axis=2)
        original = np.array(image)
        combined = 0.8 * original + 0.2 * edge_rgb
        return Image.fromarray(combined.astype(np.uint8))

class RandomJPEGCompression:
    """Apply random JPEG compression."""
    def __init__(self, quality_range=(70, 90)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class RandomBlur:
    """Apply random blur."""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        radius = random.uniform(0.3, 0.7)
        return img.filter(ImageFilter.GaussianBlur(radius))

class RandomNoise:
    """Add random noise."""
    def __init__(self, p=0.2):
        self.p = p
        self.noise_type = 'all'

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_array = np.array(img, dtype=np.float32) / 255.0
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 0.01, img_array.shape).astype(np.float32)
            img_array = img_array + noise
        elif noise_type == 'salt_pepper':
            prob = 0.01
            rnd = np.random.random(img_array.shape[:2]).astype(np.float32)
            img_array[rnd < prob / 2] = 0
            img_array[rnd > 1 - prob / 2] = 1
        else:
            noise = np.random.randn(*img_array.shape).astype(np.float32) * 0.01
            img_array = img_array + img_array * noise
        img_array = np.clip(img_array, 0, 1)
        return Image.fromarray((img_array * 255).astype(np.uint8))

class RandomCrop:
    """Simple random crop."""
    def __init__(self, size):
        self.random_crop = transforms.RandomCrop(size)

    def __call__(self, img):
        return self.random_crop(img)

class ColorConstancy:
    """Apply color constancy algorithms."""
    def __init__(self, method='gray_world', p=0.2):
        self.method = method
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_array = np.array(img).astype(np.float32)
        if self.method == 'gray_world':
            for i in range(3):
                channel_mean = img_array[:, :, i].mean()
                img_array[:, :, i] = img_array[:, :, i] * (128.0 / channel_mean)
        elif self.method == 'white_patch':
            for i in range(3):
                channel_max = img_array[:, :, i].max()
                img_array[:, :, i] = img_array[:, :, i] * (255.0 / channel_max)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

class AdvancedCutout:
    """Enhanced cutout with multiple strategies."""
    def __init__(self, n_holes=2, length=15, strategy='random'):
        self.n_holes = n_holes
        self.length = length
        self.strategy = strategy

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            if self.strategy == 'random':
                y = np.random.randint(h)
                x = np.random.randint(w)
            elif self.strategy == 'grid':
                grid_size = int(np.sqrt(self.n_holes))
                idx = _
                y = (idx // grid_size) * (h // grid_size) + h // (2 * grid_size)
                x = (idx % grid_size) * (w // grid_size) + w // (2 * grid_size)
            else:
                y = int(np.random.normal(h / 2, h / 12))
                x = int(np.random.normal(w / 2, w / 12))
                y = np.clip(y, 0, h - 1)
                x = np.clip(x, 0, w - 1)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            if self.strategy == 'smooth':
                for i in range(y1, y2):
                    for j in range(x1, x2):
                        dist = np.sqrt((i - y) ** 2 + (j - x) ** 2)
                        mask[i, j] = min(1.0, dist / (self.length / 2))
            else:
                mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img).to(img.device)
        return img * mask

IMG_SIZE = 299
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ColorConstancy(method='gray_world', p=0.2),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
    transforms.RandomGrayscale(p=0.1),
    RandomJPEGCompression(quality_range=(70, 90)),
    RandomBlur(p=0.2),
    RandomNoise(p=0.2),
    transforms.ToTensor(),
    AdvancedCutout(n_holes=2, length=15, strategy='random'),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

])

val_test_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])