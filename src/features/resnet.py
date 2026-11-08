"""Pretrained ResNet-18 penultimate-layer features.

ResNet-18 (He et al., 2016) is a convolutional network trained on
ImageNet (1.2M natural images, 1000 classes). We use it purely as a
feature extractor: run an image through every layer except the final
1000-way classifier, and read off the 512-dimensional vector that layer
would have used to make its prediction (the "penultimate" layer, after
global average pooling). That vector tends to capture high-level visual
similarity (shape, object parts) learned from a much larger and more
varied dataset than CIFAR-10 alone, which is why it is a common choice
for "generic" image features such as the ones used here.
"""

import numpy as np
import torch
import torchvision

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_resnet18_feature_extractor() -> torch.nn.Module:
    """Pretrained ResNet-18 with its final classification layer removed,
    so forward() returns the 512-d penultimate feature instead of 1000
    class scores."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


def _preprocess_batch(images_u8: np.ndarray, resolution: int) -> torch.Tensor:
    """images_u8: (B, 32, 32, 3) uint8. Returns a normalized (B, 3, res, res)
    float32 tensor, resized with bilinear interpolation and normalized
    with the ImageNet mean/std the pretrained weights were trained with."""
    batch = torch.from_numpy(images_u8).permute(0, 3, 1, 2).float() / 255.0  # (B, 3, 32, 32)
    if resolution != batch.shape[-1]:
        batch = torch.nn.functional.interpolate(
            batch, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    return batch


def extract_resnet_features(
    images: np.ndarray,
    model: torch.nn.Module,
    resolution: int = 32,
    batch_size: int = 256,
) -> np.ndarray:
    """images: uint8 array (N, 32, 32, 3). Returns float32 (N, 512)."""
    n = images.shape[0]
    all_features = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_u8 = images[start:start + batch_size]
            batch = _preprocess_batch(batch_u8, resolution=resolution)
            features = model(batch)
            all_features.append(features.numpy())
    return np.concatenate(all_features, axis=0).astype(np.float32)
