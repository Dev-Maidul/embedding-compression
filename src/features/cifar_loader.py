"""Load the CIFAR-10 test set as plain NumPy arrays.

We standardize on the 10,000-image test split as the fixed retrieval set
for Part B/C (the project brief's choice), so this one function is the single
place that decides how those images get into memory, keeping every
downstream script consistent.
"""

import numpy as np
import torchvision


def load_cifar10_test(root: str = "data/cifar10") -> tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels).

    images: uint8 array of shape (10000, 32, 32, 3), RGB.
    labels: int array of shape (10000,), CIFAR-10 class indices 0-9.
    """
    dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    images = np.stack([np.array(img) for img, _ in dataset]).astype(np.uint8)
    labels = np.array(dataset.targets, dtype=np.int64)
    return images, labels
