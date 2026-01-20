from pathlib import Path
from typing import Tuple

import torch
import typer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class MyDataset(Dataset):
    """Dataset for AI-generated vs Real art classification."""

    def __init__(self, data_path: Path, transform=None) -> None:
        self.data_path = Path(data_path)
        self.transform = transform

        # Collect all image files
        self.image_paths = []
        self.labels = []

        # Load AI art (label 0)
        ai_art_path = self.data_path / "Art" / "AiArtData"
        if ai_art_path.exists():
            for img_file in ai_art_path.rglob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.image_paths.append(img_file)
                    self.labels.append(0)

        # Load Real art (label 1)
        real_art_path = self.data_path / "Art" / "RealArt"
        if real_art_path.exists():
            for img_file in real_art_path.rglob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.image_paths.append(img_file)
                    self.labels.append(1)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        img_path = self.image_paths[index]
        label = self.labels[index]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print(f"Found {len(self)} images in the dataset")
        print(f"AI-generated art: {sum(1 for l in self.labels if l == 0)} images")
        print(f"Real art: {sum(1 for l in self.labels if l == 1)} images")


# Pre-loading optimization (DTU MLOps technique)
def preload_images_as_tensors(dataset: MyDataset, target_size=(32, 32)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-load all images from disk as tensors (THE KEY OPTIMIZATION!).

    This eliminates the PIL bottleneck by loading images once instead of
    every epoch.
    """
    print(f"Pre-loading {len(dataset)} images as tensors...")

    images_list = []
    labels_list = []

    total = len(dataset)
    for idx in range(total):
        # Progress indicator (every 10%)
        if idx % max(1, total // 10) == 0:
            print(f"  Progress: {idx}/{total} ({100 * idx / total:.0f}%)")

        img_path = dataset.image_paths[idx]
        label = dataset.labels[idx]

        # Load and resize image
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size, Image.BILINEAR)

        # Convert to tensor [0, 1]
        img_tensor = TF.to_tensor(img)

        images_list.append(img_tensor)
        labels_list.append(label)

    print(f"  Progress: {total}/{total} (100%)")

    # Stack into tensors
    images_tensor = torch.stack(images_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    print(f"✓ Loaded {images_tensor.shape}")
    return images_tensor, labels_tensor


class TensorDatasetWithTransform(Dataset):
    """Dataset that applies transforms to pre-loaded tensors."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# get_datasets now uses tensor pre-loading
def get_datasets(data_path: Path = None, use_preloading: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Get training and test datasets.

    Args:
        data_path: Path to data directory
        use_preloading: If True (default), pre-load images as tensors for 5x speedup.
                       Set to False to use original PIL-based method.
    """

    # Default to data directory in project root
    if data_path is None:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data"

    if not use_preloading:
        # Original method (slow but compatible)
        return _get_datasets_original(data_path)

    # OPTIMIZED METHOD
    print("Using optimized tensor-based data loading")

    # Load original dataset to get paths
    full_dataset = MyDataset(data_path, transform=None)
    print(f"Total images found: {len(full_dataset)}")

    # Pre-load all images as tensors (THE OPTIMIZATION!)
    all_images, all_labels = preload_images_as_tensors(full_dataset, target_size=(32, 32))

    # Split into train/test (80/20)
    total_size = len(all_images)
    train_size = int(0.8 * total_size)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_size, generator=generator)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_images = all_images[train_indices]
    train_labels = all_labels[train_indices]
    test_images = all_images[test_indices]
    test_labels = all_labels[test_indices]

    print(f"Training set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")

    # Transforms (work on tensors, not PIL)
    # Note: No Resize or ToTensor needed - already done!
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Create datasets
    train_dataset = TensorDatasetWithTransform(train_images, train_labels, train_transform)
    test_dataset = TensorDatasetWithTransform(test_images, test_labels, test_transform)

    return train_dataset, test_dataset


def _get_datasets_original(data_path: Path) -> Tuple[Dataset, Dataset]:
    """Original method (kept for backwards compatibility)."""

    # Training transforms with augmentation
    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Test transforms without augmentation
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load full dataset
    full_dataset = MyDataset(data_path, transform=None)
    print(f"Total images found: {len(full_dataset)}")

    # Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_subset, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    # Wrapper to apply transforms
    class TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

        def __len__(self):
            return len(self.subset)

    train_dataset = TransformSubset(train_subset, train_transform)
    test_dataset = TransformSubset(test_subset, test_transform)

    print(f"Training set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")

    return train_dataset, test_dataset


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
