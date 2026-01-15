from pathlib import Path
import pytest
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ml_ops_project.data import MyDataset, get_datasets


def test_my_dataset_is_dataset():
    """Test that MyDataset is a valid Dataset."""
    dataset = MyDataset(Path("data"))
    assert isinstance(dataset, Dataset)


def test_my_dataset_length():
    """Test that dataset returns correct length."""
    dataset = MyDataset(Path("data"))
    assert len(dataset) > 0, "Dataset should contain images"
    assert len(dataset) == len(dataset.image_paths)


def test_my_dataset_labels():
    """Test that labels are correctly assigned."""
    dataset = MyDataset(Path("data"))
    # Should have both AI art (0) and Real art (1)
    labels_set = set(dataset.labels)
    assert 0 in labels_set, "Should have AI art images (label 0)"
    assert 1 in labels_set, "Should have Real art images (label 1)"


def test_my_dataset_getitem_without_transform():
    """Test getting item without transforms."""
    dataset = MyDataset(Path("data"), transform=None)
    img, label = dataset[0]
    
    # Check image is PIL Image
    from PIL import Image
    assert isinstance(img, Image.Image)
    
    # Check label is valid
    assert label in [0, 1]


def test_my_dataset_getitem_with_transform():
    """Test getting item with transforms."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    dataset = MyDataset(Path("data"), transform=transform)
    img, label = dataset[0]
    
    # Check image is tensor with correct shape
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32), f"Expected (3, 32, 32), got {img.shape}"
    
    # Check values are in [0, 1] range (from ToTensor)
    assert img.min() >= 0 and img.max() <= 1


def test_get_datasets_returns_tuple():
    """Test that get_datasets returns train and test sets."""
    train_set, test_set = get_datasets()
    assert train_set is not None
    assert test_set is not None


def test_get_datasets_split():
    """Test that train/test split is approximately 80/20."""
    train_set, test_set = get_datasets()
    total = len(train_set) + len(test_set)
    train_ratio = len(train_set) / total
    
    assert 0.75 < train_ratio < 0.85, f"Train ratio should be ~0.8, got {train_ratio}"


def test_get_datasets_image_shapes():
    """Test that images from datasets have correct shape."""
    train_set, test_set = get_datasets()
    
    # Test train set
    train_img, train_label = train_set[0]
    assert train_img.shape == (3, 32, 32)
    assert train_label in [0, 1]
    
    # Test test set
    test_img, test_label = test_set[0]
    assert test_img.shape == (3, 32, 32)
    assert test_label in [0, 1]


def test_get_datasets_normalization():
    """Test that images are normalized (can have negative values)."""
    train_set, _ = get_datasets()
    img, _ = train_set[0]
    
    # After normalization with ImageNet stats, images can have negative values
    # This distinguishes them from just ToTensor (which gives [0, 1])
    assert img.min() < 0 or img.max() > 1, "Images should be normalized"


def test_dataset_reproducibility():
    """Test that dataset returns same item for same index."""
    dataset = MyDataset(Path("data"), transform=None)
    img1, label1 = dataset[0]
    img2, label2 = dataset[0]
    
    assert label1 == label2, "Same index should return same label"
