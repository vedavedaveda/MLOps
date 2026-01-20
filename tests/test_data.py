from pathlib import Path

import pytest
from ml_ops_project.data import MyDataset, get_datasets
from PIL import Image
from torchvision import transforms


@pytest.fixture
def fake_data_dir(tmp_path):
    """Create a fake data directory with test images."""
    ai_art_dir = tmp_path / "Art" / "AiArtData"
    real_art_dir = tmp_path / "Art" / "RealArt"
    ai_art_dir.mkdir(parents=True)
    real_art_dir.mkdir(parents=True)

    # Create 5 fake AI art images
    for i in range(5):
        img = Image.new("RGB", (64, 64), color=(i * 50, 0, 0))
        img.save(ai_art_dir / f"ai_art_{i}.jpg")

    # Create 5 fake real art images
    for i in range(5):
        img = Image.new("RGB", (64, 64), color=(0, i * 50, 0))
        img.save(real_art_dir / f"real_art_{i}.jpg")

    return tmp_path


def test_my_dataset_is_dataset(fake_data_dir):
    """Test that MyDataset is a proper PyTorch Dataset."""
    from torch.utils.data import Dataset

    dataset = MyDataset(fake_data_dir)
    assert isinstance(dataset, Dataset)


def test_my_dataset_length(fake_data_dir):
    """Test that dataset returns correct length."""
    dataset = MyDataset(fake_data_dir)
    assert len(dataset) == 10  # 5 AI + 5 Real


def test_my_dataset_labels(fake_data_dir):
    """Test that labels are correctly assigned."""
    dataset = MyDataset(fake_data_dir)
    labels_set = set(dataset.labels)
    assert 0 in labels_set, "Should have AI art images (label 0)"
    assert 1 in labels_set, "Should have Real art images (label 1)"
    assert dataset.labels.count(0) == 5
    assert dataset.labels.count(1) == 5


def test_my_dataset_getitem_without_transform(fake_data_dir):
    """Test getting item without transforms."""
    dataset = MyDataset(fake_data_dir, transform=None)
    img, label = dataset[0]
    assert img.size == (64, 64)  # PIL Image size
    assert label in [0, 1]


def test_my_dataset_getitem_with_transform(fake_data_dir):
    """Test getting item with transforms."""
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset = MyDataset(fake_data_dir, transform=transform)
    img, label = dataset[0]
    assert img.shape == (3, 32, 32)  # Tensor shape [C, H, W]
    assert label in [0, 1]


def test_get_datasets_returns_tuple(fake_data_dir, monkeypatch):
    """Test that get_datasets returns train and test sets."""
    train_set, test_set = get_datasets(data_path=fake_data_dir)
    assert train_set is not None
    assert test_set is not None


def test_get_datasets_split(fake_data_dir):
    """Test that train/test split is approximately 80/20."""
    train_set, test_set = get_datasets(data_path=fake_data_dir)
    total = len(train_set) + len(test_set)
    assert total == 10
    assert len(train_set) == 8  # 80% of 10
    assert len(test_set) == 2  # 20% of 10


def test_get_datasets_image_shapes(fake_data_dir):
    """Test that images from datasets have correct shape."""
    train_set, test_set = get_datasets(data_path=fake_data_dir)

    # Get first image from train set
    img, label = train_set[0]
    assert img.shape == (3, 32, 32)  # After transforms: [C, H, W]


def test_get_datasets_normalization(fake_data_dir):
    """Test that images are normalized (can have negative values)."""
    train_set, _ = get_datasets(data_path=fake_data_dir)
    img, _ = train_set[0]

    # Normalized images can have negative values
    # (due to mean subtraction in normalization)
    assert img.min() < 0 or img.max() > 1


def test_dataset_reproducibility(fake_data_dir):
    """Test that dataset returns same item for same index."""
    dataset = MyDataset(fake_data_dir, transform=None)
    img1, label1 = dataset[0]
    img2, label2 = dataset[0]

    assert label1 == label2
    # Images should be the same
    import numpy as np

    assert np.array_equal(np.array(img1), np.array(img2))
