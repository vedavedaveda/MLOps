from pathlib import Path
from typing import Tuple

import typer
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


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
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.image_paths.append(img_file)
                    self.labels.append(0)
        
        # Load Real art (label 1)
        real_art_path = self.data_path / "Art" / "RealArt"
        if real_art_path.exists():
            for img_file in real_art_path.rglob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
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
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print(f"Found {len(self)} images in the dataset")
        print(f"AI-generated art: {sum(1 for l in self.labels if l == 0)} images")
        print(f"Real art: {sum(1 for l in self.labels if l == 1)} images")


def get_datasets(data_path: Path = None) -> Tuple[Dataset, Dataset]:
    """Get training and test datasets with appropriate transforms."""
    
    # Default to data directory in project root
    if data_path is None:
        # Get the project root (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data"
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = MyDataset(data_path, transform=None)
    
    # Split into train/test (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(full_dataset)))
    
    # Create separate datasets with different transforms
    train_dataset = MyDataset(data_path, transform=train_transform)
    test_dataset = MyDataset(data_path, transform=test_transform)
    
    # Use Subset to split the data
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Test set: {len(test_dataset)} images")
    
    return train_dataset, test_dataset


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
