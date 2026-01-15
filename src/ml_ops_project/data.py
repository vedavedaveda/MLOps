from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import typer


class MyDataset(Dataset):
    def __init__(self, data_path: Path, transform=None) -> None:
        self.data_path = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load AI art images (label 0)
        ai_path = data_path / "AiArtData"
        if ai_path.exists():
            for img_file in ai_path.glob("*.jpg"):
                self.images.append(img_file)
                self.labels.append(0)
            for img_file in ai_path.glob("*.png"):
                self.images.append(img_file)
                self.labels.append(0)
            for img_file in ai_path.glob("*.jpeg"):
                self.images.append(img_file)
                self.labels.append(0)
        
        # Load human art images (label 1)
        human_path = data_path / "RealArt"
        if human_path.exists():
            for img_file in human_path.glob("*.jpg"):
                self.images.append(img_file)
                self.labels.append(1)
            for img_file in human_path.glob("*.png"):
                self.images.append(img_file)
                self.labels.append(1)
            for img_file in human_path.glob("*.jpeg"):
                self.images.append(img_file)
                self.labels.append(1)
        
        print(f"Loaded {len(self.images)} images: {self.labels.count(0)} AI art, {self.labels.count(1)} human art")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        img_path = self.images[index]
        label = self.labels[index]
        
        # Load and convert image to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        # This method can be used for more advanced preprocessing if needed
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Preprocessing complete. Output folder: {output_folder}")


def get_datasets(data_path: Path = None):
    """
    Load and split the dataset into training and test sets.
    
    Args:
        data_path: Path to the folder containing AiArtData and RealArt subfolders.
                   If None, uses the project root's data/Art directory.
    
    Returns:
        tuple: (train_set, test_set)
    """
    # If no path provided, construct absolute path to project root's data/raw
    if data_path is None:
        # Get the directory where this file is located
        current_file = Path(__file__).resolve()
        # Navigate to project root (assuming this file is in src/ml_ops_project/)
        project_root = current_file.parent.parent.parent
        data_path = project_root / "data" / "Art"
    
    print(f"Looking for data in: {data_path}")
    
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32 for the CNN
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create the full dataset
    dataset = MyDataset(data_path, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_path}. Please check the path and folder structure.")
    
    # Split into train and test sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, test_set


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)