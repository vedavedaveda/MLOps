from torch import Tensor
import torch.nn as nn

class CNN(nn.Module):
    """
    CNN image classifier for fixed-size RGB inputs (N, 3, 32, 32).

    Structure:
      - Feature extractor: 3 blocks of Conv2d(3x3, padding=1) + ReLU + MaxPool2d(2)
        (pooling halves H/W each block: 32->16->8->4, channels: 3->32->64->64)
      - Classifier: Flatten (64*4*4=1024) -> Linear(1024->512) + ReLU -> Linear(512->num_classes)

    Output:
      - Returns raw logits of shape (N, num_classes).
    """

    def __init__(self, number_of_classes: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),  # assumes input images are 32x32
            nn.ReLU(),
            nn.Linear(512, number_of_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

if __name__ == "__main__":
    model = CNN()
    x = torch.rand(1, 3, 32, 32)  # Example input tensor
    print(f"Output shape of model: {model(x).shape}")
