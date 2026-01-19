from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from torchvision import transforms
from torchvision.utils import make_grid

from data import MyDataset


def dataset_statistics(datadir: str = "data", samples_per_class: int = 25) -> None:
    """Compute dataset statistics and save separate sample grids for AI (0) and Real (1)."""
    dataset = MyDataset(
        data_path=Path(datadir),
        transform=transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        ),
    )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check your data folder structure.")

    print("Dataset: AI vs Real Art")
    print(f"Number of images: {len(dataset)}")
    print(f"Image shape: {dataset[0][0].shape}")
    print("")

    labels_tensor = torch.tensor(dataset.labels, dtype=torch.long)
    label_counts = torch.bincount(labels_tensor, minlength=2)

    print("Label distribution:")
    print(f"AI (label 0): {int(label_counts[0].item())}")
    print(f"Real (label 1): {int(label_counts[1].item())}")

    # Helper: collect first N images for a given label
    def collect_samples_for_label(target_label: int, max_samples: int) -> list[torch.Tensor]:
        samples = []
        for idx in range(len(dataset)):
            image_tensor, label = dataset[idx]
            if int(label) == target_label:
                samples.append(image_tensor)
                if len(samples) >= max_samples:
                    break
        return samples

    ai_samples = collect_samples_for_label(target_label=0, max_samples=samples_per_class)
    real_samples = collect_samples_for_label(target_label=1, max_samples=samples_per_class)

    # Save AI grid
    if len(ai_samples) > 0:
        ai_grid = make_grid(ai_samples, nrow=5, padding=2)
        plt.figure(figsize=(8, 8))
        plt.imshow(ai_grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title("Sample images: AI-generated (label 0)")
        plt.tight_layout()
        plt.savefig("art_samples_ai.png", dpi=200)
        plt.close()
    else:
        print("Warning: no AI (label 0) images found to plot.")

    # Save Real grid
    if len(real_samples) > 0:
        real_grid = make_grid(real_samples, nrow=5, padding=2)
        plt.figure(figsize=(8, 8))
        plt.imshow(real_grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title("Sample images: Real art (label 1)")
        plt.tight_layout()
        plt.savefig("art_samples_real.png", dpi=200)
        plt.close()
    else:
        print("Warning: no Real (label 1) images found to plot.")

    # Label distribution plot
    plt.figure()
    plt.bar([0, 1], label_counts.tolist())
    plt.xticks([0, 1], ["AI (0)", "Real (1)"])
    plt.title("Label distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("art_label_distribution.png", dpi=200)
    plt.close()

    print("Saved:")
    print(" - art_samples_ai.png")
    print(" - art_samples_real.png")
    print(" - art_label_distribution.png")


if __name__ == "__main__":
    typer.run(dataset_statistics)
