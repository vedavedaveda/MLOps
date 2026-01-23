from ml_ops_project.model import CNN
from ml_ops_project.data import  get_dataloaders

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import OmegaConf
from loguru import logger
import os
import wandb
from pathlib import Path
from hydra.utils import get_original_cwd

log = logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../../configs", config_name="default_config")
def train(cfg) -> None:
    # Configure loguru to save logs to hydra output directory
    try:
        hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except ValueError:
        # Hydra not initialized (e.g., in tests)
        hydra_path = "."

    logger.add(os.path.join(hydra_path, "training.log"), level="DEBUG")

    logger.info("Starting training session")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    # Initialize Weights & Biases
    hparams = cfg.experiment
    wandb.init(
        project="art-classifier",  # Change this to your project name
        config={
            "learning_rate": hparams.training.learning_rate,
            "epochs": hparams.training.epochs,
            "batch_size": hparams.training.batch_size,
            "num_classes": hparams.model.number_of_classes,
            "optimizer": "Adam",
            "device": str(DEVICE),
        }
    )
    logger.info("Weights & Biases initialized")

    logger.info(f"Initializing CNN model with {hparams.model.number_of_classes} classes")
    model = CNN(number_of_classes=hparams.model.number_of_classes).to(DEVICE)
    logger.info(f"Model moved to device: {DEVICE}")

    logger.info("Loading datasets...")
    train_loader, test_loader, _ = get_dataloaders(
        batch_size=hparams.training.batch_size,
        num_workers=hparams.training.num_workers,  # distributed/parallel loading
        pin_memory=(DEVICE.type == "cuda"),
    )
    logger.debug(f"DataLoaders created with batch_size={hparams.training.batch_size}")
    logger.info(f"Train DataLoader num_workers = {train_loader.num_workers}")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.training.learning_rate)
    logger.info(f"Optimizer: Adam with learning_rate={hparams.training.learning_rate}")

    statistics = {"train_loss": [], "train_accuracy": []}

    logger.info(f"Starting training for {hparams.training.epochs} epochs...")
    for epoch in range(hparams.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())
            accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            epoch_loss += loss.item()
            epoch_acc += accuracy
            num_batches += 1

            # Log to Weights & Biases every step
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "epoch": epoch
            })

            if i % hparams.training.log_every_n_steps == 0:
                log.info(f"Epoch {epoch+1}/{hparams.training.epochs}, Batch {i}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

                # Log sample images to wandb (first batch of each logging step)
                if i == 0:
                    wandb.log({
                        "examples": [wandb.Image(img) for img in images[:5].cpu()]
                    })

        # Log epoch summary
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")

        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch_avg_accuracy": avg_acc
        })

    logger.info("Training completed, saving model...")

    try:
        project_root = Path(get_original_cwd())
    except ValueError:
        # Hydra is not initialized (e.g., in unit tests calling __wrapped__)
        project_root = Path.cwd()
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.success(f"Model saved to {model_path}")
    print(f"Saved model to {model_path}")

    logger.info("Generating training statistics plots...")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Accuracy")
    plot_path = "training_statistics.png"
    fig.savefig(plot_path)
    logger.success(f"Plot saved to {plot_path}")
    print(f"Saved plot to {plot_path}")

    # Log the training plot to wandb
    wandb.log({"training_curves": wandb.Image(plot_path)})
    plt.close(fig)

    logger.info("Starting model evaluation on test set...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE, dtype=torch.long)
            logits = model(images)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

            if batch_idx % 10 == 0:
                logger.debug(f"Evaluated {batch_idx}/{len(test_loader)} batches")

    test_accuracy = correct / total
    logger.success(f"Test accuracy: {test_accuracy:.4f}")

    # Log test accuracy to wandb
    wandb.log({"test_accuracy": test_accuracy})

    # Save model as wandb artifact
    logger.info("Saving model as Weights & Biases artifact...")
    artifact = wandb.Artifact(
        name="art-classifier-model",
        type="model",
        description="CNN model for AI vs Real art classification",
        metadata={
            "test_accuracy": test_accuracy,
            "epochs": hparams.training.epochs,
            "learning_rate": hparams.training.learning_rate,
        }
    )
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)
    logger.success("Model artifact saved to Weights & Biases")

    # Finish the wandb run
    wandb.finish()
    logger.info("Training and evaluation completed successfully!")

if __name__ == "__main__":
    train()
