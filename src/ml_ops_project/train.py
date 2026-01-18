from ml_ops_project.model import CNN
from ml_ops_project.data import get_datasets      # adjust: should return (train_set, test_set)

import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import OmegaConf
from loguru import logger
import os

log = logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../../configs", config_name="default_config")
def train(cfg) -> None:
    # Configure loguru to save logs to hydra output directory
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "training.log"), level="DEBUG")
    
    logger.info("Starting training session")
    logger.debug(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    hparams = cfg.experiment
    logger.info(f"Initializing CNN model with {hparams.model.number_of_classes} classes")
    model = CNN(number_of_classes=hparams.model.number_of_classes).to(DEVICE)
    logger.info(f"Model moved to device: {DEVICE}")

    logger.info("Loading datasets...")
    train_set, test_set = get_datasets()
    logger.info(f"Dataset loaded - Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.training.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hparams.training.batch_size, shuffle=False)
    logger.debug(f"DataLoaders created with batch_size={hparams.training.batch_size}")
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

            if i % hparams.training.log_every_n_steps == 0:
                log.info(f"Epoch {epoch+1}/{hparams.training.epochs}, Batch {i}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        
        # Log epoch summary
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}")

    logger.info("Training completed, saving model...")
    torch.save(model.state_dict(), "cnn_model.pth")
    logger.success("Model saved to cnn_model.pth")
    print("Saved model to cnn_model.pth")

    logger.info("Generating training statistics plots...")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")
    logger.success("Plot saved to training_statistics.png")
    print("Saved plot to training_statistics.png")

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
    logger.info("Training and evaluation completed successfully!")

if __name__ == "__main__":
    train()