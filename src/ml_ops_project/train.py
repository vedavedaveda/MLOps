from ml_ops_project.model import CNN
from ml_ops_project.data import get_datasets      # adjust: should return (train_set, test_set)

import matplotlib.pyplot as plt
import torch
import hydra
from omegaconf import OmegaConf
import logging

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../../configs", config_name="default_config")
def train(cfg) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    hparams = cfg.experiment
    model = CNN(number_of_classes=hparams.model.number_of_classes).to(DEVICE)

    train_set, test_set = get_datasets()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.training.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=hparams.training.batch_size, shuffle=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.training.learning_rate)

    statistics = {"train_loss": [], "train_accuracy": []}

    log.info("Starting training CNN model...")
    for epoch in range(hparams.training.epochs):
        model.train()
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

            if i % hparams.training.log_every_n_steps == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, acc: {accuracy:.4f}")

    torch.save(model.state_dict(), "cnn_model.pth")
    print("Saved model to cnn_model.pth")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")
    print("Saved plot to training_statistics.png")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE, dtype=torch.long)
            logits = model(images)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

    log.info(f"Test accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    train()

