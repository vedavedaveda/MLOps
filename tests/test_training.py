import ml_ops_project.train as train
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset


def make_fake_datasets(number_of_classes: int):
    """Small fake dataset, shaped like CIFAR: (N, 3, 32, 32)."""
    torch.manual_seed(0)

    train_images = torch.randn(8, 3, 32, 32)
    train_targets = torch.randint(0, number_of_classes, (8,))

    test_images = torch.randn(4, 3, 32, 32)
    test_targets = torch.randint(0, number_of_classes, (4,))

    return (
        TensorDataset(train_images, train_targets),
        TensorDataset(test_images, test_targets),
    )


def make_test_config(number_of_classes: int = 3):
    """Minimal config matching what train.py expects."""
    return OmegaConf.create(
        {
            "experiment": {
                "model": {"number_of_classes": number_of_classes},
                "training": {
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "epochs": 1,
                    "log_every_n_steps": 999999,  # keep output quiet
                },
            }
        }
    )


def setup_wandb_mocks(monkeypatch):
    """Setup all wandb-related mocks to prevent actual wandb calls during tests."""
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(train.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.wandb, "finish", lambda: None)

    # Mock wandb.Artifact
    class FakeArtifact:
        def __init__(self, *args, **kwargs):
            pass

        def add_file(self, *args, **kwargs):
            pass

    monkeypatch.setattr(train.wandb, "Artifact", FakeArtifact)
    monkeypatch.setattr(train.wandb, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.wandb, "Image", lambda *args, **kwargs: None)


def test_optimizer_step_is_called(tmp_path, monkeypatch):
    """Checks that the training loop actually performs optimization steps."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(train, "get_datasets", lambda: make_fake_datasets(3))

    # Setup wandb mocks
    setup_wandb_mocks(monkeypatch)

    cfg = make_test_config(number_of_classes=3)

    step_counter = {"count": 0}
    OriginalAdam = train.torch.optim.Adam

    class CountingAdam(OriginalAdam):
        def step(self, closure=None):
            step_counter["count"] += 1
            return super().step(closure=closure)

    monkeypatch.setattr(train.torch.optim, "Adam", CountingAdam)

    train.train.__wrapped__(cfg)

    assert step_counter["count"] > 0


def test_training_changes_model_weights(tmp_path, monkeypatch):
    """Checks that training produces different weights than a fresh initialization."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(train, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(train, "get_datasets", lambda: make_fake_datasets(3))

    # Setup wandb mocks
    setup_wandb_mocks(monkeypatch)

    cfg = make_test_config(number_of_classes=3)

    # Make a baseline model with the SAME initialization as the one used in training
    torch.manual_seed(123)
    baseline_model = train.CNN(number_of_classes=3)
    baseline_state = {k: v.detach().clone() for k, v in baseline_model.state_dict().items()}

    # Reset seed so the training model starts identically
    torch.manual_seed(123)

    saved = {}

    def fake_save(state_dict, path):
        saved["state_dict"] = state_dict

    monkeypatch.setattr(train.torch, "save", fake_save)

    train.train.__wrapped__(cfg)

    trained_state = saved["state_dict"]

    any_parameter_changed = any(
        not torch.allclose(baseline_state[name], trained_state[name]) for name in baseline_state
    )

    assert any_parameter_changed
