import pytest
import torch

from ml_ops_project.model import CNN


def test_model_initialization():
    """Test that model initializes with default and custom number of classes."""
    model_default = CNN()
    assert model_default is not None

    model_custom = CNN(number_of_classes=10)
    assert model_custom is not None


def test_forward_pass_shape():
    """Test that forward pass returns correct output shape."""
    model = CNN(number_of_classes=5)
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)

    assert output.shape == (batch_size, 5)


def test_forward_pass_default_classes():
    """Test forward pass with default number of classes."""
    model = CNN()
    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert output.shape == (2, 2)


def test_different_batch_sizes():
    """Test forward pass with various batch sizes."""
    model = CNN(number_of_classes=3)

    for batch_size in [1, 8, 16, 32]:
        x = torch.randn(batch_size, 3, 32, 32)
        output = model(x)
        assert output.shape == (batch_size, 3)


def test_gradient_flow():
    """Test that gradients flow through the model during backprop."""
    model = CNN(number_of_classes=4)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_model_eval_mode():
    """Test model switches between train and eval modes."""
    model = CNN()

    model.train()
    assert model.training

    model.eval()
    assert not model.training


def test_wrong_input_shape():
    """Test that model raises error with incorrect input shape."""
    model = CNN()

    # Wrong spatial dimensions
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(RuntimeError):
        model(x)


def test_wrong_channels():
    """Test that model raises error with wrong number of channels."""
    model = CNN()

    # Wrong number of channels (grayscale instead of RGB)
    x = torch.randn(1, 1, 32, 32)
    with pytest.raises(RuntimeError):
        model(x)
