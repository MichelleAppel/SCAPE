import torch
import pytest
from models import Decoder, get_decoder


def test_decoder_forward_and_shape():
    """
    Test that Decoder produces an output of correct shape and value range.
    """
    batch_size = 2
    in_ch = 1
    out_ch = 1
    H, W = 256, 256

    # Create dummy input
    x = torch.rand(batch_size, in_ch, H, W)
    model = Decoder(in_channels=in_ch, out_channels=out_ch)

    # Forward pass
    y = model(x)

    # Check output shape
    assert y.shape == (batch_size, out_ch, H, W), \
        f"Expected shape {(batch_size, out_ch, H, W)}, got {y.shape}"

    # Check output value range [0,1]
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0), \
        "Decoder output values should be between 0 and 1"


def test_get_decoder_factory():
    """
    Test that get_decoder factory reads config correctly.
    """
    # Custom in/out channels
    cfg = {'model': {'in_channels': 3, 'out_channels': 2}}
    decoder = get_decoder(cfg)

    # Ensure correct class
    assert isinstance(decoder, Decoder), "Factory should return a Decoder instance"

    # Forward pass with matching channels
    x = torch.rand(1, 3, 256, 256)
    y = decoder(x)
    assert y.shape == (1, 2, 256, 256), \
        f"Expected shape (1,2,256,256), got {y.shape}"


def test_invalid_factory_config():
    """
    Test that get_decoder uses defaults if config keys are missing.
    """
    # Empty config should default to in=1, out=1
    decoder = get_decoder({})
    x = torch.rand(1, 1, 256, 256)
    y = decoder(x)
    assert y.shape == (1, 1, 256, 256), \
        f"Expected default shape (1,1,256,256), got {y.shape}"
