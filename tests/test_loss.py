import torch
import torch.nn.functional as F
import pytest
from loss import HybridLoss
from pytorch_msssim import ssim


def test_identical_inputs_zero_loss():
    """
    If reconstruction and target are identical, both MSE and (1-SSIM) should be zero, resulting in zero hybrid loss.
    """
    torch.manual_seed(0)
    # Create random input
    x = torch.rand(2, 1, 32, 32)
    loss_fn = HybridLoss(alpha=0.7, beta=0.3)
    loss_val = loss_fn(x, x)
    assert torch.isclose(loss_val, torch.tensor(0.0), atol=1e-6), \
        f"Expected zero loss for identical inputs, got {loss_val.item()}"


def test_mse_only_mode():
    """
    With alpha=1, beta=0, HybridLoss should equal plain MSE.
    """
    torch.manual_seed(1)
    recon = torch.rand(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    mse = F.mse_loss(recon, target)
    loss_fn = HybridLoss(alpha=1.0, beta=0.0)
    hybrid = loss_fn(recon, target)
    assert torch.allclose(hybrid, mse, atol=1e-6), \
        f"MSE-only mode: expected {mse.item()}, got {hybrid.item()}"


def test_ssim_only_mode():
    """
    With alpha=0, beta=1, HybridLoss should equal (1 - SSIM).
    """
    torch.manual_seed(2)
    recon = torch.rand(1, 1, 16, 16)
    target = torch.rand(1, 1, 16, 16)
    ssim_val = ssim(recon, target, data_range=1.0, size_average=True)
    expected = 1.0 - ssim_val
    loss_fn = HybridLoss(alpha=0.0, beta=1.0)
    hybrid = loss_fn(recon, target)
    assert torch.allclose(hybrid, expected, atol=1e-6), \
        f"SSIM-only mode: expected {expected.item()}, got {hybrid.item()}"


def test_combined_weights():
    """
    Hybrid loss with custom alpha and beta should equal alpha*mse + beta*(1-ssim).
    """
    torch.manual_seed(3)
    recon = torch.rand(1, 1, 8, 8)
    target = torch.rand(1, 1, 8, 8)
    alpha, beta = 0.4, 0.6
    mse = F.mse_loss(recon, target)
    ssim_val = ssim(recon, target, data_range=1.0, size_average=True)
    expected = alpha * mse + beta * (1.0 - ssim_val)
    loss_fn = HybridLoss(alpha=alpha, beta=beta)
    hybrid = loss_fn(recon, target)
    assert torch.allclose(hybrid, expected, atol=1e-6), \
        f"Combined mode: expected {expected.item()}, got {hybrid.item()}"
