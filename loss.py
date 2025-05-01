import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


class HybridLoss(torch.nn.Module):
    """
    Hybrid loss combining Mean Squared Error (MSE) and Structural Similarity (SSIM).

    Loss = alpha * MSE + beta * (1 - SSIM)

    Args:
        alpha (float): weight for MSE term.
        beta (float): weight for SSIM term.
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid loss between reconstruction and target.

        Args:
            recon (torch.Tensor): reconstructed image, in [0, 1].
            target (torch.Tensor): ground-truth image, in [0, 1].

        Returns:
            torch.Tensor: scalar loss.
        """
        # MSE term
        mse_loss = F.mse_loss(recon, target)

        # SSIM term (similarity in [0,1])
        # Note: pytorch_msssim.ssim returns average similarity
        ssim_val = ssim(recon, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        # Combine
        return self.alpha * mse_loss + self.beta * ssim_loss