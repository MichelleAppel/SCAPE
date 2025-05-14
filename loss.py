import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import piq


def get_loss(cfg):
    """
    Factory to retrieve loss modules based on configuration.
    Supported loss_function values:
      - HybridLoss
      - InformationWeightedSSIMLoss
      - MSE
      - SSIMLoss
      - LPIPSLoss
      - VGGPerceptualLoss
      - FSIMLoss
      - DISTSLoss
      - PieAPPLoss
      - PseudoHuberLoss
    For HybridLoss, cfg['loss'] must also specify:
      - losses: List[str] of loss names
      - weights: List[float] of same length
    Each individual loss may have its own nested config under cfg['loss'],
    e.g. cfg['loss']['ssim'] or cfg['loss']['lpips'], etc.
    """
    name = cfg['loss']['loss_function']
    if name == 'HybridLoss':
        return HybridLoss(
            loss_names=cfg['loss']['losses'],
            weights=cfg['loss']['weights'],
            cfg=cfg
        )
    elif name == 'InformationWeightedSSIMLoss':
        return InformationWeightedSSIMLoss(
            data_range=cfg['loss'].get('data_range', 1.0),
            reduction=cfg['loss'].get('reduction', 'mean')
        )
    elif name == 'MSE':
        return MSE(
            reduction=cfg['loss'].get('reduction', 'mean')
        )
    elif name in ('SSIM', 'SSIMLoss'):
        params = cfg['loss'].get('ssim', {})
        return SSIMLoss(**params)
    elif name in ('LPIPS', 'LPIPSLoss'):
        params = cfg['loss'].get('lpips', {})
        return LPIPSLoss(**params)
    elif name in ('VGGPerceptualLoss', 'ContentLoss'):
        params = cfg['loss'].get('vgg', {})
        return VGGPerceptualLoss(**params)
    elif name in ('FSIM', 'FSIMLoss'):
        params = cfg['loss'].get('fsim', {})
        return FSIMLoss(**params)
    elif name in ('DISTS', 'DISTSLoss'):
        params = cfg['loss'].get('dists', {})
        return DISTSLoss(**params)
    elif name in ('PieAPP', 'PieAPPLoss'):
        params = cfg['loss'].get('pieapp', {})
        return PieAPPLoss(**params)
    elif name in ('PseudoHuberLoss', 'PseudoHuber'):
        params = cfg['loss'].get('pseudo_huber', {})
        return PseudoHuberLoss(**params)
    else:
        raise ValueError(f"Unknown loss function: {name}.")


class HybridLoss(torch.nn.Module):
    """
    Hybrid loss combining multiple sub-losses.
    Loss = \sum_i weights[i] * loss_i(recon, target)

    Args:
        loss_names (List[str]): names of sub-losses to combine (must be supported by get_loss)
        weights (List[float]): weights for each sub-loss
        cfg (dict): full config dict (get_loss uses it to instantiate sub-losses)
    """
    def __init__(self, loss_names, weights, cfg):
        super().__init__()
        if len(loss_names) != len(weights):
            raise ValueError("Number of losses and weights must match")
        self.weights = weights
        # instantiate each loss via factory
        self.losses = torch.nn.ModuleList()
        for ln in loss_names:
            subcfg = {'loss': {**cfg['loss'], 'loss_function': ln}}
            self.losses.append(get_loss(subcfg))

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(recon, target)
        return total


class InformationWeightedSSIMLoss(torch.nn.Module):
    """
    Information-Weighted SSIM Loss (IW-SSIM).
    """
    def __init__(self, data_range: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = piq.InformationWeightedSSIMLoss(
            data_range=data_range,
            reduction=reduction
        )

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.9 * self.loss_fn(recon, target) + 0.1 * F.mse_loss(recon, target)


class MSE(torch.nn.Module):
    """
    Mean Squared Error Loss.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, target, reduction=self.reduction)


class SSIMLoss(torch.nn.Module):
    """
    Structural Similarity (SSIM) Loss.
    """
    def __init__(self,
                 kernel_size: int = 11,
                 kernel_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 downsample: bool = True,
                 reduction: str = 'mean',
                 data_range: float = 1.0):
        super().__init__()
        self.loss_fn = piq.SSIMLoss(
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
            downsample=downsample,
            reduction=reduction,
            data_range=data_range
        )

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(recon, target)


class LPIPSLoss(torch.nn.Module):
    """
    Learned Perceptual Image Patch Similarity Loss (LPIPS).
    """
    def __init__(self,
                 replace_pooling: bool = False,
                 distance: str = 'mse',
                 reduction: str = 'mean',
                 mean: list = None,
                 std: list = None):
        super().__init__()
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        self.loss_fn = piq.LPIPS(
            replace_pooling=replace_pooling,
            distance=distance,
            reduction=reduction,
            mean=mean,
            std=std
        )

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(recon, target)


class VGGPerceptualLoss(torch.nn.Module):
    """
    VGG-based perceptual content loss.
    """
    def __init__(self,
                 feature_extractor: str = 'vgg16',
                 layers: list = None,
                 weights: list = None,
                 replace_pooling: bool = False,
                 distance: str = 'mse',
                 reduction: str = 'mean',
                 mean: list = None,
                 std: list = None,
                 normalize_features: bool = False,
                 allow_layers_weights_mismatch: bool = False):
        super().__init__()
        layers = layers or ['relu3_3']
        weights = weights or [1.0]
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        self.loss_fn = piq.ContentLoss(
            feature_extractor=feature_extractor,
            layers=layers,
            weights=weights,
            replace_pooling=replace_pooling,
            distance=distance,
            reduction=reduction,
            mean=mean,
            std=std,
            normalize_features=normalize_features,
            allow_layers_weights_mismatch=allow_layers_weights_mismatch
        )

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(recon, target)


class FSIMLoss(torch.nn.Module):
    """
    Feature Similarity Index Measure Loss (FSIM).
    """
    def __init__(self,
                 reduction: str = 'mean',
                 data_range: float = 1.0,
                 chromatic: bool = True,
                 scales: int = 4,
                 orientations: int = 4,
                 min_length: int = 6,
                 mult: int = 2,
                 sigma_f: float = 0.55,
                 delta_theta: float = 1.2,
                 k: float = 2.0):
        super().__init__()
        self.loss_fn = piq.FSIMLoss(
            reduction=reduction,
            data_range=data_range,
            chromatic=chromatic,
            scales=scales,
            orientations=orientations,
            min_length=min_length,
            mult=mult,
            sigma_f=sigma_f,
            delta_theta=delta_theta,
            k=k
        )

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(recon, target)


class DISTSLoss(torch.nn.Module):
    """
    Deep Image Structure and Texture Similarity Loss (DISTS).
    """
    def __init__(self,
                 reduction: str = 'mean',
                 mean: list = None,
                 std: list = None):
        super().__init__()
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        self.loss_fn = piq.DISTS(
            reduction=reduction,
            mean=mean,
            std=std
        )

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(recon, target)


class PieAPPLoss(torch.nn.Module):
    """
    Perceptual Image-Error Assessment through Pairwise Preference Loss (PieAPP).
    """
    def __init__(self,
                 reduction: str = 'mean',
                 data_range: float = 1.0,
                 stride: int = 27,
                 enable_grad: bool = True):
        super().__init__()
        self.loss_fn = piq.PieAPP(
            reduction=reduction,
            data_range=data_range,
            stride=stride,
            enable_grad=enable_grad
        )

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(recon, target)

class PseudoHuberLoss(torch.nn.Module):
    """
    Pseudo-Huber loss.

    L_delta(x, y) = δ² * (sqrt(1 + ((x - y)/δ)²) - 1)

    Args:
        delta (float): transition point between quadratic and linear.
        reduction (str): 'mean' or 'sum'.
    """
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = recon - target
        abs_term = torch.sqrt(1.0 + (diff / self.delta) ** 2) - 1.0
        loss = (self.delta ** 2) * abs_term
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss