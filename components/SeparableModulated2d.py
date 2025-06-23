import torch
import torch.nn.functional as F
from torch import nn

class SeparableModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        max_radius: int = 71,
        sigma_ratio: float = 1.6,
        stimulus_scale: float = 1.0,
        cps_half: float = 1.0,
        gauss_epsilon: float = 1e-3,
        padding_mode: str = 'reflect',
        use_learnable: bool = False,
        kernel_net_hidden: int = 64,
        sigma_map: torch.Tensor = None
    ):
        """
        Args:
          in_channels:   number of channels in input x
          max_radius:    maximum half-kernel size R
          sigma_ratio:   ratio sigma2/sigma1 for DoG
          stimulus_scale / cps_half: Unity scaling
          gauss_epsilon: threshold below which Gaussian weights are zeroed
          padding_mode:  padding mode for conv
          use_learnable: if True, use learnable MLP for 1D kernels (not DoG)
          kernel_net_hidden: hidden size for kernel_net
          sigma_map:     tensor of shape [H,W] or [1,H,W] or [B,H,W] to use for sigma
        """
        super().__init__()
        self.in_channels = in_channels
        self.R = max_radius
        self.sigma_ratio = sigma_ratio
        self.stimulus_scale = stimulus_scale
        self.cps_half = cps_half
        self.eps = gauss_epsilon
        self.padding_mode = padding_mode
        self.use_learnable = use_learnable

        # sigma_map as buffer
        if sigma_map is None:
            raise ValueError("sigma_map must be provided")
        if sigma_map.dim() == 2:
            sigma_map = sigma_map.unsqueeze(0)
        elif sigma_map.dim() == 3:
            pass
        else:
            raise ValueError("sigma_map must have shape [H,W] or [1,H,W] or [B,H,W]")
        self.register_buffer('sigma_map', sigma_map)

        # 1D coordinate grid for kernels
        coords = torch.arange(-self.R, self.R + 1, dtype=torch.float32)
        self.register_buffer('coords', coords.view(1, -1, 1, 1))  # [1,K,1,1]

        # learnable MLP (optional)
        if self.use_learnable:
            K = 2 * self.R + 1
            self.kernel_net = nn.Sequential(
                nn.Linear(1, kernel_net_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(kernel_net_hidden, 2 * K)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # expand sigma_map to [B,H,W]
        sigma_map = self.sigma_map
        if sigma_map.size(0) == 1:
            sigma_map = sigma_map.expand(B, -1, -1)

        # compute two sigmas
        sigma1 = sigma_map * self.stimulus_scale / self.cps_half  # [B,H,W]
        sigma2 = sigma1 * self.sigma_ratio                       # [B,H,W]
        s1 = sigma1.unsqueeze(1)  # [B,1,H,W]
        s2 = sigma2.unsqueeze(1)

        K = 2 * self.R + 1
        r = self.coords  # [1,K,1,1]

        # build analytic separable Gaussians for each sigma
        G1 = torch.exp(-0.5 * (r / s1).pow(2))   # [B,K,H,W]
        G2 = torch.exp(-0.5 * (r / s2).pow(2))
        # threshold small weights
        G1 = G1 * (G1 >= self.eps)
        G2 = G2 * (G2 >= self.eps)
        # normalize each so sum across kernel = 1
        sum1 = G1.sum(dim=1, keepdim=True) + 1e-8
        sum2 = G2.sum(dim=1, keepdim=True) + 1e-8
        G1 = G1 / sum1
        G2 = G2 / sum2

        # two-pass separable blur at sigma1
        x_pad = F.pad(x, (self.R, self.R, 0, 0), mode=self.padding_mode)
        patches_h = F.unfold(x_pad, (1, K))
        patches_h = patches_h.view(B, C, K, H, W)
        out1_h = (patches_h * G1.unsqueeze(1)).sum(dim=2)   # [B,C,H,W]
        out1_pad = F.pad(out1_h, (0, 0, self.R, self.R), mode=self.padding_mode)
        patches_v = F.unfold(out1_pad, (K, 1))
        patches_v = patches_v.view(B, C, K, H, W)
        out1 = (patches_v * G1.unsqueeze(1)).sum(dim=2)     # [B,C,H,W]

        # two-pass separable blur at sigma2
        x_pad = F.pad(x, (self.R, self.R, 0, 0), mode=self.padding_mode)
        patches_h2 = F.unfold(x_pad, (1, K))
        patches_h2 = patches_h2.view(B, C, K, H, W)
        out2_h = (patches_h2 * G2.unsqueeze(1)).sum(dim=2)
        out2_pad = F.pad(out2_h, (0, 0, self.R, self.R), mode=self.padding_mode)
        patches_v2 = F.unfold(out2_pad, (K, 1))
        patches_v2 = patches_v2.view(B, C, K, H, W)
        out2 = (patches_v2 * G2.unsqueeze(1)).sum(dim=2)

        # true DoG: difference of the two blurred images
        return out2 - out1
        
