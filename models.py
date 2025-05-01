import torch
import torch.nn as nn
import torch.nn.functional as F

from phosphene.density import VisualFieldMapper
from components.modulated_conv2d import UnifiedInputModulation

# --- Helper blocks ---

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3))                  # (B, C)
        y = self.fc1(y).relu()
        y = self.fc2(y).sigmoid()           # (B, C)
        return x * y.view(b, c, 1, 1)

class DilatedResidualBlock(nn.Module):
    """Residual block with parallel dilations for multi-scale context"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=False)
        self.bn    = nn.BatchNorm2d(channels)
        self.relu  = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x) + self.conv2(x) + self.conv3(x)
        out = self.bn(out)
        out = out + x
        return self.relu(out)

class AttentionGate(nn.Module):
    """Spatial attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, g, x):
        # g: gating signal (decoder feature), x: skip connection feature
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Decoder(nn.Module):
    """
    Attention-UNet + Dilated-Residual decoder
    In:  (B,1,256,256) phosphene map
    Out: (B,1,256,256) reconstructed grayscale
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Down path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            SEBlock(32)
        )
        self.pool1 = nn.MaxPool2d(2)  # ->128

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            SEBlock(64)
        )
        self.pool2 = nn.MaxPool2d(2)  # ->64

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DilatedResidualBlock(64),
            DilatedResidualBlock(64),
            SEBlock(64)
        )

        # Up path
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # ->128
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            SEBlock(64)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # ->256
        self.att1 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            SEBlock(32)
        )

        # Final projection
        self.final = nn.Conv2d(32, out_channels, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)      # (B,32,256,256)
        p1 = self.pool1(e1)    # (B,32,128,128)
        e2 = self.enc2(p1)     # (B,64,128,128)
        p2 = self.pool2(e2)    # (B,64,64,64)
        # Bottleneck
        bn = self.bottleneck(p2)  # (B,64,64,64)
        # Decode
        u2 = self.up2(bn)         # (B,64,128,128)
        c2 = self.att2(g=u2, x=e2)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2)         # (B,32,256,256)
        c1 = self.att1(g=u1, x=e1)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))
        # Output
        out = self.final(d1)      # (B,1,256,256)
        return self.out_act(out)

# Factory

def get_decoder(cfg):
    """
    Returns a Decoder instance based on config:
    cfg['model']['in_channels'], cfg['model']['out_channels']
    """
    in_ch = cfg.get('model', {}).get('in_channels', 1)
    out_ch = cfg.get('model', {}).get('out_channels', 1)
    return Decoder(in_channels=in_ch, out_channels=out_ch)


def build_modulation_layer(cfg, simulator):
    """
    Construct a UnifiedInputModulation layer based on cortical or KDE density.

    Args:
        cfg (dict): Configuration dict with 'modulation' settings.
        simulator (PhospheneSimulator): Simulator to provide mapping.

    Returns:
        nn.Module: A modulation layer ready for use.
    """
    # Instantiate mapper
    mapper = VisualFieldMapper(simulator=simulator)
    total_phos = cfg['general']['n_phosphenes']
    method = cfg['modulation'].get('method', 'cortical')  # 'cortical' or 'kde'

    # Build density map
    if method == 'cortical':
        density = mapper.build_density_map_cortical(total_phos)
    else:
        k = cfg['modulation'].get('kde_k', 16)
        alpha = cfg['modulation'].get('kde_alpha', 1.0)
        density = mapper.build_density_map_kde(k, alpha, total_phos)

    # Convert to sigma map in pixel space
    sigma_map = mapper.build_sigma_map_from_density(density, space='pixel')
    device = cfg['general']['device']
    sigma_tensor = torch.tensor(sigma_map, device=device, dtype=torch.float32)

    # Create modulation layer
    mod_cfg = cfg['modulation']
    layer = UnifiedInputModulation(
        kernel_size=mod_cfg['kernel_size'],
        kernel_type=mod_cfg.get('kernel_type', 'log'),
        sigma_map=sigma_tensor,
        dilation=mod_cfg.get('dilation', 1),
        padding_mode=mod_cfg.get('padding_mode', 'reflect')
    )
    # Move to device and set eval
    layer = layer.to(device)
    layer.eval()
    return layer