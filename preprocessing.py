import torch

def generate_phosphenes(batch, simulator, stim_weights, cfg, modulation_layer):
    """
    From a batch of raw images (and optional contour maps), produce:
      - preprocessed stimulus (LoG, contour, etc.)
      - phosphene image via simulator
    """
    # 1) Move inputs to device
    device = simulator.device
    imgs = batch['image'].to(device)                # (B, C, H, W)
    method = cfg['dataset']['processing']           # e.g. 'LoG'
    
    # 2) Preprocess
    if method == 'LoG':
        gray = imgs.mean(1, keepdim=True)          # (B,1,H,W)
        LoG  = modulation_layer(gray).clamp(min=0)
        pre  = (LoG - LoG.min())/(LoG.max()-LoG.min()+1e-12)
    elif method == 'contour':
        pre = batch['contour'].unsqueeze(1).to(device)
    else:
        raise ValueError(f"Unsupported method {method}")
    
    # 3) Sample stimulation vector
    B = pre.shape[0]
    # simulator.sample_stimulus expects (B, n_phos)
    stim = simulator.sample_stimulus(pre, rescale=True)   # (B, n_phos)
    # normalize per‑batch using your robust percentile fn
    normed = torch.stack([
        robust_percentile_normalization(
            stim[i], 
            simulator.params['sampling']['stimulus_scale'], 
            simulator.params['thresholding']['rheobase']
        ) for i in range(B)
    ]) * stim_weights                               # (B, n_phos)
    
    # 4) Forward to phosphene image
    simulator.reset()
    phos = simulator(normed)                         # (B, H, W)
    phos = phos.unsqueeze(1)                         # (B,1,H,W)
    
    return pre, phos

def robust_percentile_normalization(electrode, amplitude, threshold, low_perc=5, high_perc=95, gamma=1.0):
    """
    Normalize the stimulation vector in a robust, contrast-preserving way.

    This function performs the following steps:
      1. Subtracts a given threshold (values below threshold become 0).
      2. Computes low and high percentiles (e.g., 5th and 95th) of the resulting values.
      3. Linearly scales values between these percentiles to [0, 1], then applies an optional gamma correction.
      4. Scales the result to the desired amplitude.

    Args:
        electrode (torch.Tensor): Input stimulation values.
        amplitude (float): Desired maximum amplitude.
        threshold (float): Activation threshold (e.g. your activation_threshold).
        low_perc (float): Lower percentile (default 5).
        high_perc (float): Upper percentile (default 95).
        gamma (float): Gamma exponent for power-law scaling (default 1.0 means linear scaling).
        
    Returns:
        torch.Tensor: Normalized stimulation values, with zeros preserved.
    """
    # Subtract threshold and clamp to 0 so that values below threshold remain 0.
    above_thresh = electrode - threshold
    above_thresh = torch.clamp(above_thresh, min=0.0)
    
    # Compute robust lower and upper bounds using percentiles.
    # Use torch.quantile (available in PyTorch 1.7+; adjust if needed).
    low_val = torch.quantile(above_thresh, low_perc / 100.0)
    high_val = torch.quantile(above_thresh, high_perc / 100.0)
    
    # Avoid division by zero if high_val == low_val.
    range_val = high_val - low_val + 1e-8

    # Linearly map values in the range [low_val, high_val] to [0, 1]
    normalized = (above_thresh - low_val) / range_val
    normalized = torch.clamp(normalized, 0.0, 1.0)

    # Apply gamma correction to adjust contrast if needed.
    normalized = normalized ** gamma

    # Scale to the desired amplitude.
    normalized = normalized * amplitude

    return normalized


# Dilation is used for the reg-loss on the phosphene image: phosphenes do not have to map 1 on 1, small offset is allowed.
def dilation5x5(img, kernel=None):
    if kernel is None:
        kernel = torch.tensor([[[[0., 0., 1., 0., 0.],
                              [0., 1., 1., 1., 0.],
                              [1., 1., 1., 1., 1.],
                              [0., 1., 1., 1., 0.],
                              [0., 0., 1., 0., 0.]]]], requires_grad=False, device=img.device)
    return torch.clamp(torch.nn.functional.conv2d(img, kernel, padding=kernel.shape[-1]//2), 0, 1)

def dilation3x3(img, kernel=None):
    if kernel is None:
        kernel = torch.tensor([[[
                              [ 0, 1., 0.],
                              [ 1., 1., 1.],
                              [ 0., 1., 0.],]]], requires_grad=False, device=img.device)
    return torch.clamp(torch.nn.functional.conv2d(img, kernel, padding=kernel.shape[-1]//2), 0, 1)