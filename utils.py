import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np


def robust_percentile_normalization(electrode: torch.Tensor,
                                   amplitude: float,
                                   threshold: float,
                                   low_perc: float = 5.0,
                                   high_perc: float = 95.0,
                                   gamma: float = 1.0) -> torch.Tensor:
    """
    Normalize stimulation vector in a robust contrast-preserving way.

    Steps:
      1. Subtract threshold (values < threshold zeroed).
      2. Compute low/high percentiles of remaining values.
      3. Linearly scale between these percentiles to [0,1].
      4. Apply optional gamma correction.
      5. Scale by amplitude.
    """
    above = electrode - threshold
    above = torch.clamp(above, min=0.0)

    low_val = torch.quantile(above, low_perc / 100.0)
    high_val = torch.quantile(above, high_perc / 100.0)
    range_val = high_val - low_val + 1e-8

    norm = (above - low_val) / range_val
    norm = torch.clamp(norm, 0.0, 1.0)
    norm = norm.pow(gamma)
    return norm * amplitude


def generate_phosphenes(batch: dict,
                         simulator,
                         stim_weights: torch.Tensor,
                         cfg: dict,
                         modulation_layer = None) -> tuple:
    """
    Given a batch of images (and possibly contours), process and generate phosphenes.

    Returns:
      pre: Tensor (B,1,H,W) preprocessing output
      phos: Tensor (B,1,H_out,W_out) phosphene percepts
    """
    method = cfg['dataset']['processing']
    device = cfg['general']['device']

    images = batch['image'].to(device)
    B, C, H, W = images.shape

    # Prepare stimulus based on method
    if method == 'grayscale':
        pre = images.mean(dim=1, keepdim=True)
    elif method == 'LoG':
        gray = images.mean(dim=1, keepdim=True)
        if modulation_layer is None:
            raise ValueError("Modulation layer is required for 'LoG' method.")
        log = modulation_layer(gray)
        log = torch.clamp(log, 0.0, None)
        pre = (log - log.min()) / (log.max() - log.min() + 1e-8)
    elif method == 'canny':
        edges = []
        for img in images.cpu().numpy():
            c = cv2.Canny((img*255).transpose(1,2,0).astype(np.uint8),
                          100, 200)
            c = torch.tensor(c, device=device).unsqueeze(0).unsqueeze(0)
            edges.append(c)
        pre = torch.cat(edges, dim=0)
    elif method == 'random':
        rand = np.random.rand(H, W).astype(np.float32)
        pre = torch.tensor(rand, device=device).unsqueeze(0).unsqueeze(0).repeat(B,1,1,1)
    else:
        raise ValueError(f"Unknown processing method: {method}")

    # Generate stimulation vector
    simulator.reset()
    stim = simulator.sample_stimulus(pre, rescale=True)
    amplitude = simulator.params['sampling']['stimulus_scale']
    threshold = simulator.params['thresholding']['rheobase']
    stim = robust_percentile_normalization(stim, amplitude, threshold,
                                           low_perc=5, high_perc=95, gamma=2/3)
    stim = stim * stim_weights

    # Generate phosphenes
    simulator.reset()
    phos = simulator(stim)
    phos = phos.unsqueeze(1)

    return pre, phos


def visualize_training_sample(batch: dict,
                              stimulus: torch.Tensor,
                              phosphene_inputs: torch.Tensor,
                              reconstructions: torch.Tensor,
                              losses: list = None,
                              epoch: int = 0,
                              step: int = 0) -> plt.Figure:
    """
    Creates a matplotlib Figure showing:
      [0] Input image (grayscale)
      [1] Preprocessed stimulus
      [2] Phosphene input
      [3] Reconstruction
      [4] Loss history (optional)
    """
    figs = plt.figure(figsize=(15, 5))
    items = [batch['image'].mean(1, keepdim=True),
             stimulus, phosphene_inputs, reconstructions]
    titles = ['Input Image', 'Stimulus', 'Phosphenes', 'Reconstruction']

    # Add loss plot
    if losses is not None:
        items.append(torch.tensor(losses))
        titles.append('Loss History')

    for i, (item, title) in enumerate(zip(items, titles)):
        ax = figs.add_subplot(1, len(items), i+1)
        if i < 4:
            img = item[0].detach().cpu().squeeze()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f"{title}\n(E{epoch}, S{step})")
        else:
            ax.plot(item)
            ax.set_title(title)
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
    plt.tight_layout()
    return figs
