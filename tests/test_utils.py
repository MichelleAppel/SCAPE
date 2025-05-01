import torch
import numpy as np
import pytest
from matplotlib.figure import Figure

from utils import (
    robust_percentile_normalization,
    generate_phosphenes,
    visualize_training_sample,
)

# Dummy simulator for testing
class DummySim:
    def __init__(self, n_phos: int, H_out: int, W_out: int):
        self.n_phos = n_phos
        self.params = {
            'sampling': {'stimulus_scale': 1.0},
            'thresholding': {'rheobase': 0.0}
        }
        self.H_out = H_out
        self.W_out = W_out

    def reset(self):
        pass

    def sample_stimulus(self, pre, rescale=True):
        B = pre.shape[0]
        # Return a constant stimulation vector of ones
        return torch.ones(B, self.n_phos)

    def __call__(self, stim):
        B = stim.shape[0]
        # Return a constant phosphene image
        return torch.ones(B, self.H_out, self.W_out)


def test_robust_percentile_normalization_simple():
    # Given a known electrode vector and params
    electrode = torch.tensor([0.0, 0.5, 1.0, 1.5])
    amplitude = 2.0
    threshold = 0.5

    norm = robust_percentile_normalization(
        electrode,
        amplitude,
        threshold,
        low_perc=0.0,
        high_perc=100.0,
        gamma=1.0
    )
    # above = [0,0,0.5,1.0] -> scaled by amplitude -> [0,0,1.0,2.0]
    expected = torch.tensor([0.0, 0.0, 1.0, 2.0])
    assert torch.allclose(norm, expected, atol=1e-6)


@pytest.mark.parametrize("method", ['grayscale', 'LoG', 'canny', 'random'])
def test_generate_phosphenes_shapes(method):
    B, C, H, W = 2, 3, 4, 4
    batch = {'image': torch.rand(B, C, H, W)}
    cfg = {
        'dataset': {'processing': method},
        'general': {'device': 'cpu'}
    }

    n_phos, H_out, W_out = 10, 2, 2
    sim = DummySim(n_phos, H_out, W_out)
    stim_weights = torch.ones(1, n_phos)
    modulation_layer = lambda x: x  # identity

    pre, phos = generate_phosphenes(batch, sim, stim_weights, cfg, modulation_layer)

    assert pre.shape == (B, 1, H, W)
    assert phos.shape == (B, 1, H_out, W_out)


def test_visualize_training_sample_returns_figure():
    # Create dummy inputs
    batch = {'image': torch.rand(1, 3, 8, 8)}
    stimulus   = torch.rand(1, 1, 8, 8)
    phos       = torch.rand(1, 1, 8, 8)
    recon      = torch.rand(1, 1, 8, 8)
    losses     = [0.1, 0.2, 0.3]

    fig = visualize_training_sample(
        batch=batch,
        stimulus=stimulus,
        phosphene_inputs=phos,
        reconstructions=recon,
        losses=losses,
        epoch=1,
        step=1
    )

    assert isinstance(fig, Figure)
    # Expect 5 subplots: input, stimulus, phosphenes, reconstruction, loss
    assert len(fig.axes) == 5
