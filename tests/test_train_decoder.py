import torch
import torch.nn as nn
import os
import pytest
import train_decoder as td

# Dummy dataset returning fixed images
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length=2):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return {'image': torch.ones(1, 1, 4, 4)}

# Dummy decoder that passes input through
class DummyDecoder(nn.Module):
    def forward(self, x):
        return x

# Dummy simulator for generate_phosphenes
class DummySim:
    def __init__(self):
        self.num_phosphenes = 5
        self.params = {
            'sampling': {'stimulus_scale': 1.0},
            'thresholding': {'rheobase': 0.0},
            'run': {'batch_size': 1}
        }
        self.device = 'cpu'
    def reset(self): pass
    def sample_stimulus(self, pre, rescale=True): return torch.ones(pre.size(0), self.num_phosphenes)
    def __call__(self, stim): return torch.ones(stim.size(0), 2, 2)

@pytest.fixture(autouse=True)
def patch_environment(monkeypatch, tmp_path):
    # Patch dataset
    monkeypatch.setattr(td, 'LaPaDataset', lambda cfg: DummyDataset())
    # Patch DataLoader
    monkeypatch.setattr(td, 'DataLoader', torch.utils.data.DataLoader)
    # Patch build_simulator and compute_stim_weights
    monkeypatch.setattr(td, 'build_simulator', lambda cfg: DummySim())
    monkeypatch.setattr(td, 'compute_stim_weights', lambda sim, cfg: torch.ones(1, sim.num_phosphenes))
    # Patch build_modulation_layer to no-op
    monkeypatch.setattr(td, 'build_modulation_layer', lambda cfg, sim: None)
    # Patch generate_phosphenes to identity mapping
    def fake_generate_phos(batch, simulator, stim_weights, cfg, layer):
        img = batch['image']
        return img.mean(1, keepdim=True), img
    monkeypatch.setattr(td, 'generate_phosphenes', fake_generate_phos)
    # Patch decoder factory
    monkeypatch.setattr(td, 'get_decoder', lambda cfg: DummyDecoder())
    # Patch HybridLoss to MSE
    monkeypatch.setattr(td, 'HybridLoss', lambda alpha, beta: nn.MSELoss())
    # Patch wandb
    class DummyWandb:
        def init(self, **kwargs): pass
        def watch(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
        def finish(self): pass
        def Image(self, x): return x
    monkeypatch.setattr(td, 'wandb', DummyWandb())

    # Provide minimal config
    cfg = {
        'general': {
            'project_name': 'test',
            'entity': 'test',
            'run_name': 'test',
            'device': 'cpu',
            'save_path': str(tmp_path),
            'model_log_freq': 1,
            'use_normalization': False
        },
        'dataset': {'processing': 'grayscale'},
        'train': {
            'batch_size': 1,
            'num_workers': 0,
            'lr': 1e-3,
            'weight_decay': 0,
            'epochs': 1
        },
        'loss': {
            'alpha': 0.5,
            'beta': 0.5
        }
    }
    return cfg
