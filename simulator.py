import torch
from dynaphos import utils, cortex_models
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from phosphene.uniformity import DynamicAmplitudeNormalizer

def build_simulator(cfg):
    """
    Initialize the phosphene simulator with parameters and coordinates.

    Args:
        cfg (dict): Configuration dictionary with keys:
            - general.params: path to params.yaml
            - general.n_phosphenes: number of electrodes/phosphenes

    Returns:
        simulator (PhospheneSimulator): Initialized simulator instance.
    """
    # Load simulator parameters
    params = utils.load_params(cfg['general']['params'])
    # Sample phosphene coordinates
    n_phos = cfg['general']['n_phosphenes']
    coords = cortex_models.get_visual_field_coordinates_probabilistically(params, n_phos)
    # Instantiate simulator
    sim = PhospheneSimulator(params, coords)
    return sim


def compute_stim_weights(simulator, cfg):
    """
    Optionally normalize electrode amplitudes for uniform perceived brightness.

    Args:
        simulator (PhospheneSimulator): Simulator to use for sampling.
        cfg (dict): Configuration dictionary with keys under 'simulator':
            - simulator.use_normalization (bool): whether to apply normalization
            - simulator.base_size (int)
            - simulator.scale (float)
            - simulator.A_min (float)
            - simulator.A_max (float)
            - simulator.learning_rate (float)
            - simulator.steps (int)
            - simulator.target (float or None)

    Returns:
        stim_weights (torch.Tensor): Shape (1, n_phosphenes), or None if disabled.
    """
    if not cfg['simulator'].get('use_normalization', False):
        # If normalization disabled, return uniform weights
        return torch.ones(1, simulator.num_phosphenes, device=cfg['general']['device'])

    # Retrieve amplitude and thresholds
    amplitude = simulator.params['sampling']['stimulus_scale']

    # Set up normalizer
    normalizer = DynamicAmplitudeNormalizer(
        simulator=simulator,
        base_size=3,
        scale=0.0001,
        A_min=0,
        A_max=amplitude,
        learning_rate=0.002,
        steps=2000,
        target=None  # or a specific float
    )
    # Initialize stimulation vector
    batch_size = cfg['train']['batch_size']
    stim_init = amplitude * torch.ones(batch_size, simulator.num_phosphenes, device=cfg['general']['device'])
    # Run normalization to adjust amplitudes
    normalizer.run(stim_init, verbose=True)
    # Return final weights
    return normalizer.weights.unsqueeze(0).to(cfg['general']['device'])