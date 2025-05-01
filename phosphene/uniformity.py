import torch
import torch.nn.functional as F

class DynamicAmplitudeNormalizer:
    """
    Iteratively adjusts electrode amplitudes so that each phosphene 
    has a similar perceived brightness in the simulated percept.
    
    (See documentation above.)
    """
    def __init__(
        self,
        simulator,
        base_size=1,
        scale=0.05,
        A_min=1e-7,
        A_max=1e-3,
        learning_rate=0.5,
        steps=5,
        target=None,
        center=(0,0)
    ):
        self.simulator = simulator
        # Store electrode coordinates as torch tensors on CPU (they will be moved later as needed)
        self.phos_x = torch.tensor(simulator.coordinates._x, dtype=torch.float32)
        self.phos_y = torch.tensor(simulator.coordinates._y, dtype=torch.float32)
        self.n_phos = len(self.phos_x)

        # Read bounding coordinates from simulator parameters (assuming symmetric FoV about 0)
        fov = simulator.params['run']['view_angle']
        half_fov = fov / 2.0
        self.x_min = -half_fov
        self.x_max = +half_fov
        self.y_min = -half_fov
        self.y_max = +half_fov

        self.base_size = base_size
        self.scale = scale
        self.center = center
        self.A_min = A_min
        self.A_max = A_max
        self.learning_rate = learning_rate
        self.steps = steps
        self.target = target

        self.weights = torch.ones(self.n_phos, dtype=torch.float32)
        self.loss_history = None

    def run(self, stim_init: torch.Tensor, verbose=False) -> torch.Tensor:
        stim = stim_init.clone()
        self.loss_history = []  # To store loss values for each iteration

        # Optionally use tqdm for iteration progress
        iterator = range(self.steps)
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Iterations", unit="iter")

        for step_idx in iterator:
            self.simulator.reset()
            phos_image = self.simulator(stim)  # shape: [H, W], a torch.Tensor
            # Ensure phos_image is 2D [H, W]
            if phos_image.ndim == 3:
                phos_image = phos_image[0]
                

            # Use the vectorized brightness measurement (see _measure_brightness_vectorized below)
            brightness = self._measure_brightness(phos_image)
            
            # Choose target T
            if self.target is None:
                nonzero = brightness[brightness > 1e-12]
                T = torch.mean(nonzero) if len(nonzero) > 0 else 1.0
            else:
                T = self.target

            loss = torch.mean((brightness - T)**2)
            self.loss_history.append(loss.item())
            if verbose:
                iterator.set_description(f"Loss = {loss.item():.4f}")

            # Vectorize the electrode update as well
            # Avoid division by zero by adding a small epsilon.
            epsilon = 1e-12
            ratio = T / (brightness + epsilon)
            stim = stim * (1.0 + self.learning_rate * (ratio - 1.0))
            stim = torch.clamp(stim, self.A_min, self.A_max)

        self.weights = stim / self.A_max
        return stim

    def _measure_brightness(self, phos_image: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized brightness measurement.
        For each electrode, extract a patch from the phosphene image around the electrode's
        pixel location. The patch size is determined by:
        
            half_n = round(base_size + scale * r)
        
        where r is the radial distance (in degrees) from self.center.
        
        This implementation uses torch.nn.functional.grid_sample to extract patches in parallel.
        
        Returns:
        brightness_tensor : torch.Tensor of shape [n_phos]
            The brightness value for each electrode.
        """
        # Ensure phos_image is 2D [H, W]
        if phos_image.ndim == 3:
            phos_image = phos_image[0]

        # Add batch and channel dimensions: shape (1, 1, H, W)
        phos_image = phos_image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        H, W = phos_image.shape[-2:]
        device = phos_image.device

        # Move electrode coordinates to the proper device and type
        phos_x = self.phos_x.to(device=device, dtype=torch.float32)  # shape: (N,)
        phos_y = self.phos_y.to(device=device, dtype=torch.float32)  # shape: (N,)
        N = self.n_phos

        # Add a small epsilon to avoid division by zero in case the FoV is zero or very small.
        eps = 1e-6

        # Convert electrode coordinates (in degrees) to pixel coordinates.
        # (x_deg - x_min) / (x_max - x_min) * (W - 1)
        px = (phos_x - self.x_min) / ((self.x_max - self.x_min) + eps) * (W - 1)
        py = (phos_y - self.y_min) / ((self.y_max - self.y_min) + eps) * (H - 1)

        # Compute radial distance for each electrode (in degrees)
        cx, cy = self.center
        rx = phos_x - cx
        ry = phos_y - cy
        r = torch.sqrt(rx**2 + ry**2)  # shape: (N,)

        # Compute half patch size for each electrode: half_n = round(base_size + scale * r)
        half_n = torch.round(self.base_size + self.scale * r).to(torch.int64)
        half_n = torch.clamp(half_n, min=1)
        
        # Use the maximum half-size among electrodes to define a uniform patch size.
        H_max = int(half_n.max().item())
        K = 2 * H_max + 1  # Patch size for all electrodes

        # Create a base grid of offsets of shape (K, K) ranging from -H_max to H_max
        linspace = torch.linspace(-H_max, H_max, steps=K, device=device)
        y_offsets, x_offsets = torch.meshgrid(linspace, linspace, indexing='ij')  # Both shape: (K, K)

        # Expand the base grid to each electrode: for each electrode, the grid is centered at (px,py)
        px_exp = px.view(N, 1, 1)  # (N,1,1)
        py_exp = py.view(N, 1, 1)  # (N,1,1)
        grid_x = px_exp + x_offsets  # (N, K, K)
        grid_y = py_exp + y_offsets  # (N, K, K)
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (N, K, K, 2)

        # Convert grid from pixel coordinates to normalized coordinates in [-1, 1]
        grid[..., 0] = grid[..., 0] / (W - 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / (H - 1) * 2 - 1

        # Expand phos_image along the batch dimension so that we have one copy per electrode.
        phos_image_exp = phos_image.expand(N, -1, -1, -1)  # (N, 1, H, W)

        # Use grid_sample to extract patches.
        # Set padding_mode='zeros' to avoid NaNs for out-of-bound coordinates.
        patches = F.grid_sample(phos_image_exp, grid, mode='bilinear', align_corners=True, padding_mode='zeros')
        patches = patches.squeeze(1)  # (N, K, K)

        # Create a binary mask for valid pixels for each electrode.
        # For electrode i, valid if |offset| <= half_n[i] for both x and y.
        abs_x = x_offsets.unsqueeze(0).expand(N, -1, -1)  # (N, K, K)
        abs_y = y_offsets.unsqueeze(0).expand(N, -1, -1)  # (N, K, K)
        half_n_exp = half_n.view(N, 1, 1).float()  # (N,1,1)
        mask = ((abs_x <= half_n_exp) & (abs_y <= half_n_exp)).float()  # (N, K, K)

        # Compute the masked average brightness per electrode.
        weighted_sum = (patches * mask).view(N, -1).sum(dim=1)
        valid_counts = mask.view(N, -1).sum(dim=1)
        brightness_tensor = weighted_sum / (valid_counts + 1e-12)  # (N,)

        return brightness_tensor