import numpy as np
import torch

class DynamicAmplitudeNormalizer:
    """
    Iteratively adjusts electrode amplitudes so that each phosphene 
    has a similar perceived brightness in the simulated percept.

    This version only needs the simulator object, and extracts:
      - electrode coordinates (phos_x, phos_y) in degrees
      - bounding region (x_min..x_max, y_min..y_max)
    from the simulator. Then it calls simulator(stim) each iteration 
    to measure brightness around each electrode in the resulting image.
    
    Example usage:
    -------------
    normalizer = DynamicAmplitudeNormalizer(
        simulator=simulator,
        base_size=1,
        scale=0.5,
        A_min=1e-7,
        A_max=1e-3,
        learning_rate=0.05,
        steps=50,
        target=None
    )
    # Start with uniform amplitudes:
    stim_init = amplitude * torch.ones(simulator.num_phosphenes).cuda()
    stim_final = normalizer.run(stim_init)
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
        """
        Parameters
        ----------
        simulator : object
            Your simulator instance, e.g. PhospheneSimulator, which must provide:
              - .phos_x, .phos_y : arrays (or lists) of electrode coords in degrees
              - .params['run']['view_angle'] : field of view in degrees
              - .reset() and .__call__(stim_vector) -> 2D image (torch Tensor)
        base_size : int
            Minimal half-size of the patch for measuring brightness.
        scale : float
            Factor controlling how patch size grows with radial distance from 'center'.
        A_min, A_max : float
            Clamping range for amplitudes.
        learning_rate : float
            Partial update factor for amplitude correction.
        steps : int
            Number of iterations for the uniformization procedure.
        target : float or None
            If not None, all electrodes aim for this brightness. 
            If None, we compute an average from the nonzero brightness each iteration.
        center : tuple
            (cx, cy), the reference center for measuring radial distance. 
            Often (0,0) for visual field center.
        """
        self.simulator = simulator
        self.phos_x = np.array(simulator.coordinates._x)  # or however you store them
        self.phos_y = np.array(simulator.coordinates._y)
        self.n_phos = len(self.phos_x)

        # Read bounding coords from simulator params (assuming e.g. ±(FoV/2))
        fov = simulator.params['run']['view_angle']
        half_fov = fov / 2.0
        # We'll assume the center is (0,0) => so x_min..x_max = -half_fov..+half_fov
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

    def run(self, stim_init: torch.Tensor) -> torch.Tensor:
        """
        Runs the iterative procedure for 'steps' iterations.
        
        Args:
          stim_init : torch.Tensor of shape (n_phos,) 
              The initial electrode amplitudes (e.g. all equal to some amplitude).
        
        Returns:
          stim_final : torch.Tensor of shape (n_phos,) 
              The updated amplitude vector that yields a more uniform brightness.
        """
        stim = stim_init.clone()

        for step_idx in range(self.steps):
            # Generate the current phosphene image
            self.simulator.reset()
            phos_image = self.simulator(stim)  # shape [H,W], a torch Tensor

            # measure brightness
            brightness = self._measure_brightness(phos_image)

            # choose target T
            if self.target is None:
                nonzero = brightness[brightness > 1e-12]
                T = np.mean(nonzero) if len(nonzero) > 0 else 1.0
            else:
                T = self.target

            # partial update
            new_stim = []
            for i in range(self.n_phos):
                oldA = stim[i].item()
                meas = brightness[i]
                if meas < 1e-12:
                    # small => boost
                    updated = oldA * 1.1
                else:
                    ratio = T / meas
                    updated = oldA * (1.0 + self.learning_rate * (ratio - 1.0))

                # clamp
                updated = max(self.A_min, min(self.A_max, updated))
                new_stim.append(updated)

            stim = torch.tensor(new_stim, device=stim.device, dtype=stim.dtype)

        self.weights = stim / self.A_max

        return stim

    def _measure_brightness(self, phos_image: torch.Tensor) -> np.ndarray:
        """
        For each electrode, measure brightness in phos_image around 
        a patch whose size depends on radial distance from self.center.
        
        Returns a NumPy array shape [n_phos].
        """
        # Convert to numpy
        if isinstance(phos_image, torch.Tensor):
            phos_image = phos_image.cpu().numpy()

        H, W = phos_image.shape
        brightness = np.zeros(self.n_phos, dtype=float)

        cx, cy = self.center

        def to_pixel_coords(x_deg, y_deg):
            px = (x_deg - self.x_min) / (self.x_max - self.x_min) * (W - 1)
            py = (y_deg - self.y_min) / (self.y_max - self.y_min) * (H - 1)
            return int(round(px)), int(round(py))

        for i in range(self.n_phos):
            # radial distance
            rx = self.phos_x[i] - cx
            ry = self.phos_y[i] - cy
            r_i = np.sqrt(rx**2 + ry**2)

            half_n = int(round(self.base_size + self.scale * r_i))
            half_n = max(half_n, 1)

            px, py = to_pixel_coords(self.phos_x[i], self.phos_y[i])

            vals = []
            for dx in range(-half_n, half_n+1):
                for dy in range(-half_n, half_n+1):
                    qx = px + dx
                    qy = py + dy
                    if 0 <= qx < W and 0 <= qy < H:
                        vals.append(phos_image[qy, qx])

            brightness[i] = np.mean(vals) if vals else 0.0

        return brightness
