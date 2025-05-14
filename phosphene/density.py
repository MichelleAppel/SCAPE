import numpy as np
import torch
from scipy.spatial.distance import cdist

def get_cortical_magnification(r, params):
    """
    A simple function to compute cortical magnification M(r) for each radius r.
    This is just an example using 'dipole' or 'monopole'.
    """
    model = params.get("model", "dipole")
    a, b, k = params["a"], params["b"], params["k"]

    if model == "monopole":
        return k / (r + a)
    elif model == "dipole":
        return k * (1.0/(r + a) - 1.0/(r + b))
    else:
        raise ValueError(f"Unsupported cortical magnification model '{model}'.")

class VisualFieldMapper:
    """
    A class to build phosphene density maps (and corresponding sigma maps)
    either from a cortical-magnification model or by adaptive-KDE on actual
    phosphene coordinates.
    """

    def __init__(
        self,
        simulator,
        config=None
    ):
        if simulator is None:
            raise ValueError("A simulator must be provided to the VisualFieldMapper.")
        self.simulator = simulator
        self.config = config
        self.image_size = self.config['dataset']['imsize'] if config else self.simulator.params['run']['resolution']
        self.view_angle = self.simulator.params['run']['view_angle']
        self.min_ecc = self.simulator.params['run'].get('min_angle', 0.001)
        self.phos_x = self.simulator.coordinates._x
        self.phos_y = self.simulator.coordinates._y
        self.cortex_params = self.simulator.params['cortex_model']

    def build_density_map_cortical(
        self, 
        total_phosphenes=1024
    ) -> np.ndarray:
        """
        Build a 2D density map from the cortical magnification approach:
          density(r) = M(r)/(2 pi r)
        scaled so total # of phosphenes over the entire field = total_phosphenes.
        Returns a 2D np.ndarray shape [H, W] in visual-field space.
        """

        # build a grid of [-view_angle/2.. +view_angle/2]
        half_fov = self.view_angle / 2.0
        x = np.linspace(-half_fov, half_fov, self.image_size[0])
        y = np.linspace(-half_fov, half_fov, self.image_size[1])
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        r[r < self.min_ecc] = self.min_ecc  # avoid division by zero

        # cortical magnification
        M = get_cortical_magnification(r, self.cortex_params)
        density_map = M / (2.0 * np.pi * r)

        # compute exact area per pixel from grid spacing
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        area_per_pixel = dx * dy

        # normalize so integral equals total_phosphenes
        current_sum = density_map.sum() * area_per_pixel
        scaling_factor = total_phosphenes / current_sum
        density_map *= scaling_factor

        # optional check
        final_sum = density_map.sum() * area_per_pixel
        if not np.isclose(final_sum, total_phosphenes, atol=1e-3):
            print(f"[Warning] Density sum mismatch: {final_sum:.4f} vs {total_phosphenes}")

        return density_map

    def build_density_map_kde(
        self,
        k=10,
        alpha=1.0,
        total_phosphenes=1024
    ) -> np.ndarray:
        """
        Build a 2D density map from an adaptive KDE of self.phos_x, self.phos_y.
        """
        if self.phos_x is None or self.phos_y is None:
            raise ValueError("No phosphene coords found. Cannot do KDE approach.")
        coords = np.column_stack((self.phos_x, self.phos_y))

        half_fov = self.view_angle / 2.0
        x = np.linspace(-half_fov, half_fov, self.image_size[0])
        y = np.linspace(-half_fov, half_fov, self.image_size[1])
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        
        density_vals = self._adaptive_kde_2d(coords, grid_points, k, alpha)
        density_map = density_vals.reshape(self.image_size[0], self.image_size[1])

        # compute exact area per pixel
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        area_per_pixel = dx * dy

        # normalize so integral equals total_phosphenes
        current_sum = density_map.sum() * area_per_pixel
        scaling_factor = total_phosphenes / current_sum
        density_map *= scaling_factor

        return density_map

    def build_sigma_map_from_density(self, density_map, space="fov", beta=0.55):
        dm = np.clip(density_map, 1e-16, None)

        # σ_fov derived from desired frequency as a fraction of Nyquist
        coeff = 1.0 / (np.pi * np.sqrt(2) * beta * 0.5)  # = 2 / (π √2 β)
        sigma_fov = coeff / np.sqrt(dm)

        if space == "fov":
            return sigma_fov
        elif space == "pixel":
            half_fov = self.view_angle / 2.0
            x = np.linspace(-half_fov, half_fov, self.image_size[0])
            dx = x[1] - x[0]
            pix_per_deg = 1.0 / dx
            return sigma_fov * pix_per_deg
        else:
            raise ValueError("space must be 'fov' or 'pixel'")


    def _adaptive_kde_2d(self, phos_coords, grid_coords, k=10, alpha=1.0):
        dist_matrix = cdist(phos_coords, phos_coords)
        dist_sorted = np.sort(dist_matrix, axis=1)
        kth_dist = dist_sorted[:, k]
        local_bandwidths = alpha * kth_dist

        M = len(grid_coords)
        density = np.zeros(M, dtype=np.float64)
        for i in range(M):
            d = np.linalg.norm(phos_coords - grid_coords[i], axis=1)
            h = local_bandwidths
            kernel_val = np.exp(-0.5 * (d / h)**2) / (2.0 * np.pi * (h**2))
            density[i] = kernel_val.sum()
        return density
