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

    Usage example:
    -------------
        # 1) Provide a simulator or the relevant config
        simulator = ...
        # OR config = {...}  # read from a YAML

        # 2) Create the builder
        builder = SigmaMapBuilder(simulator=simulator)

        # 3) Build a density map using cortical approach:
        density_model = builder.build_density_map_cortical()

        # 4) Convert that density to a sigma map (pixel space):
        sigma_pixels_model = builder.build_sigma_map_from_density(density_model, space="pixel")

        # 5) Or do an adaptive KDE approach:
        density_kde = builder.build_density_map_kde(k=16, alpha=1.0)
        sigma_pixels_kde = builder.build_sigma_map_from_density(density_kde, space="pixel")
    """

    def __init__(
        self,
        simulator=None,
        config=None,
        image_size=None,
        view_angle=None
    ):
        """
        You can either pass a 'simulator' (with .params) or provide a 'config' dictionary 
        manually. You can also specify image_size & view_angle if they are not in the config.
        
        The class can store relevant info for building density & sigma maps.

        Args:
            simulator : A simulator object that has 'params' describing 
                        run: { resolution, view_angle, etc. }
                        and phos_x, phos_y if you want to do the KDE approach.
            config : A dictionary containing the run settings if no simulator is used.
            image_size : (int) The image resolution (assuming square NxN).
            view_angle : (float) The total field-of-view in degrees.

        We'll store:
         - self.image_size
         - self.view_angle
         - self.min_ecc (optional)
         - self.phos_x, self.phos_y if simulator has them
        """
        # 1) read from simulator if provided
        self.simulator = simulator
        self.config = config or {}
        
        # If we have a simulator, read config from simulator.params
        if simulator is not None and hasattr(simulator, "params"):
            run_conf = simulator.params.get("run", {})
            self.image_size = run_conf.get("resolution", [256,256])[0]
            self.view_angle = run_conf.get("view_angle", 16.0)
            self.coordinates = getattr(simulator, "coordinates", None)
            self.phos_x = self.coordinates._x
            self.phos_y = self.coordinates._y
        else:
            # else read from self.config or from the direct args
            run_conf = self.config.get("run", {})
            self.image_size = image_size or run_conf.get("resolution",[256,256])[0]
            self.view_angle = view_angle or run_conf.get("view_angle", 16.0)
            self.phos_x = None
            self.phos_y = None

        # minimal eccentricity if needed
        self.min_ecc = run_conf.get("min_angle", 0.001)

        # some default cortical model params
        self.cortex_params = self.config.get("cortex_model", {
            "model":"dipole",
            "k":17.3,
            "a":0.75,
            "b":120
        })

    def build_density_map_cortical(
        self, 
        total_phosphenes=1024
    ) -> np.ndarray:
        """
        Build a 2D density map from the cortical magnification approach:
          density(r) = M(r)/(2 pi r)
        scaled so total # of phosphenes over the entire field = total_phosphenes.

        Returns a 2D np.ndarray shape [image_size, image_size] in visual-field space.
        """

        # build a grid of [-view_angle/2.. +view_angle/2]
        half_fov = self.view_angle / 2.0
        x = np.linspace(-half_fov, half_fov, self.image_size)
        y = np.linspace(-half_fov, half_fov, self.image_size)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        r[r < self.min_ecc] = self.min_ecc  # avoid division by zero

        # cortical magnification
        M = get_cortical_magnification(r, self.cortex_params)  # shape [image_size, image_size]
        # 2D density
        density_map = M / (2.0 * np.pi * r)

        # scale so the integral => total_phosphenes
        # approximate integral by sum(density_map)*(view_angle/image_size)**2
        # because each pixel covers (view_angle/image_size) degrees in x & y
        area_per_pixel = (self.view_angle / self.image_size)**2
        current_sum = (density_map.sum() * area_per_pixel)
        scaling_factor = total_phosphenes / current_sum
        density_map *= scaling_factor

        # optional check
        final_sum = density_map.sum() * area_per_pixel
        if not np.isclose(final_sum, total_phosphenes, atol=1e-3):
            print(f"[Warning] Density sum mismatch: {final_sum} vs {total_phosphenes}")

        return density_map

    def build_density_map_kde(
        self,
        k=10,
        alpha=1.0,
        total_phosphenes=1024
    ) -> np.ndarray:
        """
        Build a 2D density map from an adaptive KDE of self.phos_x, self.phos_y.
        We must have phos_x, phos_y available in the simulator.
        
        Args:
            k (int): k-th neighbor for local bandwidth
            alpha (float): scale factor on the bandwidth
            total_phosphenes: used for final scaling so integral => total_phosphenes

        Returns a 2D np.ndarray [image_size, image_size] in visual-field space 
        with the density in phosphenes/deg^2.
        """
        if self.phos_x is None or self.phos_y is None:
            raise ValueError("No phosphene coords found (phos_x, phos_y). "
                             "Cannot do KDE approach.")
        coords = np.column_stack((self.phos_x, self.phos_y))

        half_fov = self.view_angle / 2.0
        x = np.linspace(-half_fov, half_fov, self.image_size)
        y = np.linspace(-half_fov, half_fov, self.image_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        
        density_vals = self._adaptive_kde_2d(coords, grid_points, k, alpha)
        density_map = density_vals.reshape(self.image_size, self.image_size)

        # Scale so integral => total_phosphenes
        area_per_pixel = (self.view_angle / self.image_size)**2
        current_sum = density_map.sum() * area_per_pixel
        scale = total_phosphenes / current_sum
        density_map *= scale

        return density_map

    def build_sigma_map_from_density(
        self, density_map: np.ndarray, space="fov"
    ) -> np.ndarray:
        """
        Given a density_map in visual-field space, compute the LoG sigma map,
        either in 'fov' degrees or convert to 'pixel' space.

        sigma_fov = 1/(pi * sqrt(density_map))  # from your formula
        If space=='pixel', multiply sigma_fov by (n_phosphenes/view_angle).

        Returns an np.ndarray of the same shape as density_map.
        """
        # avoid zero in sqrt
        density_map_clamped = np.copy(density_map)
        density_map_clamped[density_map_clamped < 1e-16] = 1e-16


        sigma_fov = 1.0 / (np.pi * np.sqrt(density_map_clamped))

        if space == "fov":
            return sigma_fov
        elif space == "pixel":
            total_phosphenes = density_map.sum() * (self.view_angle / self.image_size)**2
            factor = total_phosphenes / self.view_angle
            sigma_px = sigma_fov * factor
            return sigma_px
        else:
            raise ValueError(f"Unknown space='{space}'. Use 'fov' or 'pixel'.")

    # ------------------------------------------------------------------------
    # A helper method for the adaptive KDE approach
    # ------------------------------------------------------------------------
    def _adaptive_kde_2d(self, phos_coords, grid_coords, k=10, alpha=1.0):
        """
        A naive adaptive-KDE approach. For each phosphene j, bandwidth h_j is 
        the distance to its k-th nearest neighbor times alpha.
        Then for each grid point, sum up gaussian contributions from each phosphene.

        Returns shape [M,] for M= len(grid_coords).
        """
        dist_matrix = cdist(phos_coords, phos_coords)  # NxN
        dist_sorted = np.sort(dist_matrix, axis=1)
        kth_dist = dist_sorted[:, k]  # shape (N,)
        local_bandwidths = alpha * kth_dist

        N = len(phos_coords)
        M = len(grid_coords)
        density = np.zeros(M, dtype=np.float64)

        for i in range(M):
            d = np.linalg.norm(phos_coords - grid_coords[i], axis=1)
            # each phosphene's kernel
            h = local_bandwidths
            kernel_val = np.exp(-0.5 * (d / h)**2) / (2.0 * np.pi * (h**2))
            density[i] = kernel_val.sum()

        return density
