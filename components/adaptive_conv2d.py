import torch
import math

class AdaptiveConv2d(torch.nn.Module):
    """
    A convolutional layer in which each pixel can have a *different* kernel,
    modulated by either:
      - distance from a 'focus-of-attention' coordinate, or
      - a user-provided sigma map (e.g., from phosphene density),
      - or a neural network that generates kernels per pixel.

    This generalizes the old 'foveated' concept to *any* per-pixel logic,
    e.g. for cortical or phosphene-based sigma maps.  

    Available 'kernel_type' modes:
      1) "LoG": a Laplacian-of-Gaussian-based per-pixel filter (modulated by 
         distance or a sigma map).
      2) "gaussian_modulated": a simpler Gaussian filter modulated similarly.
      3) "net_modulated": a neural network outputs an 'input modulation' kernel.
      4) "net_generated": a neural network directly outputs the convolution kernel.

    Example usage:
    --------------
        from my_package.conv2d_adaptive import AdaptiveConv2d

        net = AdaptiveConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            kernel_type="LoG",
            # possibly pass log_kernel_size=...,
            # sigma_map=some_map,
            # sigma_min=...,
            # sigma_max=...,
            # etc.
        )

        output = net(input_tensor, foa_xy)
        # shape => [B, 16, H, W] typically, or depends on the kernel_type

    Under the hood, 'input_modulation' is one of the specialized per-pixel
    classes (LoGCoordConv2d, GaussianCoordConv2d, NeuralCoordConv2d, etc.),
    and 'convolution' is either a standard PyTorch Conv2d or a neural approach,
    depending on kernel_type.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 kernel_size=5,
                 dilation=1,
                 padding_mode='reflect',
                 kernel_type="LoG",
                 **kwargs):
        """
        Arguments:
            in_channels (int): Input channels for the final convolution
            out_channels (int): Output channels for the final convolution
            kernel_size (int): The base kernel size used in the classic conv or net.
            dilation (int): Dilation factor for the final convolution
            padding_mode (str): e.g. 'zeros' or 'reflect'
            kernel_type (str): "LoG", "gaussian_modulated", "net_modulated", "net_generated"
            **kwargs: Additional parameters passed to the 'input_modulation'
                      classes. E.g. 'log_kernel_size', 'sigma_map', etc.

        The constructor wires up two sub-modules:
            self.input_modulation : a 'coord-based' submodule (LoGCoordConv2d, etc.)
            self.convolution      : either a standard PyTorch Conv2d or a custom net.
        """
        super(AdaptiveConv2d, self).__init__()

        valid_types = ["LoG", "gaussian_modulated", "net_modulated", "net_generated"]
        assert kernel_type in valid_types, \
            f"Invalid kernel type: {kernel_type} (must be one of: {valid_types})"

        self.input_modulation = None
        self.convolution = None

        # We define a local class that wraps PyTorch Conv2d to accept (x, foa_xy)
        class Conv2dWrapper(torch.nn.Conv2d):
            def __init__(self, *args, **kargs):
                super().__init__(*args, **kargs)
                self._init_debug_weights(1)

            def forward(self, x, foa_xy):
                return super().forward(x)

            def _init_debug_weights(self, w_type=0):
                """Optionally for debugging: set conv weights to identity or uniform."""
                with torch.no_grad():
                    if w_type == 0:
                        # identity-like: only center is 1, rest is 0
                        self.weight *= 0.
                        mid = int(self.kernel_size[0]) // 2
                        self.weight[:, :, mid, mid] = 1.
                        self.bias.zero_()
                    elif w_type == 1:
                        # uniform kernel
                        self.weight.fill_(1.)
                        # normalizing
                        denom = float(self.weight.numel())
                        self.weight /= denom
                        self.bias.zero_()

        # A trivial identity to unify interface (accepts x, foa_xy and returns x).
        class IdentityModule:
            def __call__(self, a, b):
                return a

            def __str__(self):
                return self.__class__.__name__

        # Depending on kernel_type, we set up the 'input_modulation' and the final 'convolution'
        if kernel_type == "gaussian_modulated":
            # A Gaussian approach for per-pixel kernels
            raise NotImplementedError("Gaussian modulated not implemented yet"
            )

        elif kernel_type == "LoG":
            # A Laplacian-of-Gaussian approach for per-pixel kernels
            self.input_modulation = LoGCoordConv2d(**kwargs)
            self.convolution = Conv2dWrapper(
                in_channels, out_channels, kernel_size,
                padding=int((kernel_size-1)*dilation + 1)//2,
                stride=1, dilation=dilation, groups=1,
                bias=True, padding_mode=padding_mode
            )

        elif kernel_type == "net_modulated":
            # A neural net modifies the *input*, then a standard conv
            # We expect 'kernel_net' in kwargs
            raise NotImplementedError("net_generated not implemented yet")

        elif kernel_type == "net_generated":
            # A neural net directly generates the convolution kernel.
            # So the 'input_modulation' is effectively Identity, and the 'convolution'
            # is the neural approach.
            raise NotImplementedError("net_generated not implemented yet")

    def forward(self, input_data, foa_xy, compute_region_indices=False):
        """
        Forward pass:
          1) input_modulation => modifies or filters the input_data in a per-pixel manner
          2) self.convolution => typically a standard Conv2d or a neural approach

        Args:
            input_data (torch.Tensor): shape [B, in_channels, H, W]
            foa_xy (torch.Tensor): shape [B, 2] or [1, 2], or not used if sigma_map mode
            compute_region_indices (bool): optional, rarely used

        Returns:
            output_data (torch.Tensor): shape [B, out_channels, H, W] by default
        """
        x = self.input_modulation(input_data, foa_xy)
        x = self.convolution(x, foa_xy)

        if compute_region_indices:
            return x, None
        else:
            return x

    def __str__(self):
        s = f"[{self.__class__.__name__}]"
        s += "\n- input_modulation:\n{\n\t"
        s += str(self.input_modulation).replace('\n', '\n\t') + "\n}"
        s += "\n- convolution:\n{\n\t"
        s += str(self.convolution).replace('\n', '\n\t')
        s += "\n}"
        return s


# --------------------------------------------------------------------------
# Now define your "coord-based" submodules:
# --------------------------------------------------------------------------

class BaseCoordConv2d:
    """
    Base class for per-pixel convolution in which the kernel depends on
    coordinate-based logic (e.g., distance from FOA or a sigma map).
    Subclasses must implement `generate_per_pixel_kernels(foa_xy)`.
    """

    def __init__(self, dilation=1, padding_mode='reflect'):
        assert dilation >= 1
        self.dilation = int(dilation)
        self.padding_mode = padding_mode
        self.h = -1
        self.w = -1
        self.xx_yy = None  # a [hw,2] coordinate map

    def __call__(self, input_data, foa_xy):
        """
        Perform the 'per-pixel kernel' convolution. This code:
          - builds the per-pixel kernels
          - unfolds the input
          - multiplies each patch by the correct kernel
          - sums to produce the output

        You might prefer a separate method, but for clarity we keep it in __call__.
        """
        B, C, H, W = input_data.shape
        self._build_shared_cache(input_data)
        self._build_custom_cache(input_data)

        # Build kernels => shape [B, out_ch, in_ch, kernel_area, HW]
        per_pixel_kernels = self.generate_per_pixel_kernels(foa_xy)

        out_channels = per_pixel_kernels.shape[1]
        in_channels = per_pixel_kernels.shape[2]
        kernel_size = int(math.sqrt(per_pixel_kernels.shape[3]))
        spatial_only_kernel = (in_channels == 1 and out_channels == 1)

        # check dimension consistency
        assert in_channels == 1 or in_channels == C, (
            f"Invalid kernel in_channels={in_channels} for input {C} channels."
        )

        # Unfold input => shape [B, (C*kernel_area), HW]
        if self.padding_mode == 'zeros':
            patches = torch.nn.functional.unfold(
                input_data,
                kernel_size=kernel_size,
                dilation=self.dilation,
                padding=((kernel_size-1)*self.dilation+1)//2,
                stride=1
            )
        else:
            pad_val = ((kernel_size-1)*self.dilation+1)//2
            padded_input_data = torch.nn.functional.pad(
                input_data,
                (pad_val, pad_val, pad_val, pad_val),
                mode=self.padding_mode
            )
            patches = torch.nn.functional.unfold(
                padded_input_data,
                kernel_size=kernel_size,
                dilation=self.dilation,
                stride=1
            )
        # shape => [B, C*kernel_area, HW]
        patches = patches.view(B, 1, C, kernel_size**2, -1)  # [B,1,C,kk,HW]

        # Multiply by kernel
        if spatial_only_kernel:
            dims_to_sum = 3  # sum over the kernel_area dimension only
            conv_out_channels = C
        else:
            dims_to_sum = [2, 3]  # sum over in_channels & kernel_area
            conv_out_channels = out_channels

        # [B,1,C,kk,HW] * [B,out_ch,in_ch,kk,HW] => sum(dims) => [B, out_ch, HW]
        conv_data = torch.sum(patches * per_pixel_kernels, dim=dims_to_sum)
        # => shape [B, conv_out_channels, HW]

        # reshape to [B, conv_out_channels, H, W]
        return conv_data.view(B, conv_out_channels, H, W)

    def _build_shared_cache(self, input_data):
        # if shape changed or first time:
        if self.h != input_data.shape[2] or self.w != input_data.shape[3] or self.xx_yy is None:
            self.h = input_data.shape[2]
            self.w = input_data.shape[3]
            device = input_data.device
            # Build a coordinate map: we treat y as row, x as col
            ys = torch.arange(self.h, device=device)
            xs = torch.arange(self.w, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            # shape => [H,W], we flatten => [H*W, 2]
            self.xx_yy = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()

    def _build_custom_cache(self, input_data):
        """Any subclass-specific caching logic goes here."""
        pass

    def generate_per_pixel_kernels(self, foa_xy):
        raise NotImplementedError("Subclasses must implement this method.")


# --------------------------------------------------------------------------
# LoGCoordConv2d: A space-variant Laplacian-of-Gaussian filter
# --------------------------------------------------------------------------
class LoGCoordConv2d(BaseCoordConv2d):
    """
    A per-pixel Laplacian-of-Gaussian, modulated by:
      - a distance-based sigma function (linear/exponential), or
      - a user-provided 'sigma_map'.

    If sigma_map is used, it typically comes from a phosphene density or
    cortical magnification map.

    merge_mode is *not* handled here. That can happen up in the 
    'AdaptiveConv2d' layer if multi-channel input is processed.

    Example:
        log_mod = LoGCoordConv2d(
            log_kernel_size=9,
            sigma_min=0.5,
            sigma_max=10.0,
            sigma_function="map",
            sigma_map=torch.tensor(...),
        )
    """

    def __init__(
        self,
        log_kernel_size=99,
        sigma_min=0.01,
        sigma_max=1.0,
        sigma_function="linear",
        bias=False,
        sigma_map=None
    ):
        super().__init__(dilation=1, padding_mode='reflect')
        assert log_kernel_size > 0, "log_kernel_size must be positive & odd"
        assert sigma_max > sigma_min > 0, "sigma_min < sigma_max required."
        assert sigma_function in ["linear", "exponential", "map"], \
            "sigma_function must be one of: linear, exponential, map."

        self.log_kernel_size = log_kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_function_type = sigma_function

        # Possibly store the map as a buffer
        if sigma_map is not None:
            self.sigma_map = torch.tensor(sigma_map, dtype=torch.float32)

        else:
            self.sigma_map = None

        # We'll store r^2 for the kernel, etc.
        self._kernel_sq_dists = None

    def _build_custom_cache(self, input_data):
        # Build the kernel squared distances if needed
        device = input_data.device
        if self._kernel_sq_dists is not None and self._kernel_sq_dists.device == device:
            return

        radius = self.log_kernel_size // 2
        xs = torch.arange(-radius, radius+1, device=device)
        yy, xx = torch.meshgrid(xs, xs, indexing='ij')
        self._kernel_sq_dists = (xx**2 + yy**2).view(-1).float()

    def generate_per_pixel_kernels(self, foa_xy):
        """
        Build Laplacian-of-Gaussian kernels for each pixel. shape => [B, out_ch=1, in_ch=1, kernel_area, HW].
        """
        B = foa_xy.shape[0] # batch size
        kk = self.log_kernel_size**2 # kernel_area
        HW = self.h*self.w # spatial size

        # 1) compute sigma per pixel => shape [B,HW]
        sigmas = self._get_sigma(foa_xy)  # shape [B,HW]

        # 2) vectorized LoG
        r_sq = self._kernel_sq_dists.view(1, kk, 1)  # [1,kk,1]
        sigma = sigmas.view(B,1,HW)                  # [B,1,HW]

        tmp = r_sq / (2.*sigma*sigma)                # [B,kk,HW]
        factor = -1./(math.pi*(sigma**4))            # [B,1,HW]
        bracket = 1.-(r_sq/(2.*sigma*sigma))         # [B,kk,HW]
        per_pixel_kernels = factor*bracket*torch.exp(-tmp)  # [B,kk,HW]

        # optional: multiply or scale to your liking
        scale_factor = sigma**0.5
        per_pixel_kernels = per_pixel_kernels*scale_factor

        # if you do additional distance-based weighting:
        diff_xx_yy = self.xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)  # [B,HW,2]
        dists = torch.sqrt(torch.sum(diff_xx_yy**2, dim=2))         # [B,HW]
        # multiply
        per_pixel_kernels *= dists.view(B,1,HW)

        # reshape => [B, out_ch=1, in_ch=1, kk, HW]
        return per_pixel_kernels.view(B, 1, 1, kk, HW)

    def _get_sigma(self, foa_xy):
        """
        Return per-pixel sigma => shape [B,HW], 
        depending on self.sigma_function_type.
        """
        if self.sigma_function_type == "map":
            # flatten sigma_map => shape [HW], expand for B
            flat_map = self.sigma_map.view(-1)  # [HW]
            B = foa_xy.shape[0]
            return flat_map.unsqueeze(0).expand(B, -1)  # [B,HW]

        # else we do distance-based
        diff = self.xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)  # [B,HW,2]
        dists = torch.sqrt(torch.sum(diff**2, dim=2))         # [B,HW]
        # normalize by diagonal
        diag = math.sqrt(self.h**2 + self.w**2)
        dists_norm = dists/diag

        if self.sigma_function_type == "linear":
            # sigma = (1-dist_norm)*sigma_min + dist_norm*sigma_max
            return (1.-dists_norm)*self.sigma_min + dists_norm*self.sigma_max

        elif self.sigma_function_type == "exponential":
            alpha = 9.0
            weights = torch.exp(-alpha*(dists**2)/float(diag**2))
            return weights*self.sigma_min + (1.-weights)*self.sigma_max

        else:
            raise ValueError(f"Unknown sigma_function_type {self.sigma_function_type}")

    def __str__(self):
        s = f"[{self.__class__.__name__}]"
        s += f"\n- log_kernel_size: {self.log_kernel_size}"
        s += f"\n- sigma_min: {self.sigma_min}"
        s += f"\n- sigma_max: {self.sigma_max}"
        s += f"\n- sigma_function_type: {self.sigma_function_type}"
        return s

