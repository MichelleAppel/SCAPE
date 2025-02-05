import torch
import math


class Conv2dWrapper(torch.nn.Conv2d):
    """
    A simple wrapper around torch.nn.Conv2d that accepts an extra FOA argument 
    (or sigma_map) in its forward method for a uniform interface.
    """
    def __init__(self, *args, **kwargs):
        super(Conv2dWrapper, self).__init__(*args, **kwargs)

    def forward(self, x, foa_xy=None, sigma_map=None):
        # The extra argument is ignored here.
        return super(Conv2dWrapper, self).forward(x)


class IdentityModule:
    """
    A trivial identity module that returns its input unchanged.
    Useful for the net_generated mode where input modulation is not applied.
    """
    def forward(self, x, foa_xy=None, sigma_map=None):
        return x

    def __str__(self):
        return self.__class__.__name__


# ================================================================
# ANALYTIC (Gaussian/LoG) Input Modulation
# ================================================================

class UnifiedInputModulation(torch.nn.Module):
    """
    Input modulation using an analytic kernel computed from either a Gaussian 
    or a Laplacian-of-Gaussian (LoG) function.
    
    Sigma can be computed from a focus-of-attention (FOA) coordinate using a 
    specified function ('linear' or 'exponential') or provided directly via a 
    sigma_map (e.g., derived from phosphene density). In the latter case, FOA is ignored.
    
    This module extracts patches from the input image and applies a per-pixel 
    modulation kernel (computed from the chosen analytic function).
    """
    def __init__(self,
                 kernel_size,
                 kernel_type="gaussian_modulated",  # or "LoG"
                 dilation=1,
                 padding_mode='reflect',
                 sigma_min=None,
                 sigma_max=None,
                 sigma_function="linear",
                 sigma_map=None):
        super(UnifiedInputModulation, self).__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.kernel_size = int(kernel_size)
        self.kernel_type = kernel_type.lower()
        self.dilation = int(dilation)
        self.padding_mode = padding_mode
        self.sigma_map = sigma_map

        if self.sigma_map is None:
            if sigma_min is None or sigma_max is None:
                raise ValueError("sigma_min and sigma_max must be provided when sigma_map is not used.")
            self.sigma_min = float(sigma_min)
            self.sigma_max = float(sigma_max)
            if sigma_max <= sigma_min:
                raise ValueError("sigma_max must be greater than sigma_min.")
            if sigma_function not in ["linear", "exponential"]:
                raise ValueError("sigma_function must be 'linear' or 'exponential'.")
            self.sigma_function = sigma_function
        else:
            self.sigma_min = None
            self.sigma_max = None
            self.sigma_function = None

        # Caches for the coordinate grid and kernel squared distances.
        self._cached_input_shape = None  # (H, W)
        self._xx_yy = None               # Tensor of shape [H*W, 2]
        self._kernel_sq_dists = None     # Tensor of shape [kernel_size^2]

    def _build_shared_cache(self, input_data):
        """Build and cache a coordinate grid for the input image."""
        B, C, H, W = input_data.shape
        if self._cached_input_shape != (H, W) or self._xx_yy is None:
            self._cached_input_shape = (H, W)
            device = input_data.device
            ys = torch.arange(H, device=device)
            xs = torch.arange(W, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self._xx_yy = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()

    def _build_kernel_cache(self, input_data):
        """Precompute and cache squared distances for kernel positions."""
        device = input_data.device
        if self._kernel_sq_dists is None or self._kernel_sq_dists.device != device:
            radius = self.kernel_size // 2
            coords = torch.arange(-radius, radius + 1, device=device).float()
            yy, xx = torch.meshgrid(coords, coords, indexing='ij')
            self._kernel_sq_dists = (xx**2 + yy**2).view(-1)

    def _compute_sigma(self, foa_xy, B):
        """
        Compute per-pixel sigma values.
        
        If sigma_map is provided, it is used (after flattening and expanding along the batch);
        otherwise sigma is computed from the Euclidean distance from each pixel to the FOA.
        """
        if self.sigma_map is not None:
            sigma_map_flat = self.sigma_map.view(-1)  # [H*W]
            sigma = sigma_map_flat.unsqueeze(0).expand(B, -1)
            return sigma
        else:
            if foa_xy is None:
                raise ValueError("FOA coordinates must be provided when sigma_map is not used.")
            diff = self._xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)  # [B, H*W, 2]
            dists = torch.sqrt(torch.sum(diff ** 2, dim=2))          # [B, H*W]
            H, W = self._cached_input_shape
            diag = math.sqrt(H**2 + W**2)
            dists_norm = dists / diag
            if self.sigma_function == "linear":
                sigma = (1. - dists_norm) * self.sigma_min + dists_norm * self.sigma_max
            else:
                alpha = 9.0
                weights = torch.exp(-alpha * (dists**2) / (diag**2))
                sigma = weights * self.sigma_min + (1. - weights) * self.sigma_max
            return sigma

    def _generate_kernel(self, sigma, foa_xy=None):
        """
        Generate per-pixel modulation kernels for analytic modes.
        
        Depending on self.kernel_type:
        
        - For "gaussian_modulated" (or "gaussian"):
            Computes a standard Gaussian kernel:
                G(r, σ) = exp( - r² / (2σ²) )
            and normalizes it so that the sum over the kernel area is 1.
        
        - For "log" (or "laplacian-of-gaussian"):
            Computes the Laplacian-of-Gaussian kernel as:
                tmp     = r² / (2σ²)
                factor  = -1 / (π σ⁴)
                bracket = 1 - (r² / (2σ²))
                LoG     = factor × bracket × exp( - tmp )
            Then applies an additional scaling factor σ^(2/3) and extra weighting:
                - If self.sigma_map is provided, it is flattened and normalized to [0, 1] and used.
                - Otherwise, if foa_xy is provided, the Euclidean distance from each pixel to the FOA is used.
        
        Parameters
        ----------
        sigma : torch.Tensor
            Tensor of shape [B, HW] containing the sigma values (in pixel units) per pixel.
        foa_xy : torch.Tensor, optional
            Tensor of shape [B, 2] containing the FOA coordinates (used if no sigma_map is provided).
        
        Returns
        -------
        kernel : torch.Tensor
            Per-pixel modulation kernels of shape [B, 1, 1, kernel_area, HW].
        """
        B, HW = sigma.shape
        # Reshape sigma for broadcasting: [B, 1, HW]
        sigma_reshaped = sigma.view(B, 1, HW)
        # Precomputed squared distances for the kernel, shape: [1, kernel_area, 1]
        r_sq = self._kernel_sq_dists.view(1, -1, 1)
        
        if self.kernel_type in ["gaussian_modulated", "gaussian"]:
            # Compute the Gaussian kernel:
            #    G(r,σ) = exp( - r² / (2σ²) )
            kernel = torch.exp(-r_sq / (2. * sigma_reshaped**2))
            # Normalize over the kernel area so that the sum is 1.
            kernel_sum = torch.sum(kernel, dim=1, keepdim=True)
            kernel = kernel / (kernel_sum + 1e-8)
            
        elif self.kernel_type in ["log", "laplacian-of-gaussian"]:
            # Compute basic LoG components:
            tmp = r_sq / (2. * sigma_reshaped.clip(1, None)**2)          # [B, kernel_area, HW]
            factor = -1.0 / (math.pi * sigma_reshaped.clip(1, None)**4)     # [B, 1, HW]
            bracket = 1.0 - (r_sq / (2. * sigma_reshaped.clip(1, None)**2)) # [B, kernel_area, HW]
            kernel = factor * bracket * torch.exp(-tmp)       # [B, kernel_area, HW]
            
            # Apply additional scaling: multiply by sigma^(2/3)
            scale_factor = sigma_reshaped.clip(1, None)**(2/3)              # [B, 1, HW]
            kernel = kernel * scale_factor                    # [B, kernel_area, HW]
            
            # Apply extra weighting:
            if self.sigma_map is not None:
                # Use sigma_map instead of FOA distance.
                # Assume sigma_map has shape [H, W]. Flatten it to [1, HW] and normalize.
                sigma_map_flat = self.sigma_map.view(1, -1)
                sigma_map_norm = sigma_map_flat / (sigma_map_flat.max() + 1e-8)
                # Expand to [B, 1, HW] (and then to [B, kernel_area, HW])
                weight = sigma_map_norm.unsqueeze(1).expand(B, r_sq.shape[1], HW)
                kernel = kernel * weight
            elif foa_xy is not None:
                # Use FOA-based distance weighting.
                diff = self.xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)  # [B, HW, 2]
                dists = torch.sqrt(torch.sum(diff**2, dim=2))           # [B, HW]
                kernel = kernel * dists.view(B, 1, HW)
            # Note: Do not re-normalize here so that the extra scaling is preserved.
        else:
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")
        
        # Reshape kernel to [B, 1, 1, kernel_area, HW]
        return kernel.view(B, 1, 1, -1, HW)


    def forward(self, input_data, foa_xy=None, sigma_map=None):
        """
        Apply analytic input modulation.
        
        Uses either FOA coordinates (if sigma_map is not provided) or a sigma_map to compute per-pixel kernels,
        then applies the kernels via an unfold–multiply–sum procedure.
        """
        B, C, H, W = input_data.shape
        self._build_shared_cache(input_data)
        self._build_kernel_cache(input_data)
        # If a sigma_map is passed here, override the module's stored sigma_map.
        if sigma_map is not None:
            sigma_val = sigma_map.view(-1).unsqueeze(0).expand(B, -1)
        else:
            sigma_val = self._compute_sigma(foa_xy, B)
        kernel = self._generate_kernel(sigma_val)
        pad = ((self.kernel_size - 1) * self.dilation + 1) // 2
        if self.padding_mode == 'zeros':
            patches = torch.nn.functional.unfold(input_data,
                                                   kernel_size=self.kernel_size,
                                                   dilation=self.dilation,
                                                   padding=pad,
                                                   stride=1)
        else:
            padded = torch.nn.functional.pad(input_data, (pad, pad, pad, pad), mode=self.padding_mode)
            patches = torch.nn.functional.unfold(padded,
                                                   kernel_size=self.kernel_size,
                                                   dilation=self.dilation,
                                                   padding=0,
                                                   stride=1)
        patches = patches.view(B, C, self.kernel_size**2, H * W)
        kernel = kernel.expand(B, 1, C, self.kernel_size**2, H * W)
        modulated = torch.sum(patches.unsqueeze(1) * kernel, dim=3)
        return modulated.view(B, C, H, W)


# ================================================================
# NEURAL MODULATION
# ================================================================

class NeuralModulatedInput(torch.nn.Module):
    """
    Input modulation using a neural network.
    
    In the net_modulated mode, a neural network (kernel_net) receives as input either:
      - the difference between pixel coordinates and FOA (if sigma_map is not provided), or
      - the sigma_map values (if provided),
    and outputs a modulation kernel for each pixel.
    
    The modulation kernel is then applied using an unfold–multiply–sum operation.
    """
    def __init__(self, kernel_net, kernel_size, dilation=1, padding_mode='reflect'):
        super(NeuralModulatedInput, self).__init__()
        self.kernel_net = kernel_net
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding_mode = padding_mode
        self._cached_input_shape = None
        self._xx_yy = None

    def _build_shared_cache(self, input_data):
        B, C, H, W = input_data.shape
        if self._cached_input_shape != (H, W) or self._xx_yy is None:
            self._cached_input_shape = (H, W)
            device = input_data.device
            ys = torch.arange(H, device=device)
            xs = torch.arange(W, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self._xx_yy = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()

    def forward(self, input_data, foa_xy=None, sigma_map=None):
        """
        Compute modulation kernels via the neural network.
        
        If sigma_map is provided, it is flattened and used as input to the network; otherwise,
        the network input is the coordinate differences between each pixel and FOA.
        Then, the generated per-pixel kernels are applied to the input.
        """
        B, C, H, W = input_data.shape
        self._build_shared_cache(input_data)
        if sigma_map is not None:
            # Use sigma_map values as features.
            if sigma_map.dim() == 2:
                sigma_map = sigma_map.unsqueeze(0).expand(B, -1, -1)
            net_input = sigma_map.view(B * H * W, 1).to(torch.float)
        else:
            if foa_xy is None:
                raise ValueError("Either FOA coordinates or sigma_map must be provided for net_modulated mode.")
            diff = self._xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)
            net_input = diff.view(B * H * W, 2).to(torch.float)
        # Pass through kernel_net; expected output shape: [B*H*W, kernel_size^2]
        net_output = self.kernel_net(net_input)
        # Reshape to [B, H*W, kernel_size^2]
        kernel = net_output.view(B, H * W, self.kernel_size**2)
        # Permute to [B, 1, 1, kernel_size^2, H*W]
        kernel = kernel.permute(0, 2, 1).unsqueeze(1)
        pad = ((self.kernel_size - 1) * self.dilation + 1) // 2
        if self.padding_mode == 'zeros':
            patches = torch.nn.functional.unfold(input_data,
                                                   kernel_size=self.kernel_size,
                                                   dilation=self.dilation,
                                                   padding=pad,
                                                   stride=1)
        else:
            padded = torch.nn.functional.pad(input_data, (pad, pad, pad, pad), mode=self.padding_mode)
            patches = torch.nn.functional.unfold(padded,
                                                   kernel_size=self.kernel_size,
                                                   dilation=self.dilation,
                                                   padding=0,
                                                   stride=1)
        patches = patches.view(B, C, self.kernel_size**2, H * W)
        kernel = kernel.expand(B, 1, C, self.kernel_size**2, H * W)
        modulated = torch.sum(patches.unsqueeze(1) * kernel, dim=3)
        return modulated.view(B, C, H, W)


class NeuralGeneratedConv(torch.nn.Module):
    """
    Convolution where a neural network directly generates the convolution kernels.
    
    In the net_generated mode, the input modulation stage is an identity; the neural network (kernel_net)
    directly produces, for each pixel, a full convolution kernel.
    
    Optionally, if a sigma_map is provided, it is used as the network input instead of FOA differences.
    """
    def __init__(self, in_channels, out_channels, kernel_net, dilation=1, padding_mode='reflect'):
        super(NeuralGeneratedConv, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_net = kernel_net
        self.dilation = int(dilation)
        self.padding_mode = padding_mode
        self._cached_input_shape = None
        self._xx_yy = None

    def _build_shared_cache(self, input_data):
        B, C, H, W = input_data.shape
        if self._cached_input_shape != (H, W) or self._xx_yy is None:
            self._cached_input_shape = (H, W)
            device = input_data.device
            ys = torch.arange(H, device=device)
            xs = torch.arange(W, device=device)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            self._xx_yy = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()

    def forward(self, input_data, foa_xy=None, sigma_map=None):
        """
        Directly generate convolution kernels via the neural network and apply them.
        
        The network input is either the coordinate differences (if sigma_map is None) or the sigma_map values.
        The network output is reshaped into per-pixel kernels of shape 
        [B, out_channels, in_channels, kernel_size^2, H*W] which are then applied to the input.
        """
        B, C, H, W = input_data.shape
        self._build_shared_cache(input_data)
        if sigma_map is not None:
            if sigma_map.dim() == 2:
                sigma_map = sigma_map.unsqueeze(0).expand(B, -1, -1)
            net_input = sigma_map.view(B * H * W, 1).to(torch.float)
        else:
            if foa_xy is None:
                raise ValueError("Either FOA coordinates or sigma_map must be provided for net_generated mode.")
            diff = self._xx_yy.unsqueeze(0) - foa_xy.unsqueeze(1)
            net_input = diff.view(B * H * W, 2).to(torch.float)
        net_output = self.kernel_net(net_input)
        tot_elems = net_output.shape[1]
        k_area = tot_elems // (self.out_channels * self.in_channels)
        kernel_size = int(math.sqrt(k_area))
        if kernel_size % 2 == 0:
            raise ValueError("Derived kernel size must be odd.")
        kernel = net_output.view(B, H * W, self.out_channels, self.in_channels, k_area)
        kernel = kernel.permute(0, 2, 3, 4, 1)
        pad = ((kernel_size - 1) * self.dilation + 1) // 2
        if self.padding_mode == 'zeros':
            patches = torch.nn.functional.unfold(input_data,
                                                   kernel_size=kernel_size,
                                                   dilation=self.dilation,
                                                   padding=pad,
                                                   stride=1)
        else:
            padded = torch.nn.functional.pad(input_data, (pad, pad, pad, pad), mode=self.padding_mode)
            patches = torch.nn.functional.unfold(padded,
                                                   kernel_size=kernel_size,
                                                   dilation=self.dilation,
                                                   padding=0,
                                                   stride=1)
        patches = patches.view(B, self.in_channels, kernel_size**2, H * W)
        conv_out = torch.sum(patches.unsqueeze(1) * kernel, dim=[2,3])
        return conv_out.view(B, self.out_channels, H, W)


# ================================================================
# TOP-LEVEL MODULATED CONVOLUTIONAL LAYER
# ================================================================

class ModulatedConv2d(torch.nn.Module):
    """
    A foveated convolutional layer supporting four modes:
    
      1. "gaussian_modulated": Analytic modulation using a Gaussian before a standard convolution.
      2. "LoG": Analytic modulation using a Laplacian-of-Gaussian before a standard convolution.
      3. "net_modulated": A neural network produces a modulation kernel that is applied before a standard convolution.
      4. "net_generated": A neural network directly generates per-pixel convolution kernels.
    
    In analytic modes, you can supply either FOA coordinates or a sigma_map (e.g. a phosphene density map)
    to determine local kernel parameters.
    
    For the neural modes, if a sigma_map is provided, it is passed to the neural network instead of FOA differences.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 padding_mode='reflect',
                 kernel_type="gaussian_modulated",
                 sigma_min=None,
                 sigma_max=None,
                 sigma_function="linear",
                 sigma_map=None,
                 kernel_net=None):
        super(ModulatedConv2d, self).__init__()
        kernel_type = kernel_type.lower()
        valid_types = ["gaussian_modulated", "gaussian", "laplacian", "log", "net_modulated", "net_generated"]
        if kernel_type not in valid_types:
            raise ValueError(f"Invalid kernel_type: {kernel_type}. Must be one of {valid_types}.")
        self.kernel_type = kernel_type

        if kernel_type in ["gaussian_modulated", "log", "gaussian", "laplacian"]:
            self.input_modulation = UnifiedInputModulation(
                kernel_size=kernel_size,
                kernel_type=kernel_type,
                dilation=dilation,
                padding_mode=padding_mode,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_function=sigma_function,
                sigma_map=sigma_map
            )
            pad = ((kernel_size - 1) * dilation + 1) // 2
            self.convolution = Conv2dWrapper(in_channels,
                                             out_channels,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             padding=pad,
                                             dilation=dilation,
                                             bias=True,
                                             padding_mode=padding_mode)
        elif kernel_type == "net_modulated":
            if kernel_net is None:
                raise ValueError("kernel_net must be provided for net_modulated mode.")
            self.input_modulation = NeuralModulatedInput(
                kernel_net=kernel_net,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_mode=padding_mode
            )
            pad = ((kernel_size - 1) * dilation + 1) // 2
            self.convolution = Conv2dWrapper(in_channels,
                                             out_channels,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             padding=pad,
                                             dilation=dilation,
                                             bias=True,
                                             padding_mode=padding_mode)
        elif kernel_type == "net_generated":
            if kernel_net is None:
                raise ValueError("kernel_net must be provided for net_generated mode.")
            self.input_modulation = IdentityModule()
            self.convolution = NeuralGeneratedConv(in_channels,
                                                     out_channels,
                                                     kernel_net=kernel_net,
                                                     dilation=dilation,
                                                     padding_mode=padding_mode)

    def forward(self, input_data, foa_xy=None, sigma_map=None):
        """
        Perform the modulated convolution.
        
        For analytic modes, either FOA coordinates or a sigma_map is used to compute local kernels.
        For neural modes, if sigma_map is provided it is passed to the neural network instead of FOA.
        """
        x = self.input_modulation.forward(input_data, foa_xy, sigma_map)
        x = self.convolution.forward(x, foa_xy, sigma_map)
        return x

    def __str__(self):
        s = f"[{self.__class__.__name__} - kernel_type: {self.kernel_type}]\n"
        s += f"Input Modulation:\n\t{str(self.input_modulation)}\n"
        s += f"Convolution:\n\t{str(self.convolution)}"
        return s
