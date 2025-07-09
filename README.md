# SCAPE: Shift-variant Cortical-implant Adaptive Phosphene Encoding

SCAPE (Shift-variant Cortical implant Adaptive Phosphene Encoding) is a bio-inspired framework that compensates for the uneven spatial sampling of cortical electrode arrays by adapting filter scale locally. We first estimate a continuous sampling density map ρ(x,y), convert it via Nyquist principles into a spatially varying Gaussian‐filter width σ(x,y), then apply a shift-variant Difference-of-Gaussians (DoG) filter before training a decoder to reconstruct natural scenes from sparse phosphenes.

## Concept and Mathematical Justification

1. **Sampling Density ρ(x,y)**  
   Estimate electrodes per unit area (in degrees or pixels) via cortical magnification or kernel‐density estimation (KDE).

2. **Local Nyquist Frequency**  
   ```math
   f_N(x,y) = \sqrt{\frac{\rho(x,y)}{\pi}}
   ```
   is the maximum representable spatial frequency at (x,y).

3. **Gaussian σ‐Map**  
   We choose σ(x,y) so that the Gaussian low-pass cuts off near f_N. A convenient approximation is  
   ```math
     \sigma(x,y) = \frac{1}{2\pi\,f_N(x,y)}
                 = \frac{1}{2\pi}\sqrt{\frac{\pi}{\rho(x,y)}}
                 = \frac{1}{2}\sqrt{\frac{1}{\pi\,\rho(x,y)}}.
   ```

4. **Shift‐Variant DoG Filtering**  
   At each pixel, apply a DoG with local σ for pre‐processing:
   ```math
     I_{\rm filt}(x,y)
     = G_{\sigma(x,y)} * I(x,y)\;-\;G_{k\,\sigma(x,y)} * I(x,y),
   ```
   implemented efficiently with separable, modulated convolution.

## Generate Sigma Map

```python
from phosphene.density import VisualFieldMapper

mapper = VisualFieldMapper(simulator=simulator)

# Cortical approach:
density_cortical = mapper.build_density_map_cortical(
    total_phosphenes=n_phosphenes
)
sigma_cortical = mapper.build_sigma_map_from_density(
    density_cortical, space="pixel"
)

# KDE approach:
density_kde = mapper.build_density_map_kde(
    k=6, alpha=1.0, total_phosphenes=n_phosphenes
)
sigma_kde = mapper.build_sigma_map_from_density(
    density_kde, space="pixel"
)
```

## Apply Shift-Variant DoG Filter

```python
from spatial_frequency.components.SeparableModulated2d import SeparableModulatedConv2d
import torch

# Wrap σ-map in a modulation layer and filter an image
σ = torch.tensor(sigma_kde).float().cuda().detach()
mod_layer = SeparableModulatedConv2d(
    in_channels=1,
    sigma_map=σ
).cuda().eval()

filtered = mod_layer(orig_image)
```

