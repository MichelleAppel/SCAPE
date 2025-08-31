# SCAPE: Shift-variant Cortical-implant Adaptive Phosphene Encoding

[📄 **Paper**](paper/main.pdf)

SCAPE (Shift-variant Cortical-implant Adaptive Phosphene Encoding) is a bio-inspired framework for encoding visual information in cortical prostheses. 
Unlike conventional pipelines that apply uniform filters across the visual field, SCAPE adapts spatial filtering to the **local sampling density** imposed 
by electrode layouts. This ensures that fine detail is preserved where electrodes are dense, while clutter is suppressed where coverage is sparse. 
SCAPE is principled, lightweight, and efficient: it is grounded in sampling theory, implemented with a shift-variant Difference-of-Gaussians (DoG), and 
runs in real time. The framework can be combined with existing phosphene simulators, extended with alternative kernels, and adapted to patient-specific layouts.

---

## Concept and Mathematical Justification

1. **Sampling Density ρ(x,y)**  
   Estimate electrodes per unit area (in degrees or pixels) via cortical magnification or kernel-density estimation (KDE).

2. **Local Nyquist Frequency**  
   ```math
   f_N(x,y) = \sqrt{\frac{\rho(x,y)}{\pi}}
   ```

   is the maximum representable spatial frequency at location (x,y).

3. **Gaussian σ-Map**  
   Choose σ(x,y) so that the Gaussian low-pass cuts off near f_N:
   ```math
   \sigma(x,y) = \frac{1}{2\pi f_N(x,y)}
                = \tfrac{1}{2}\sqrt{\tfrac{1}{\pi \rho(x,y)}}.
   ```

4. **Shift-Variant DoG Filtering**  
   At each pixel, apply a DoG with local σ:
   ```math
   I_{\rm filt}(x,y)
     = G_{\sigma(x,y)} * I(x,y) - G_{k\sigma(x,y)} * I(x,y),
   ```
   implemented efficiently with separable, modulated convolution.

---

## Example Usage

### Generate Sigma Map

```python
from phosphene.density import VisualFieldMapper

mapper = VisualFieldMapper(simulator=simulator)

# Cortical approach
density_cortical = mapper.build_density_map_cortical(total_phosphenes=n_phosphenes)
sigma_cortical   = mapper.build_sigma_map_from_density(density_cortical, space="pixel")

# KDE approach
density_kde = mapper.build_density_map_kde(k=6, alpha=1.0, total_phosphenes=n_phosphenes)
sigma_kde   = mapper.build_sigma_map_from_density(density_kde, space="pixel")
```

### Apply Shift-Variant DoG Filter

```python
from spatial_frequency.components.SeparableModulated2d import SeparableModulatedConv2d
import torch

σ = torch.tensor(sigma_kde).float().cuda().detach()
mod_layer = SeparableModulatedConv2d(in_channels=1, sigma_map=σ).cuda().eval()

filtered = mod_layer(orig_image)
```

---

## Features
- **Density-aware filtering**: adapts filter scale to electrode layout.  
- **Real-time performance**: separable DoG runs efficiently on GPU/embedded hardware.  
- **General framework**: compatible with cortical magnification, KDE, or patient-specific maps.  
- **Extensible**: can be integrated with differentiable phosphene simulators for end-to-end optimization.  

---

## Citation
If you use SCAPE in your research, please cite our paper:

```bibtex
@article{Appel2025scape,
  author    = {Michelle Appel and Antonio Lozano and Eleftherios Papadopoulos and Umut Güçlü and Yağmur Güçlütürk},
  title     = {SCAPE: Shift-variant Cortical-implant Adaptive Phosphene Encoding},
  journal   = {Preprint},
  year      = {2025},
  url       = {paper/main.pdf}
}
```
