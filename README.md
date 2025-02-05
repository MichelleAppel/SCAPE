Below is a revised README that incorporates an overview, motivation & context, biological inspiration, mathematical framework, and a description of the architecture. A concise usage snippet is provided at the end.

---

# Phosphene-Aware Adaptive Filtering for Prosthetic Vision

This repository implements an adaptive spatial frequency filtering framework designed specifically for prosthetic vision. Our method adapts visual filtering based on the non-uniform distribution of phosphenes (or cortical density) to better reflect the visual information available to prosthetic devices.

---

## Overview

Traditional filtering applies the same operation uniformly over an image. In contrast, our approach leverages **phosphene density maps** to modulate spatial frequency processing. By adapting the filter kernels based on local density, we can enhance detail in high-resolution regions while smoothing in low-resolution areas—improving the overall perceptual quality of prosthetic vision.

---

## Motivation & Context

Visual prostheses suffer from irregular spatial resolution caused by non-uniform phosphene distribution. Conventional filters, being spatially invariant, may either oversmooth or inadequately capture essential details. Our adaptive filtering framework overcomes these limitations by:
- Dynamically adjusting kernel properties based on local phosphene density.
- Optimizing spatial frequency processing in accordance with the available visual “pixels.”

This targeted adaptation is crucial for improving the clarity and effectiveness of prosthetic vision.

---

## Biological Inspiration

The human visual system naturally prioritizes central vision, where acuity is highest, via the cortical magnification factor. Inspired by this principle, our method allocates higher resolution (i.e., finer spatial filtering) where phosphene density is higher and coarser filtering where it is lower. This mimics the adaptive strategy of the human visual cortex and enhances perceptual performance in prosthetic vision.

---

## Mathematical Framework

Our adaptive filtering relies on relating local phosphene density to the scale of filtering. For example, in the Laplacian-of-Gaussian (LoG) case, the kernel is computed as:


$$\text{LoG}(r,\sigma) = -\frac{1}{\pi\sigma^4}\left(1 - \frac{r^2}{2\sigma^2}\right) \exp\!\left(-\frac{r^2}{2\sigma^2}\right)$$

Here:
- $\sigma$ represents the effective spatial scale (derived from the phosphene density or a cortical map).
- Additional scaling (e.g., multiplication by $\sigma^{2/3}$) and weighting (via a normalized sigma map) are applied to enhance the filter response in desired regions.

This framework ensures that the filter adapts its spatial extent in a biologically informed manner.

---

## Architecture

At the core of our framework is the **UnifiedInputModulation** module, which implements adaptive filtering in two analytic modes:
- **Gaussian Modulation:** Computes a standard Gaussian kernel:
  
  $$G(r,\sigma) = \exp\!\left(-\frac{r^2}{2\sigma^2}\right)$$
  
  and normalizes it for uniform blurring.
  
- **Laplacian-of-Gaussian (LoG):** Computes a LoG kernel as detailed above and applies extra scaling and weighting—using either a provided sigma (or phosphene density) map or focus-of-attention distances—to accentuate edges and transitions.

These analytic modes can be extended or replaced by neural alternatives (such as net_modulated or net_generated) in a modular design. The architecture separates the input modulation from any downstream convolution, allowing each stage to be optimized independently and facilitating integration into larger prosthetic vision pipelines.

---

## Example Usage

Below is a minimal example demonstrating how to use the UnifiedInputModulation module in LoG mode with a precomputed sigma map:

```python
import torch
from modulation_module import UnifiedInputModulation  # adjust import as needed

# Assume sigma_cortical_pix is a precomputed phosphene (cortical) density map in pixel units.
sigma_map_tensor = torch.tensor(sigma_cortical_pix).float().cuda().detach()

# Create the modulation layer using LoG mode.
layer = UnifiedInputModulation(
    kernel_size=119,
    kernel_type="log",  # "log" applies the Laplacian-of-Gaussian
    sigma_map=sigma_map_tensor,
    dilation=1,
    padding_mode='reflect'
).cuda()

# Process an image tensor (img) of shape [B, C, H, W]
filtered_img = layer(img).detach().cpu().clip(0, None)
```

In this example, the module uses the sigma map to generate spatially adaptive LoG kernels. The filtered image output can be further processed (e.g., for generating phosphene stimuli).

---

## Future Directions

- **Neural Adaptive Filtering:** Integrate learnable components for dynamic, task-specific modulation.
- **Real-Time Prosthetic Simulation:** Optimize the pipeline for real-time visual prosthesis applications.
- **User-Centric Evaluations:** Validate adaptive filtering with perceptual studies to further refine the model.

---

This repository lays the foundation for biologically informed image processing in prosthetic vision—improving the translation of visual scenes into effective, adaptive stimuli for prosthetic devices.