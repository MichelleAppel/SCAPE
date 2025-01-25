# Cortical Frequency Mapping & Spatially Adaptive LoG for Prosthetic Vision

This repository demonstrates how to compute and utilize a **maximum resolvable frequency map** rooted in cortical implant constraints—such as phosphene density—in order to perform non‐uniform, spatially adaptive image filtering. The core concept is that each pixel in the visual field can only resolve up to a certain frequency, reflecting the distribution of phosphenes or other cortical implant limitations.

## Maximum Frequency Mapping

We first build a **2D map** of local maximum frequency, \(F_{\mathrm{max}}(x,y)\), which captures how implant‐driven vision might vary across the field. Denser phosphene regions can support higher frequencies (more detail), whereas sparser regions are limited to lower frequencies. 

This map is versatile and can be used for:
- **Adaptive Filtering**: Convolve the image differently depending on local resolution limits.
- **Rendering Optimization**: Coarse rendering in areas that cannot resolve fine detail.
- **Biologically Inspired Models**: Simulating how cortical implants impose non‐uniform “vision.”

In short, the max frequency map is a **general‐purpose foundation** for any scenario that needs to respect a non‐uniform sampling density or cortical layout.

## Spatially Adaptive LoG (LoGFOAConv2d)

To illustrate a practical use, we apply a **space‐variant Laplacian‐of‐Gaussian** filter. By converting \(F_{\mathrm{max}}(x,y)\) into a local Gaussian width \(\sigma(x,y)\), the filter:
- **Uses small \(\sigma\)** where finer details can be resolved (denser phosphene regions).
- **Uses large \(\sigma\)** in sparser implant areas, enforcing a lower‐frequency limit.

Hence, this “cortically guided” LoG captures edges more sharply where the implant can provide finer detail and more diffusely where resolution is inherently limited.

## Optional Gaze Following

We also include a gaze mechanism for re‐centering the higher‐frequency region around the user’s current focus of attention. While secondary to the main frequency‐mapping concept, it shows how dynamic control over the resolvable “sweet spot” might further personalize prosthetic vision feedback.

---

**In summary**, our work emphasizes:
1. **Mapping local maximum frequency** to reflect cortical implant geometry and phosphene density.
2. **Using that map** in a spatially adaptive LoG filter that matches each pixel’s resolvable detail.
3. **Optionally** shifting the high‐resolution zone to the user’s gaze to emulate real‐time attention shifts.

This provides a **powerful abstraction** for simulating prosthetic or implant‐based vision, where the visual field is inherently non‐uniform and must be processed accordingly.

