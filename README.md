# README: Max Frequency Mapping & Foveated LoG Demonstration

This repository focuses on **building a spatial map of the maximum resolvable frequency** at each pixel and using that information to process images in a way that reflects non‐uniform visual acuity.

## Max Frequency Mapping

At the heart of this approach is a **2D map** that defines the local maximum spatial frequency $F_{\mathrm{max}}(x,y)$ for every pixel, often tied to cortical magnification or prosthetic “phosphene density.” The idea is that the center of vision—or the “highest resolution” region—can handle much finer spatial detail than the periphery, which only supports lower frequencies.

This frequency map can be used in **many** ways:
- **Foveated Filters**: Different parts of the image can be convolved with filters at different scales.
- **Adaptive Rendering**: Graphics can be rendered more coarsely in the periphery to save computation.
- **Phosphene Simulations**: Where fewer phosphenes are available in the periphery, we define lower max frequency.

Essentially, **the max frequency map is a flexible core tool** for any scenario that must account for non‐uniform resolution or sampling in the visual field.

## Foveated LoG (LoGFOAConv2d)

In this repository, we provide one concrete demonstration: a **space‐variant Laplacian‐of‐Gaussian** filter that respects the max frequency map by converting $F_{\mathrm{max}}(x,y)$ into a local Gaussian width $\sigma(x,y)$. This yields a **foveated LoG** that:
- Uses **small $\sigma\$** (high‐frequency) near the most resolvable region.
- Uses **large $\sigma\$** (low‐frequency) in peripheral or low‐acuity regions.

That way, edges in the center are captured at high detail, whereas the periphery is blurrier—analogous to cortical magnification in V1.

## Gaze Following (Optional)

We also include a **simple gaze‐based** mechanism where the highest max frequency region can dynamically shift according to an estimated gaze position. This is secondary to our main emphasis on the **max frequency map** itself, but it demonstrates how the foveation could be re‐centered in real time based on where a user is looking.

---

**In summary**, this repository highlights:
1. **A robust max frequency map** approach, general enough for multiple foveation or retinotopic tasks.
2. **A LoG demonstration (LoGFOAConv2d)** that uses that map to produce a biologically plausible, space‐variant edge filter.
3. An **optional** gaze‐based extension that moves the fovea to the point of user attention.

This code can serve as a foundation for **any** project that requires non‐uniform resolution or frequency constraints in visual processing—especially for prosthetic or foveated vision research.
