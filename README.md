# **Phosphene-Aware Spatial Frequency Processing**  
A computational framework for **adaptive spatial frequency modulation** in prosthetic vision, driven by **phosphene density maps** to tailor spatial resolution. 

---

## **Overview**  
This repository introduces an approach to **adaptive visual filtering** that integrates the non-uniform distribution of phosphenes inherent in visual prostheses. Unlike traditional methods that apply static filters across the visual field, our approach uses **phosphene density maps** as a foundation for **spatially adaptive frequency processing**. This technique is designed to optimize how visual information is conveyed through prosthetic devices by modulating spatial frequency resolution in accordance with the distribution of available visual "pixels" or phosphenes.

The core of this framework is the **AdaptiveConv2d** module, which dynamically adjusts its filtering behavior based on phosphene density, providing a flexible tool for biologically informed visual processing.

---

## **Context & Motivation**  
### **Challenges in Prosthetic Vision**  
Visual prostheses are characterized by an irregular spatial resolution due to the variability in phosphene distribution across the visual field. This non-uniformity leads to:  
- **Detail loss in high-density regions**, where filtering should enhance finer details.
- **Oversmoothing in low-density areas**, where emphasizing broader patterns is crucial.

Standard filtering techniques ignore these variations, potentially degrading the effectiveness of visual representation for prosthetic users. 

### **Biological Insight**  
The human visual system inherently adapts spatial frequency processing according to the **cortical magnification factor**, allocating more resources to central vision where acuity is highest. Our approach draws inspiration from this principle, adjusting visual processing computationally to reflect the **non-uniform phosphene distribution** in prosthetic vision. 

---

## **Mathematical Framework**  
Central to this method is the relationship between **phosphene density** and the **scale of spatial filtering**. We define the sigma (\(\sigma\)) of spatial frequency modulation using the formula:

$$\sigma = \frac{1}{\pi \sqrt{\text{density}}}$$

Where:  
- $\sigma$ determines the **spatial extent of filtering** (e.g., the standard deviation in a Laplacian of Gaussian filter).
- **density** is the **local phosphene density**, guiding spatial frequency adjustments.

This framework ensures that **spatial filtering is modulated adaptively** based on the underlying phosphene map, providing a biologically relevant transformation of visual information.

---

## **AdaptiveConv2d: Spatially Modulated Convolutional Filtering**  
The **AdaptiveConv2d** module is the primary implementation of this adaptive filtering concept. Unlike standard convolutional layers, which apply uniform filters across the entire image, **AdaptiveConv2d** dynamically modulates its behavior according to the local phosphene distribution.

### **Features**  
- **Spatially Varying Filters**: Filters are **adaptive** and change across the image based on phosphene density.
- **Multi-Frequency Integration**: Supports **multi-frequency responses** using Laplacian of Gaussian (LoG) filters, **Gaussian blurring**, or **neural-generated kernels**.
- **Multi-Channel Processing**: Integrates across color channels to highlight **edges and features** that are perceptually significant, using merge strategies like **maximum or summed activations**.
- **Biologically Inspired Normalization**: Balances frequency content to ensure that both **high- and low-frequency details** are perceptually relevant.

### **Example Implementation**  
```python
from phosphene.density import VisualFieldMapper
from components.adaptive_conv2d import AdaptiveConv2d

mapper = VisualFieldMapper(simulator=simulator)

sigma_kde_pix = mapper.build_sigma_map_from_density(density_kde, space="pixel")
sigma_map_tensor = torch.Tensor(sigma_kde_pix).float().cuda().unsqueeze(0).unsqueeze(0).detach()

net = AdaptiveConv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=1,
    kernel_type="LoG",
    log_kernel_size=137,
    sigma_function="map",
    sigma_map=sigma_map_tensor,
    padding_mode="reflect"
).cuda()

foa_xy = torch.tensor([img.shape[3] // 2, img.shape[2] // ]).float().cuda().unsqueeze(0).detach()

filtered_img = net(image_tensor, foa_xy)
```

**AdaptiveConv2d** facilitates **spatially aware processing** by tailoring filtering operations to the visual characteristics defined by phosphene density maps.

---

## **Applications & Future Directions**  
This approach has broad implications for **prosthetic vision simulation** and **neuroadaptive image processing**:

- **Realistic Visual Prosthetic Simulation**: Generates more accurate visual representations based on actual prosthetic constraints.
- **Neuroadaptive Visual Processing**: Adapts image processing to enhance the interpretability of visual scenes in low-resolution contexts.
- **Adaptive AI Models**: Integrate into deep learning pipelines where spatially adaptive filtering is required.

### **Future Developments**  
- **Dynamic Phosphene Representation Learning**: Developing task-specific phosphene maps.
- **Learnable Adaptive Filters**: Optimizing sigma maps and spatial filters directly through neural network training.
- **Biologically Inspired Models**: Incorporating principles of visual perception for improved visual representations.

---

## **Conclusion**  
This repository introduces a computational framework for **phosphene-aware spatial frequency processing** that links the **density of phosphenes in prosthetic vision** to **adaptive visual filtering**. By leveraging this relationship, we create a more nuanced representation of visual scenes that can significantly improve the quality of visual input provided by prosthetic devices.

🚀 **This work establishes the foundation for biologically informed image processing, adaptive neural models, and innovative solutions in vision restoration technology.** 
