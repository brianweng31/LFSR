# Light Field Super-Resoultion by Superimposed Projections

## Introduction

Light Field (LF) Imaging represents an emerging and innovative display system designed to address the issue of Vergence Accommodation Conflict (VAC) often encountered in traditional AR displays. Nonetheless, LF imaging is not without its inherent challenges.

One of the most prominent challenges is the inherent tradeoff between angular and spatial resolution. In essence, if one seeks to enhance angular information, the resulting refocused image may exhibit reduced resolution. Conversely, if the goal is to improve the spatial resolution of the refocused image, some angular information may be sacrificed in the process.

Additionally, it's worth noting that the refocused image is generated through a fundamental operation known as "shift-and-add" applied to sub-view images.

Hence, our objective is to harness this operation of overlapping images to enhance the spatial resolution of the refocused image.

## Methods
### Inspiration
Drawing inspiration from the concept introduced in [DVC07], which aimed to improve image resolution through the use of multiple projectors, we find a parallel with the setup of light field displays. In their approach, they generated an equal number of sub-images as there were projectors and then aligned these sub-images using precise sub-pixel shifts.

- Sub-pixel shift
<p align="center">
<img width="493" alt="subpixel_shift" src="https://github.com/brianweng31/LFSR/assets/55650127/1ae424da-1d6c-4681-a0f5-2c2506fab319">
</p>

- Their results

|**regular projection**|**superimpoed projection**|
|:-:|:-:|
|![regular_projection](https://github.com/brianweng31/LFSR/assets/55650127/89fcc781-bc79-4fd5-b0b3-bfd46d11aff9)| ![superimposed_projection](https://github.com/brianweng31/LFSR/assets/55650127/597b4efa-4d5e-43f0-9cbb-33bb33f4a44b)|

This methodology appears to be ideally suited to the requirements of light field displays, which rely on sub-view images as their input. The sub-pixel shifting aligns closely with the "shift" component of the "shift-and-add" operation, while the process of overlapping resonates with the "add" component.


### Our approach
Below is the example of our appraoch to the problem that we are dealing with.

#### Expample
Resolution of input light field:
- angular resolution: 3x3
- sptial resolution: 510x510

The light field display has a restriction wherein each projector can only project images with a spatial resolution of 170x170, whereas our available sub-view images boast a higher spatial resolution of 510x510.

To address this, we employ a specific downsampling technique, which will be elaborated upon shortly. Following the downsampling of the sub-view images, we proceed with the corresponding sub-pixel shift, as outlined below. As a result of these processes, we ultimately obtain a refocused image with a spatial resolution of 510x510.

<p align="center">
<img width="700" alt="5102170" src="https://github.com/brianweng31/LFSR/assets/55650127/90b2bbc9-79fc-4c3e-b6c6-d503785a716f">
</p>

#### Simulation of Sub-Pixel Shifts
As an example, we show how the top-left downsample image will be shifted. 

<p align="center">
<img width="700" alt="topleft_view" src="https://github.com/brianweng31/LFSR/assets/55650127/64bbf278-a0fd-4ee7-9db2-9e1b9cd7a570">
</p>


For software simulation, our initial step involves upscaling the downsampled 170x170 image to a 510x510 pixel resolution, as illustrated in the image presented. Subsequently, we shift the image by one pixel in an upward and leftward direction.


#### Downsampling
For downsampling, opting for a single pixel within the 3x3 box inevitably leads to aliasing issues. To address this concern, we implement a more sophisticated approach: downsampling each sub-view image using a distinct 2D filter sized at 7x7. There are a total of 3x3 such filters, one for each sub-view image, and they are initially configured as Gaussian filters.

Subsequently, we initiate a process of back-propagation, continuing until the defined loss reaches a stable state.

<p align="center">
<img width="550" alt="gaussian_filter" src="https://github.com/brianweng31/LFSR/assets/55650127/4275c107-8b30-4315-9d19-4c3bd94759e6">
</p>

Furthermore, given that we perform a sub-pixel shift on the downsampled sub-view images, it becomes imperative to apply a pixel-shift in the reverse direction before downsampling. This step is essential to maintain consistency with the inherent light field properties.

<p align="center">
<img width="906" alt="reverse_shift" src="https://github.com/brianweng31/LFSR/assets/55650127/d1e5a93e-ea42-48be-9506-b0ddc058af1b">
</p>


## Results

The method we aim to surpass as a baseline downsamples each sub-view image individually using distinct filters (3x3 sets in total, one for each sub-view image). Subsequently, it merely enlarges these images to a resolution of 510x510 without incorporating sub-pixel shifting.

In the context of this aforementioned approach, we train the downsampling filters using the HCI dataset, which comprises 20 training light fields and 4 testing light fields.

| | Baseline | SR |
| :-: | :-: | :-: |
| PSNR | 31.66 | <ins>34.62</ins> |
| SSIM | 0.978 | <ins>0.982</ins> |
| GMSD | 0.027 | <ins>0.023</ins> |
| LPIPS | 0.206 | <ins>0.133</ins> |

- Baseline: Baseline method
- SR: Our method (Super-resolution)

## Further Problems & Current Approach


The aforementioned method aims to enhance the overall image quality. Nevertheless, given the capabilities of light field imaging to enable refocusing, our attention should be directed towards improving the image quality specifically within the in-focus areas.

Hence, it becomes necessary to modify the loss calculation used during backpropagation, assigning greater significance to the in-focus regions. In the following section, we will delve into the details of the proposed adjustments to the loss calculation.

### Loss Calculation
Light field refocusing relies on the shifting and averaging of all sub-view images. When examining pixels sharing the same spatial coordinates $(x, y)$ across each sub-view image, it is reasonable to assume that their values will exhibit minimal variation when situated within the in-focus area.

As a result, our initial step involves computing the "Difference" matrix, as defined below, and subsequently utilizing this matrix to calculate the loss accordingly.

#### Difference Matrix
First, we compute the pixel-wise difference between each sub-view image and the central-view image, denoting this as $\text{diff}_{i,j}$ for the sub-view image located at angular coordinates $(i,j)$.

```math
\text{diff}_{i,j}(x,y) = |\text{sub-view}_{i,j}(x,y)-\text{central-view}_{i,j}(x,y)|
```

Once we've obtained these difference matrices for each sub-view image, we proceed to create the collective "Difference" matrix, denoted as $\text{Diff}$, by averaging these individual matrices.

```math
\text{Diff}(x,y) = \frac{1}{\text{ang}_x * \text{ang}_y} \sum_{i,j} \text{diff}_{i,j}(x,y)
```

Note that the dimensions of $\text{Diff}$ will match the spatial resolution of the input light field, which is 510x510 in the scenario described above. Furthermore, it's reasonable to expect that the values corresponding to the in-focus regions will be relatively small.

#### Actual Loss Calculation
We will introduce two distinct methods for loss calculation, each of which will be discussed separately.

To begin, we establish the definitions of two critical components:

- $\textbf{SR} :$ refocused image created using our method for super-resolution
- $\textbf{GT} :$ ground truth image derived from averaging the values across all input sub-view images.

It is important to note that both of these images possess a spatial resolution identical to that of the input light field, which, in the context described above, is 510x510.

##### Exponential Weight

The loss function is formulated as follows:

```math
loss = \underset{x,y}{\text{avg}}\{|\text{SR}(x,y)-\text{GT}(x,y)|*e^{-\alpha\text{Diff}(x,y)}\}
```

In the above definition, it's evident that regions potentially in focus are assigned greater weights.

##### Threshold

In this method, we first establish a threshold value and generate the threshold matrix $\Gamma$ accordingly:

```math
\begin{align*}
&\text{set } \gamma = \text{threshold value} \\
&\Gamma(x,y) = 
\begin{cases}
    1 & ,\text{Diff}(x,y) \leq \gamma \\
    0 & ,\text{Diff}(x,y) > \gamma
\end{cases} \\
\end{align*}
```

Subsequently, the loss is defined as:

```math
\text{loss} = \underset{x,y}{\text{avg}}\{|\text{SR}(x,y)-\text{GT}(x,y)| * \Gamma(x,y)\}
```

Once again, this formulation assigns greater weight to regions potentially in focus.


### Experiment Results
We tested out method of loss calculation on two focus planes of a single light field data.

|**Far plane**|**Close plane**|
|:-:|:-:|
|<img width="303" alt="far_plane" src="https://github.com/brianweng31/LFSR/assets/55650127/46554e3d-7131-4b54-8fd1-0195a11365b9">|<img width="304" alt="close_plane" src="https://github.com/brianweng31/LFSR/assets/55650127/2fe442a4-b113-454c-9424-d10301f41664">|

The following two sets of images show the comparison of the generated results. Note that downsampling of one of the method "no filter" is 

<p align="center">
<img width="100" alt="nofilter" src="https://github.com/brianweng31/LFSR/assets/55650127/1dac855e-0a1e-4989-850a-b4f87c0dc613">
</p>

- Far plane

|**Expoenential Weight**|**Threshold**|**No Filter**|**Gorund Truth**|
|:-:|:-:|:-:|:-:|
| <img width="170" alt="far_exponential" src="https://github.com/brianweng31/LFSR/assets/55650127/f32d0c5e-86d6-4ad2-b64d-f784331664a7"> | <img width="170" alt="far_threshold" src="https://github.com/brianweng31/LFSR/assets/55650127/dd1a7b26-40e0-4e61-b131-44ca30409309"> | <img width="170" alt="far_nofilter" src="https://github.com/brianweng31/LFSR/assets/55650127/e678e1af-8be3-4401-b19a-0a7644b4580c"> | <img width="170" alt="far_groundtruth" src="https://github.com/brianweng31/LFSR/assets/55650127/d34b6a15-b8fa-4a22-b8f2-7a49cb855a10"> |

- Close plane

|**Expoenential Weight**|**Threshold**|**No Filter**|**Gorund Truth**|
|:-:|:-:|:-:|:-:|
| <img width="170" alt="close_exponential" src="https://github.com/brianweng31/LFSR/assets/55650127/fdc50234-5f9d-4cde-9052-09b13a1cb96f"> | <img width="170" alt="close_threshold" src="https://github.com/brianweng31/LFSR/assets/55650127/1b53e7cf-ad58-4e13-b722-3006240a273c"> | <img width="170" alt="close_nofilter" src="https://github.com/brianweng31/LFSR/assets/55650127/058972e0-91b1-4adf-82fa-77efba5b16c0"> | <img width="170" alt="close_groundtruth" src="https://github.com/brianweng31/LFSR/assets/55650127/2e2ba014-d8d4-4c82-afbc-687a64b84043"> |

Questions:


### Experiment

## Reference

[DVC07] Amera-Venkata N., Chang N. L.: Realizing Super-Resolution with Superimposed Projection. In Proc. of IEEE International Workshop on ProjectorCamera Systems (ProCams) (2007)
