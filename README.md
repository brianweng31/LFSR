# Light Field Super-Resoultion by Superimposed Projections

add image tutorial:
https://stackoverflow.com/questions/41604263/how-do-i-display-local-image-in-markdown

## Introduction

Light Field (LF) Imaging represents an emerging and innovative display system designed to address the issue of Vergence Accommodation Conflict (VAC) often encountered in traditional AR displays. Nonetheless, LF imaging is not without its inherent challenges.

One of the most prominent challenges is the inherent tradeoff between angular and spatial resolution. In essence, if one seeks to enhance angular information, the resulting refocused image may exhibit reduced resolution. Conversely, if the goal is to improve the spatial resolution of the refocused image, some angular information may be sacrificed in the process.

Additionally, it's worth noting that the refocused image is generated through a fundamental operation known as "shift-and-add" applied to sub-view images.

Hence, our objective is to harness this operation of overlapping images to enhance the spatial resolution of the refocused image.

## Methods
### Inspiration
Drawing inspiration from the concept introduced in [DVC07], which aimed to improve image resolution through the use of multiple projectors, we find a parallel with the setup of light field displays. In their approach, they generated an equal number of sub-images as there were projectors and then aligned these sub-images using precise sub-pixel shifts.

[Image] sub-pixel shift
[Image] Their reault

This methodology appears to be ideally suited to the requirements of light field displays, which rely on sub-view images as their input. The sub-pixel shifting aligns closely with the "shift" component of the "shift-and-add" operation, while the process of overlapping resonates with the "add" component.


### Our approach
Below is the example of our appraoch to the problem that we are dealing with.

#### Expample
Resolution of input light field:
- angular resolution: 3x3
- sptial resolution: 510x510

The light field display has a restriction wherein each projector can only project images with a spatial resolution of 170x170, whereas our available sub-view images boast a higher spatial resolution of 510x510.

To address this, we employ a specific downsampling technique, which will be elaborated upon shortly. Following the downsampling of the sub-view images, we proceed with the corresponding sub-pixel shift, as outlined below. As a result of these processes, we ultimately obtain a refocused image with a spatial resolution of 510x510.

[Image] 510x510 to 170x170

#### Simulation of Sub-Pixel Shifts
As an example, we show how the top-left downsample image will be shifted. 

[Image] Top-left image shift

For software simulation, our initial step involves upscaling the downsampled 170x170 image to a 510x510 pixel resolution, as illustrated in the image presented. Subsequently, we shift the image by one pixel in an upward and leftward direction.


#### Downsampling
For downsampling, opting for a single pixel within the 3x3 box inevitably leads to aliasing issues. To address this concern, we implement a more sophisticated approach: downsampling each sub-view image using a distinct 2D filter sized at 7x7. There are a total of 3x3 such filters, one for each sub-view image, and they are initially configured as Gaussian filters.

Subsequently, we initiate a process of back-propagation, continuing until the defined loss reaches a stable state.

[Image] downsample gaussian filter

Furthermore, given that we perform a sub-pixel shift on the downsampled sub-view images, it becomes imperative to apply a pixel-shift in the reverse direction before downsampling. This step is essential to maintain consistency with the inherent light field properties.

[Image] reverse pixel-shift beforehand


## Results

The method we aim to surpass as a baseline downsamples each sub-view image individually using distinct filters (3x3 sets in total, one for each sub-view image). Subsequently, it merely enlarges these images to a resolution of 510x510 without incorporating sub-pixel shifting.

In the context of this aforementioned approach, we train the downsampling filters using the HCI dataset, which comprises 20 training light fields and 4 testing light fields.

[Image] metric result comparison

- Basleine: Baseline method
- SR: Our method (Super-resolution)




## Further Problems & Current Approach

The above approach seeks to improve the the image quality of the entire image. However, as we know that light field imaging allows us to do refocusing, we should instead focus on the image quality in the in-focus regions.

Therefore, we should adjust the loss calculation that we to for the back propagation and should give the loss in the in-focus regions more weight. Below, we will further discuss the loss calculation we propose.

### Loss Calculation
Light field refousing is based on avaraging all the sub-view images. For the in-focus regions, it's easy to guess that the pixel values of each sub-view image won't differ too much. 

As the result, we then caculate the pixel-wise difference between each sub-view image and the central-view image, denoted $\text{diff}_{i,j}$ for the sub-view image in the angular coordinate $(i,j)$.

$$\text{diff}_{i,j}(x,y) = |\text{sub-view}_{i,j}(x,y)-\text{central-view}_{i,j}(x,y)|$$

After obtaining the matrices for each sub-view image, we construct the overall "Difference" matrix, denoted $\text{Diff}$, by averaging those matrices.

$$\text{Diff}(x,y) = \frac{1}{\text{ang}_x * \text{ang}_y} \sum_{i,j} \text{diff}_{i,j}(x,y)$$

Note that the size of $\text{Diff}$ will be equivalent to the spatial resolution of the input light field (510x510 in the above case). Also, it's not hard to guess that the value correspnds to the in-focus regions will be relatively small.


### Experiment

## Reference

[DVC07] Amera-Venkata N., Chang N. L.: Realizing Super-Resolution with Superimposed Projection. In Proc. of IEEE International Workshop on ProjectorCamera Systems (ProCams) (2007)
