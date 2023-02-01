


Diffcurve: curvelet transform in differentiable programming
=====================================


Diffcurve is a Python library that integrates the curvelet transform into differentiable programming pipelines such as PyTorch and JAX. The curvelet transform decomposes an image into a set of directional components at different scales. It is useful for representing images with sharp discontinuities such as edges and corners, as it allows for a sparse representation of these features while preserving high fidelity.


# Usage 

Diffcurve consists of two steps:

1. Generating the curvelet waveforms for images at a prescribed size. This step  requires MATLAB to be installed on your device, as it calls a few MATLAB functions from Python using the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

2. Using the curvelet waveforms to perform the curvelet transform. The same set of curvelet waveforms can be used for all images of the same size. This step is differentiable and supports deep learning APIs. It can be run on GPUs and TPUs.


To view a complete demonstration of Diffcurve, please refer to the Jupyter notebooks located in the `diffcurve/diffcurve/notebooks/` directory. In what follows, we provide a brief overview of the main steps.


## Step 1: Generate a curvelet system

We refer a curvelet system as a set of curvelet waveforms of different scales and orientations in the Fourier domain.

```python
dct_kwargs = {
    'is_real': 0.0, # complex-valued curvelets
    'finest': 2.0, # use wavelets at the finest scale
    'nbscales': 6.0, # number of scales
    'nbangles_coarse': 16.0, # number of angles at the 2nd coarsest scale
    }

# Under the hood, the following `get_curvelet_system` function
# calls two MATLAB functions, `fdct_wrapping.m` and `ifdct_wrapping.m`
# in `diffcurve/diffcurve/fdct2d/`.
curvelet_system, curvelet_coeff_dim = get_curvelet_system(
    img_length=512, img_width=512, dct_kwargs)
```


## Step 2: Perform curvelet transform using the generated curvelet system



In PyTorch:

```python
from diffcurve.fdct2d.torch_frontend import torch_fdct_2d, torch_ifdct_2d

# Forward curvelet transform. The input image lena_img
# is of size (512, 512), which is consistent with the 
# shape of `curvelet_system`.
torch_coeff = torch_fdct_2d(
    torch.from_numpy(lena_img),
    torch.from_numpy(curvelet_system)) 
                            
# Inverse curvelet transform. The tensor `decomp` below is a weighted 
# collection of curvelets that represent the Lena_img image at 
# different scales and orientations. By summing the array with
# decomp.sum(0), we can reconstruct the Lena_img image 
# with high fidelity.
torch_decomp = torch_ifdct_2d(torch_coeff,
    torch.from_numpy(curvelet_system),
    torch.from_numpy(curvelet_support_size))


coeff = np.array(torch_coeff.detach().cpu())
decomp = np.array(torch_decomp.detach().cpu())

mse = np.mean( (decomp.sum(0).real - lena_img) ** 2 )

print(mse) #1.9523397759671602e-31

```

In JAX:

```python
from diffcurve.fdct2d.jax_frontend import jax_fdct_2d, jax_ifdct_2d

# Forward curvelet transform. The input image lena_img is of size 
# (512, 512), which is consistent with the shape of `curvelet_system`.
coeff = jax_fdct_2d(lena_img, curvelet_system)

# Inverse curvelet transform. The tensor `decomp` below is a weighted 
# collection of curvelets that represent the Lena_img image at 
# different scales and orientations. By summing the array with
# decomp.sum(0), we can reconstruct the Lena_img image 
# with high fidelity.
decomp = jax_ifdct_2d(coeff, curvelet_system,
                      curvelet_support_size )

mse = np.mean( (decomp.sum(0).real - lena_img) ** 2 )

print(mse) #1.1598556547941844e-31

```




# Installation

```
conda env create -n diffcurve --file environment.yml
```

Install MATLAB and [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).



Diffcurve is currently under active development. If you have any suggestions, please submit an issue or contact me at t.liu at unibas.ch.