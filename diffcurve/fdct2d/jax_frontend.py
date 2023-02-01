'''2D discrete curvelet transform in jax'''
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


def jax_perform_fft2(spatial_input):
    """Perform fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(spatial_input),
                                         norm='ortho'))


def jax_perform_ifft2(frequency_input):
    """Perform inverse fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """
    return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(frequency_input),
                                          norm='ortho'))


def jax_fdct_2d(img, curvelet_system):
    """2d fast discrete curvelet in jax

    Args:
        img: 2D array
        curvelet_system: curvelet waveforms in the frequency domain

    Returns:
        coeffs: curvelet coefficients
    """
    x_freq = jax_perform_fft2(img)
    conj_curvelet_system = jnp.conj(curvelet_system)
    coeffs = jax_perform_ifft2(x_freq * conj_curvelet_system)
    return coeffs


def jax_ifdct_2d(coeffs, curvelet_system, curvelet_support_size):
    """2d inverse fast discrete curvelet in jax

    Args:
        coeffs: curvelet coefficients
        curvelet_system: curvelet waveforms in the frequency domain
        curvelet_support_size: size of the support of each curvelet wedge

    Returns:
        decom: image decomposed in different scales and orientation in the
        curvelet basis.
    """
    coeffs_freq = jax_perform_fft2(coeffs)

    decomp = jax_perform_ifft2(
        coeffs_freq * curvelet_system)\
        * jnp.expand_dims(curvelet_support_size, [1, 2])
    return decomp
