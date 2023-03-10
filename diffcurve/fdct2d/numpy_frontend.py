'''2D discrete curvelet transform in numpy'''
import numpy as np
from numpy import fft


def perform_fft2(spatial_input: np.ndarray):
    """Perform fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """

    return fft.fftshift(fft.fft2(fft.ifftshift(spatial_input), norm='ortho'))


def perform_ifft2(frequency_input: np.ndarray):
    """Perform inverse fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """
    return fft.fftshift(fft.ifft2(fft.ifftshift(frequency_input),
                                  norm='ortho'))


def numpy_fdct_2d(img, curvelet_system):
    """2d fast discrete curvelet in numpy

    Args:
        img: 2D array
        curvelet_system: curvelet waveforms in the frequency domain

    Returns:
        coeffs: curvelet coefficients
    """
    x_freq = perform_fft2(img)
    conj_curvelet_system = np.conj(curvelet_system)
    coeffs = perform_ifft2(x_freq * conj_curvelet_system)
    return coeffs


def numpy_ifdct_2d(coeffs, curvelet_system, curvelet_support_size):
    """2d inverse fast discrete curvelet in numpy

    Args:
        coeffs: curvelet coefficients
        curvelet_system: curvelet waveforms in the frequency domain
        curvelet_support_size: size of the support of each curvelet wedge

    Returns:
        decom: image decomposed in different scales and orientation in the
        curvelet basis.
    """

    coeffs_freq = perform_fft2(coeffs)

    decomp = perform_ifft2(
        coeffs_freq * curvelet_system)\
        * np.expand_dims(curvelet_support_size, [1, 2])
    return decomp
