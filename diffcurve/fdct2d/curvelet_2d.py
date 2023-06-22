'''Get curvelet system'''
from pathlib import Path
import numpy as np
import matlab.engine
from diffcurve import utils
from diffcurve.fdct2d.numpy_frontend import perform_fft2

project_root = utils.get_project_root()


def get_curvelet_system(img_length: int, img_width: int, dct_kwargs):
    """get curvelet waveforms in the frequency domain.

    Waveforms in the spatial domain can be recovered in the following way:

    from diffcurve.fdct2d.numpy_frontend import perform_ifft2
    curvelet_idx = 0
    perform_ifft2(curvelet_system[curvelet_idx])

    Args:
        img_length (int): the length of the image to be curvelet transformed
        img_width (int): the width of the image to be curvelet transformed
        dct_kwargs (int): settings for the curvelet transforms

    Returns:
        _type_: _description_
    """
    eng = matlab.engine.start_matlab()
    eng.cd(str(Path.joinpath(project_root, "diffcurve/fdct2d")))

    zeros = np.zeros((img_length, img_width))
    zero_coeffs = eng.fdct_wrapping(zeros,
                                    dct_kwargs['is_real'],
                                    dct_kwargs['finest'],
                                    dct_kwargs['nbscales'],
                                    dct_kwargs['nbangles_coarse'])

    all_scales_all_wedges_curvelet_coeffs = []
    curvelet_coeff_dim = []
    for (scale_idx, curvelets_scale) in enumerate(zero_coeffs):
        for (wedge_idx, curvelet_wedge) in enumerate(curvelets_scale):
            coeff_length = len(curvelet_wedge)
            coeff_width = len(curvelet_wedge[0])
            curvelet_coeff_dim.append((coeff_length, coeff_width))
            coord_vert = int(coeff_length / 2)
            coord_horiz = int(coeff_width / 2)

            tmp = zero_coeffs
            tmp[scale_idx][wedge_idx][coord_vert][coord_horiz] = 1
            out = np.array(eng.ifdct_wrapping(
                tmp, float(dct_kwargs['is_real']),
                float(img_length), float(img_width)))
            out = perform_fft2(out)
            all_scales_all_wedges_curvelet_coeffs.append(out)
            tmp[scale_idx][wedge_idx][coord_vert][coord_horiz] = 0

    all_scales_all_wedges_curvelet_coeffs = np.array(
        all_scales_all_wedges_curvelet_coeffs)
    curvelet_coeff_dim = np.array(curvelet_coeff_dim)
    return all_scales_all_wedges_curvelet_coeffs, curvelet_coeff_dim

