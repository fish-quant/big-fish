# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions used to detect and clean noisy images.
"""

import numpy as np

from scipy.ndimage import convolve

from .utils import check_array, check_parameter, check_range_value, get_radius


# ### Signal-to-Noise ratio ###

def compute_snr_spots(image, spots, voxel_size_z=None, voxel_size_yx=100,
                      psf_z=None, psf_yx=200, background_factor=3):
    """Compute Signal-to-Noise ratio for every detected spot.

        SNR = (mean_spot_signal - mean_background) / std_background

    Signal corresponds to the spot region. Background is a larger region
    surrounding the spot region. Computation time will be sensitive to the
    image surface or volume, not the number of spots.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots detected, with shape (nb_spots, 3) or
        (nb_spots, 2). One coordinate per dimension (zyx or yx coordinates).
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we
        consider a 2-d spot.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer. If None, we consider a 2-d spot.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.
    background_factor : float or int
        Factor to define the surrounding background of the spot from its
        radius.
    Returns
    -------
    snr_spots : np.ndarray, np.float64
        Signal-to-Noise ratio for each spot.

    """
    # check parameters
    check_parameter(voxel_size_z=(int, float, type(None)),
                    voxel_size_yx=(int, float),
                    psf_z=(int, float, type(None)),
                    psf_yx=(int, float),
                    background_factor=(float, int))
    check_array(image,
                ndim=[2, 3],
                dtype=[np.uint8, np.uint16, np.float32, np.float64])
    check_range_value(image, min_=0)
    check_array(spots, ndim=2, dtype=np.int64)

    # check consistency between parameters
    ndim = image.ndim
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and psf_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'psf_z' parameter is missing.".format(ndim))
    if ndim != spots.shape[1]:
        raise ValueError("Provided image has {0} dimensions but 'spots' are "
                         "detected in {1} dimensions."
                         .format(ndim, spots.shape[1]))
    if ndim == 2:
        voxel_size_z, psf_z = None, None

    # compute spot radius
    radius_signal_ = get_radius(voxel_size_z=voxel_size_z,
                                voxel_size_yx=voxel_size_yx,
                                psf_z=psf_z, psf_yx=psf_yx)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * background_factor for i in radius_signal_)

    # ceil radii
    radius_signal = np.ceil(radius_signal_).astype(np.int)
    radius_background = np.ceil(radius_background_).astype(np.int)

    # kernels width
    width_signal = 1 + radius_signal * 2
    width_background = 1 + radius_background * 2

    # cast image in float if necessary
    if image.dtype in [np.uint8, np.uint16]:
        image_float = image.astype(np.float64)
    else:
        image_float = image

    # mean signal
    nb_signal = np.prod(width_signal)
    kernel_signal = np.ones(width_signal, dtype=np.float64)
    kernel_mean_signal = kernel_signal / nb_signal
    mean_signal = convolve(image_float, kernel_mean_signal)

    # mean background
    nb_background = np.prod(width_background) - nb_signal
    kernel_background = np.ones(width_background, dtype=np.float64)
    patch_signal = np.zeros(width_signal, dtype=np.float64)
    start = radius_background - radius_signal
    end = radius_background + radius_signal + 1
    if ndim == 3:
        kernel_background[start[0]:end[0],
                          start[1]:end[1],
                          start[2]:end[2]] = patch_signal
    else:
        kernel_background[start[0]:end[0], start[1]:end[1]] = patch_signal
    kernel_mean_background = kernel_background / nb_background
    mean_background = convolve(image_float, kernel_mean_background)

    # difference signal - background
    diff = np.subtract(mean_signal, mean_background,
                       out=np.zeros_like(mean_signal),
                       where=mean_signal > mean_background)

    # standard deviation background
    sum_1 = convolve(image_float ** 2, kernel_background)
    local_sum_background = convolve(image_float, kernel_background)
    sum_2 = np.multiply(mean_background, local_sum_background) * 2
    sum_3 = (mean_background ** 2) * nb_background
    std_background = np.sqrt((sum_1 - sum_2 + sum_3) / (nb_background - 1))

    # local signal-to-noise ratio
    snr = np.divide(diff, std_background,
                    out=np.zeros_like(diff),
                    where=std_background > 0)

    # spots signal-to-noise ratio
    if ndim == 3:
        snr_spots = snr[spots[:, 0], spots[:, 1], spots[:, 2]]
    else:
        snr_spots = snr[spots[:, 0], spots[:, 1]]

    return snr_spots
