# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to compute signal-to-noise ratio from detected spots.
"""

import numpy as np

import bigfish.stack as stack
from .spot_modeling import _get_spot_volume
from .spot_modeling import _get_spot_surface


# ### SNR ###

def compute_snr_spots(image, spots, voxel_size_z=None, voxel_size_yx=100,
                      psf_z=None, psf_yx=200, background_factor=3):
    """Compute Signal-to-Noise ratio for every detected spot.

        SNR = (max_spot_signal - mean_background) / std_background

    Background is a larger region surrounding the spot region.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64 or np.float64
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
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float),
                          background_factor=(float, int))
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_range_value(image, min_=0)
    stack.check_array(spots, ndim=2, dtype=[np.float64, np.int64])

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

    # cast spots coordinates if needed
    if spots.dtype == np.float64:
        spots = np.round(spots).astype(np.int64)

    # compute spot radius
    radius_signal_ = stack.get_radius(voxel_size_z=voxel_size_z,
                                      voxel_size_yx=voxel_size_yx,
                                      psf_z=psf_z, psf_yx=psf_yx)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * background_factor for i in radius_signal_)

    # ceil radii
    radius_background = np.ceil(radius_background_).astype(np.int)

    # loop over spots
    snr_spots = []
    for spot in spots:

        # extract spot images
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_background_yx = radius_background[-1]
        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            max_signal = image[spot_z, spot_y, spot_x]
            spot_background, _ = _get_spot_volume(
                image, spot_z, spot_y, spot_x,
                radius_background_z, radius_background_yx)
        else:
            max_signal = image[spot_y, spot_x]
            spot_background, _ = _get_spot_surface(
                image, spot_y, spot_x, radius_background_yx)

        # compute mean background
        sum_background = np.sum(spot_background) - max_signal
        n_background = spot_background.size - 1
        mean_background = sum_background / n_background

        # compute standard deviation background
        std_background = np.std(spot_background)

        # compute SNR
        snr = (max_signal - mean_background) / std_background
        snr_spots.append(snr)

    #  format results
    snr_spots = np.array(snr_spots, dtype=np.float64)

    return snr_spots
