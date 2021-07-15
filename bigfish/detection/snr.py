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
                      psf_z=None, psf_yx=200):
    """Compute signal-to-noise ratio (SNR) based on spot coordinates.

    .. math::

        \\mbox{SNR} = \\frac{\\mbox{max(spot signal)} -
        \\mbox{mean(background)}}{\\mbox{std(background)}}

    Background is a region twice larger surrounding the spot region. Only the
    y and x dimensions are taking into account to compute the SNR.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64 or np.float64
        Coordinate of the spots, with shape (nb_spots, 3) or (nb_spots, 2).
        One coordinate per dimension (zyx or yx coordinates).
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we consider
        a 2-d spot.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan, in
        nanometer. If None, we consider a 2-d spot.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan, in
        nanometer.

    Returns
    -------
    snr : float
        Median signal-to-noise ratio computed for every spots.

    """
    # check parameters
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float))
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

    # cast image if needed
    image_to_process = image.copy().astype(np.float64)

    # clip coordinate if needed
    if ndim == 3:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)
        spots[:, 2] = np.clip(spots[:, 2], 0, image_to_process.shape[2] - 1)
    else:
        spots[:, 0] = np.clip(spots[:, 0], 0, image_to_process.shape[0] - 1)
        spots[:, 1] = np.clip(spots[:, 1], 0, image_to_process.shape[1] - 1)

    # compute spot radius
    radius_signal_ = stack.get_radius(voxel_size_z=voxel_size_z,
                                      voxel_size_yx=voxel_size_yx,
                                      psf_z=psf_z, psf_yx=psf_yx)

    # compute the neighbourhood radius
    radius_background_ = tuple(i * 2 for i in radius_signal_)

    # ceil radii
    radius_signal = np.ceil(radius_signal_).astype(np.int)
    radius_background = np.ceil(radius_background_).astype(np.int)

    # loop over spots
    snr_spots = []
    for spot in spots:

        # extract spot images
        spot_y = spot[ndim - 2]
        spot_x = spot[ndim - 1]
        radius_signal_yx = radius_signal[-1]
        radius_background_yx = radius_background[-1]
        edge_background_yx = radius_background_yx - radius_signal_yx
        if ndim == 3:
            spot_z = spot[0]
            radius_background_z = radius_background[0]
            max_signal = image_to_process[spot_z, spot_y, spot_x]
            spot_background_, _ = _get_spot_volume(
                image_to_process, spot_z, spot_y, spot_x,
                radius_background_z, radius_background_yx)
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border (along y and x dimensions)
            expected_size = (2 * radius_background_yx + 1) ** 2
            actual_size = spot_background.shape[1] * spot_background.shape[2]
            if expected_size != actual_size:
                continue

            # remove signal from background crop
            spot_background[:,
                            edge_background_yx:-edge_background_yx,
                            edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        else:
            max_signal = image_to_process[spot_y, spot_x]
            spot_background_, _ = _get_spot_surface(
                image_to_process, spot_y, spot_x, radius_background_yx)
            spot_background = spot_background_.copy()

            # discard spot if cropped at the border
            expected_size = (2 * radius_background_yx + 1) ** 2
            if expected_size != spot_background.size:
                continue

            # remove signal from background crop
            spot_background[edge_background_yx:-edge_background_yx,
                            edge_background_yx:-edge_background_yx] = -1
            spot_background = spot_background[spot_background >= 0]

        # compute mean background
        mean_background = np.mean(spot_background)

        # compute standard deviation background
        std_background = np.std(spot_background)

        # compute SNR
        snr = (max_signal - mean_background) / std_background
        snr_spots.append(snr)

    #  average SNR
    if len(snr_spots) == 0:
        snr = 0.
    else:
        snr = np.median(snr_spots)

    return snr
