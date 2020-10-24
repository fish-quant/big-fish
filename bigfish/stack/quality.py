# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions used to detect and clean noisy images.
"""

import numpy as np

from .utils import check_array, check_parameter, check_range_value, get_radius


# ### Signal-to-Noise ratio ###

def compute_snr(image):
    """Compute Signal-to-Noise ratio for an image.

        SNR = mean / std

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x). Values should be positive.

    Returns
    -------
    snr : float
        Signal-to-Noise ratio of the image.

    """
    # check parameters
    check_array(image,
                ndim=[2, 3],
                dtype=[np.uint8, np.uint16, np.float32, np.float64])
    check_range_value(image, min_=0)

    # compute signal-to-noise ratio
    snr = image.mean() / image.std()

    return snr


def compute_snr_spots(image, spots, voxel_size_z=None, voxel_size_yx=100,
                      psf_z=None, psf_yx=200, neighbourhood_factor=3):
    """Compute Signal-to-Noise ratio for every detected spot.

        SNR = (mean_spot_signal - mean_background) / std_background

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
    neighbourhood_factor : float or int
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
                    neighbourhood_factor=(float, int))
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
    radius_spot = get_radius(voxel_size_z=voxel_size_z,
                             voxel_size_yx=voxel_size_yx,
                             psf_z=psf_z, psf_yx=psf_yx)

    # compute the neighbourhood size
    neighbourhood_spot = tuple(i * neighbourhood_factor for i in radius_spot)

    # loop over every spot
    snr_spots = []
    for spot in spots:

        # compute SNR per spot
        if ndim == 3:
            snr_spot = _compute_snr_per_spot_3d(
                image, spot, radius_spot, neighbourhood_spot)
        else:
            snr_spot = _compute_snr_per_spot_2d(
                image, spot, radius_spot, neighbourhood_spot)
        snr_spots.append(snr_spot)

    # format results
    snr_spots = np.array(snr_spots)

    return snr_spots


def _compute_snr_per_spot_3d(image, spot, radius_spot, neighbourhood_spot):
    """Extract a 3-d background and spot volume then compute the spot
    Signal-to-Noise ratio.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x).
    spot : np.ndarray, np.int64
        Coordinate of a spot, with shape (1, 3). One coordinate per dimension
        (zyx coordinates).
    radius_spot : Tuple
        Radius of the spot, one scalar per dimension (z, y, x).
    neighbourhood_spot : Tuple
        Radius of the spot background, one scalar per dimension (z, y, x).

    Returns
    -------
    snr_spot : float
        Signal-to-Noise ratio of the spot.

    """
    # get spot coordinate
    spot_z, spot_y, spot_x = spot

    # get spot and background radii
    radius_z, radius_yx, radius_yx = radius_spot
    background_z, background_yx, background_yx = neighbourhood_spot

    # extract volume around spot
    z_spot_min = max(0, int(spot_z - radius_z))
    z_spot_max = min(image.shape[0], int(spot_z + radius_z))
    y_spot_min = max(0, int(spot_y - radius_yx))
    y_spot_max = min(image.shape[1], int(spot_y + radius_yx))
    x_spot_min = max(0, int(spot_x - radius_yx))
    x_spot_max = min(image.shape[2], int(spot_x + radius_yx))
    image_spot = image[z_spot_min:z_spot_max + 1,
                       y_spot_min:y_spot_max + 1,
                       x_spot_min:x_spot_max + 1]

    # compute signal
    signal = image_spot.mean()

    # remove spot values
    image_bis = image.copy().astype(np.float64)
    mask = np.ones_like(image_spot) * -1
    image_bis[z_spot_min:z_spot_max + 1,
              y_spot_min:y_spot_max + 1,
              x_spot_min:x_spot_max + 1] = mask

    # extract a larger volume around spot to get the background
    z_spot_min = max(0, int(spot_z - background_z))
    z_spot_max = min(image.shape[0], int(spot_z + background_z))
    y_spot_min = max(0, int(spot_y - background_yx))
    y_spot_max = min(image.shape[1], int(spot_y + background_yx))
    x_spot_min = max(0, int(spot_x - background_yx))
    x_spot_max = min(image.shape[2], int(spot_x + background_yx))
    image_background = image_bis[z_spot_min:z_spot_max + 1,
                                 y_spot_min:y_spot_max + 1,
                                 x_spot_min:x_spot_max + 1]

    # compute background and noise
    image_background = image_background[image_background >= 0]
    background = image_background.mean()
    noise = max(image_background.std(), 10e-6)

    # compute SNR
    snr_spot = max(0, (signal - background)) / noise

    return snr_spot


def _compute_snr_per_spot_2d(image, spot, radius_spot, neighbourhood_spot):
    """Extract a 2-d background and spot surface then compute the spot
    Signal-to-Noise ratio.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (y, x).
    spot : np.ndarray, np.int64
        Coordinate of a spot, with shape (1, 2). One coordinate per dimension
        (yx coordinates).
    radius_spot : Tuple
        Radius of the spot, one scalar per dimension (y, x).
    neighbourhood_spot : Tuple
        Radius of the spot background, one scalar per dimension (y, x).

    Returns
    -------
    snr_spot : float
        Signal-to-Noise ratio of the spot.

    """
    # get spot coordinate
    spot_y, spot_x = spot

    # get spot and background radii
    radius_yx, radius_yx = radius_spot
    background_yx, background_yx = neighbourhood_spot

    # extract surface around spot
    y_spot_min = max(0, int(spot_y - radius_yx))
    y_spot_max = min(image.shape[0], int(spot_y + radius_yx))
    x_spot_min = max(0, int(spot_x - radius_yx))
    x_spot_max = min(image.shape[1], int(spot_x + radius_yx))
    image_spot = image[y_spot_min:y_spot_max + 1,
                       x_spot_min:x_spot_max + 1]

    # compute signal
    signal = image_spot.mean()

    # remove spot values
    image_bis = image.copy().astype(np.float64)
    mask = np.ones_like(image_spot) * -1
    image_bis[y_spot_min:y_spot_max + 1, x_spot_min:x_spot_max + 1] = mask

    # extract a larger surface around spot to get the background
    y_spot_min = max(0, int(spot_y - background_yx))
    y_spot_max = min(image.shape[0], int(spot_y + background_yx))
    x_spot_min = max(0, int(spot_x - background_yx))
    x_spot_max = min(image.shape[1], int(spot_x + background_yx))
    image_background = image_bis[y_spot_min:y_spot_max + 1,
                                 x_spot_min:x_spot_max + 1]

    # compute background and noise
    image_background = image_background[image_background >= 0]
    background = image_background.mean()
    noise = max(image_background.std(), 10e-6)

    # compute SNR
    snr_spot = max(0, (signal - background)) / noise

    return snr_spot
