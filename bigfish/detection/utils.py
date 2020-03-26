# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.stack submodule.
"""

import bigfish.stack as stack
import numpy as np

# TODO add function to calibrate psf


# ### Signal-to-Noise ratio ###

def compute_snr(image, sigma, minimum_distance=1,
                threshold_signal_detection=2000, neighbor_factor=3):
    """Compute Signal-to-Noise ratio for each spot detected.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    sigma : float or Tuple(float)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        float, the same sigma is applied to every dimensions.
    minimum_distance : int
        Minimum distance (in number of pixels) between two local peaks.
    threshold_signal_detection : float or int
        A threshold to detect peaks. Considered as a relative threshold if
        float.
    neighbor_factor : int or float
        The ratio between the radius of the neighborhood defining the noise
        and the radius of the signal.

    Returns
    -------

    """
    # cast image in np.float, apply LoG filter and find local maximum
    mask = log_lm(image, sigma, minimum_distance)

    # apply a specific threshold to filter the detected spots and compute snr
    l_snr = from_threshold_to_snr(image, sigma, mask,
                                  threshold_signal_detection,
                                  neighbor_factor)

    return l_snr


def from_threshold_to_snr(image, sigma, mask, threshold=2000,
                          neighbor_factor=3):
    """

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    sigma : float or Tuple(float)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        float, the same sigma is applied to every dimensions.
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    threshold : float or int
        A threshold to detect peaks. Considered as a relative threshold if
        float.
    neighbor_factor : int or float
        The ratio between the radius of the neighborhood defining the noise
        and the radius of the signal.

    Returns
    -------

    """
    # remove peak with a low intensity
    if isinstance(threshold, float):
        threshold *= image.max()
    mask_ = (mask & (image > threshold))

    # no spot detected
    if mask_.sum() == 0:
        return []

    # we get the xy coordinate of the detected spot
    spot_coordinates = np.nonzero(mask_)
    spot_coordinates = np.column_stack(spot_coordinates)

    # compute radius for the spot and the neighborhood
    s = np.sqrt(image.ndim)
    (z_radius, yx_radius) = (int(s * sigma[0]), int(s * sigma[1]))
    (z_neigh, yx_neigh) = (int(s * sigma[0] * neighbor_factor),
                           int(s * sigma[1] * neighbor_factor))

    # we enlarge our mask to localize the complete signal and not just
    # the peak
    kernel_size_z = 2 * z_radius + 1
    kernel_size_yx = 2 * yx_radius + 1
    kernel_size = (kernel_size_z, kernel_size_yx, kernel_size_yx)
    mask_ = ndi.maximum_filter(mask_, size=kernel_size,
                               mode='constant')

    # we define a binary matrix of noise
    noise = image.astype(np.float64)
    noise[mask_] = np.nan

    l_snr = []
    for i in range(spot_coordinates.shape[0]):
        (z, y, x) = (spot_coordinates[i, 0],
                     spot_coordinates[i, 1],
                     spot_coordinates[i, 2])

        max_z, max_y, max_x = image.shape
        if (z_neigh <= z <= max_z - z_neigh - 1
                and yx_neigh <= y <= max_y - yx_neigh - 1
                and yx_neigh <= x <= max_x - yx_neigh - 1):
            pass
        else:
            l_snr.append(np.nan)
            continue

        # extract local signal
        local_signal = image[z - z_radius: z + z_radius + 1,
                             y - yx_radius: y + yx_radius + 1,
                             x - yx_radius: x + yx_radius + 1].copy()

        # extract local noise
        local_noise = noise[z - z_neigh: z + z_neigh + 1,
                            y - yx_neigh: y + yx_neigh + 1,
                            x - yx_neigh: x + yx_neigh + 1].copy()
        local_noise[z_neigh - z_radius: z_neigh + z_radius + 1,
                    yx_neigh - yx_radius: yx_neigh + yx_radius + 1,
                    yx_neigh - yx_radius: yx_neigh + yx_radius + 1] = np.nan

        # compute snr
        snr = np.nanmean(local_signal) / np.nanstd(local_noise)
        l_snr.append(snr)

    return l_snr


# ### Utilities ###

def get_sigma(voxel_size_z=300, voxel_size_yx=103, psf_z=350, psf_yx=150):
    """Compute the standard deviation of the PSF of the spots.

    Parameters
    ----------
    voxel_size_z : float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : float
        Size of a voxel on the yx plan, in nanometer.
    psf_yx : int
        Theoretical size of the PSF emitted by a spot in
        the yx plan, in nanometer.
    psf_z : int
        Theoretical size of the PSF emitted by a spot in
        the z plan, in nanometer.

    Returns
    -------
    sigma_z : float
        Standard deviation of the PSF, along the z axis, in pixel.
    sigma_y : float
        Standard deviation of the PSF, along the y axis, in pixel.
    sigma_x : float
        Standard deviation of the PSF, along the x axis, in pixel.

    """
    # check parameters
    stack.check_parameter(voxel_size_z=int,
                          voxel_size_yx=int,
                          psf_z=int,
                          psf_yx=int)

    # compute sigma
    sigma_z = psf_z / voxel_size_z
    sigma_yx = psf_yx / voxel_size_yx

    return sigma_z, sigma_yx, sigma_yx


def get_radius(sigma, is_volume=True):
    """Approximate the radius of the detected spot.

    We use the formula:

        sqrt(ndim) * sigma

    with ndim the number of dimension of the image and sigma the standard
    deviation (in pixel) of the detected spot.

    Parameters
    ----------
    sigma : int, float or Tuple(float)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        scalar, the same sigma is applied to every dimensions. It approximates
        the standard deviation (in pixel) of the spots we want to detect.
    is_volume : bool
        Assume a 3-d or a 2-d spot.

    Returns
    -------
    radius : Tuple[float]
        Radius in pixels of the detected spots, one element per dimension.

    """
    # check parameters
    stack.check_parameter(sigma=(float, int, tuple),
                          is_volume=bool)

    # compute radius
    image_dim = 3 if is_volume else 2
    if isinstance(sigma, (int, float)):
        radius = np.sqrt(image_dim) * sigma
        radius = tuple(radius) * image_dim
    elif image_dim != len(sigma):
        raise ValueError("'sigma' should be a scalar or a tuple with one "
                         "value per dimension. Here the image has {0} "
                         "dimensions and sigma {1} elements."
                         .format(image_dim, len(sigma)))
    else:
        radius = [np.sqrt(image_dim) * sigma_ for sigma_ in sigma]
        radius = tuple(radius)

    return radius


