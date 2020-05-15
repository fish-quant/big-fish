# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.stack subpackage.
"""

import bigfish.stack as stack
import numpy as np


# TODO add function to calibrate psf
# TODO add function to compute signal-to-noise ratio

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

def get_sigma(voxel_size_z=None, voxel_size_yx=100, psf_z=None, psf_yx=200):
    """Compute the standard deviation of the PSF of the spots.

    Parameters
    ----------
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.

    Returns
    -------
    sigma : Tuple[float]
        Standard deviations in pixel of the PSF, one element per dimension.

    """
    # check parameters
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float))

    # compute sigma
    sigma_yx = psf_yx / voxel_size_yx

    if voxel_size_z is None or psf_z is None:
        return sigma_yx, sigma_yx

    else:
        sigma_z = psf_z / voxel_size_z
        return sigma_z, sigma_yx, sigma_yx


def get_radius(voxel_size_z=None, voxel_size_yx=100, psf_z=None, psf_yx=200):
    """Approximate the radius of the detected spot.

    We use the formula:

        sqrt(ndim) * sigma

    with ndim the number of dimension of the image and sigma the standard
    deviation (in pixel) of the detected spot.

    Parameters
    ----------
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.

    Returns
    -------
    radius : Tuple[float]
        Radius in pixels of the detected spots, one element per dimension.

    """
    # compute sigma
    sigma = get_sigma(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

    # compute radius
    radius = [np.sqrt(len(sigma)) * sigma_ for sigma_ in sigma]
    radius = tuple(radius)

    return radius


