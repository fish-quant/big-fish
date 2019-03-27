# -*- coding: utf-8 -*-

"""
Class and functions to detect RNA spots in 2-d and 3-d.
"""

from bigfish import stack

import scipy.ndimage as ndi
import numpy as np


# TODO complete documentation methods

# ### Spot detection ###

def detection(tensor, r, c, detection_method, **kwargs):
    """Apply spot detection.

    Parameters
    ----------
    tensor : nd.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).
    r : int
        Round index to process.
    c : int
        Channel index of the smfish image.
    detection_method : str
        Method used to detect spots.

    Returns
    -------
    peak_coordinates : np.ndarray, np.int64
        Coordinate of the local peaks with shape (nb_peaks, 3) or
        (nb_peaks, 2) for 3-d or 2-d images respectively.
    radius : float
        Radius of the detected peaks.

    """
    # check tensor dimensions and its dtype
    stack.check_array(tensor, ndim=5, dtype=[np.uint8, np.uint16])

    # get the smfish image
    image = tensor[r, c, :, :, :]

    # apply spot detection
    peak_coordinates, radius = None, None
    if detection_method == "log_lm":
        peak_coordinates, radius = detection_log_lm(image, **kwargs)

    return peak_coordinates, radius


def detection_log_lm(image, sigma, minimum_distance=1, threshold=None):
    """Apply LoG filter followed by a Local Maximum algorithm to detect spots
    in a 2-d or 3-d image.

    1) We smooth the image with a LoG filter.
    2) We apply a multidimensional maximum filter.
    3) A pixel which has the same value in the original and filtered images
    is a local maximum.
    4) We remove local peaks under a threshold.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    sigma : float or Tuple(float)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        float, the same sigma is applied to every dimensions.
    minimum_distance : int
        Minimum distance (in number of pixels) between two local peaks.
    threshold : float or int
        A threshold to detect peaks. Considered as a relative threshold if
        float.

    Returns
    -------
    peak_coordinates : np.ndarray, np.int64
        Coordinate of the local peaks with shape (nb_peaks, 3) or
        (nb_peaks, 2) for 3-d or 2-d images respectively.
    radius : float
        Radius of the detected peaks.

    """
    # cast image in np.float, apply LoG filter and find local maximum
    mask = _log_lm(image, sigma, minimum_distance)

    # remove peak with a low intensity and return coordinates and radius
    peak_coordinates, radius = _from_threshold_to_spots(image, sigma, mask,
                                                        threshold)

    return peak_coordinates, radius


def _log_lm(image, sigma, minimum_distance=1):
    """Find local maximum in a 2-d or 3-d image.

    1) We smooth the image with a LoG filter.
    2) We apply a multidimensional maximum filter.
    3) A pixel which has the same value in the original and filtered images
    is a local maximum.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image to process with shape (z, y, x) or (y, x).
    sigma : float or Tuple(float)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        float, the same sigma is applied to every dimensions.
    minimum_distance : int
        Minimum distance (in number of pixels) between two local peaks.

    Returns
    -------
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    """
    # cast image in np.float and apply LoG filter
    image_filtered = stack.log_filter(image, sigma)

    # find local maximum
    mask = _non_maximum_suppression_mask(image_filtered, minimum_distance)

    return mask


def _non_maximum_suppression_mask(image, minimum_distance):
    """Compute a mask to keep only local maximum, in 2-d and 3-d.

    1) We apply a multidimensional maximum filter.
    2) A pixel which has the same value in the original and filtered images
    is a local maximum.

    Parameters
    ----------
    image : np.ndarray, np.float
        Image to process with shape (z, y, x) or (y, x).
    minimum_distance : int
        Minimum distance (in number of pixels) between two local peaks.

    Returns
    -------
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    """
    # compute the kernel size (centered around our pixel because it is uneven
    kernel_size = 2 * minimum_distance + 1

    # apply maximum filter to the original image
    image_filtered = ndi.maximum_filter(image, size=kernel_size,
                                        mode='constant')

    # we keep the pixels with the same value before and after the filtering
    mask = image == image_filtered

    return mask


def _from_threshold_to_spots(image, sigma, mask, threshold):
    """Filter detected local maximum and get coordinates of the remaining
    spots.

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

    Returns
    -------
    peak_coordinates : np.ndarray, np.int64
        Coordinate of the local peaks with shape (nb_peaks, 3) or
        (nb_peaks, 2) for 3-d or 2-d images respectively.
    radius : float
        Radius of the detected peaks.

    """
    # remove peak with a low intensity
    if isinstance(threshold, float):
        threshold *= image.max()
    mask_ = (mask & (image > threshold))

    # get peak coordinates and radius
    peak_coordinates = np.nonzero(mask_)
    peak_coordinates = np.column_stack(peak_coordinates)
    radius = np.sqrt(image.ndim) * sigma[-1]

    return peak_coordinates, radius


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
    mask = _log_lm(image, sigma, minimum_distance)

    # apply a specific threshold to filter the detected spots and compute snr
    l_snr = _from_threshold_to_snr(image, sigma, mask,
                                   threshold_signal_detection,
                                   neighbor_factor)

    return l_snr


def _from_threshold_to_snr(image, sigma, mask, threshold=2000,
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


# ### Utils ###

def get_sigma(resolution_xy=103, resolution_z=300, psf_xy=200, psf_z=400):
    """Compute the optimal sigma to use gaussian models with spots.

    Parameters
    ----------
    resolution_xy : int
        Distance, in nanometer, between two pixels along the XY dimension.
    resolution_z : int
        Distance, in nanometer, between two pixels along the Z dimension.

    psf_xy : int
        Theoretical size (in nanometer) of the signal emitted by a spot in
        the XY plan.
    psf_z : int
        Theoretical size (in nanometer) of the signal emitted by a spot in
        the Z plan.

    Returns
    -------
    sigma : Tuple
        A Tuple with 3 items corresponding to the sigma used by a gaussian
        filter in each direction of the image (approximately the same size of
        the spot in the image).

    """
    # compute sigma
    sigma_xy = psf_xy / resolution_xy
    sigma_z = psf_z / resolution_z
    sigma = (sigma_z, sigma_xy, sigma_xy)

    return sigma
