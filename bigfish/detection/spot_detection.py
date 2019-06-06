# -*- coding: utf-8 -*-

"""
Class and functions to detect RNA spots in 2-d and 3-d.
"""

from bigfish import stack

import scipy.ndimage as ndi
import numpy as np


# TODO complete documentation methods
# TODO add sanity check functions

# ### Spot detection ###

def log_lm(image, sigma, threshold, minimum_distance=1, return_log=False):
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
    threshold : float or int
        A threshold to detect peaks. Considered as a relative threshold if
        float.
    minimum_distance : int
        Minimum distance (in number of pixels) between two local peaks.
    return_log : bool
        Return the LoG filtered image.

    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2)
        for 3-d or 2-d images respectively.
    radius : float, Tuple[float]
        Radius of the detected peaks.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_parameter(sigma=(float, int, tuple),
                          minimum_distance=(float, int),
                          threshold=(float, int))

    # cast image in np.float and apply LoG filter
    image_filtered = stack.log_filter(image, sigma, keep_dtype=True)

    # find local maximum
    mask = local_maximum_detection(image_filtered, minimum_distance)

    # remove spots with a low intensity and return coordinates and radius
    spots, radius = spots_thresholding(image, sigma, mask, threshold)

    if return_log:
        return spots, radius, image_filtered

    else:
        return spots, radius


def local_maximum_detection(image, minimum_distance):
    """Compute a mask to keep only local maximum, in 2-d and 3-d.

    1) We apply a multidimensional maximum filter.
    2) A pixel which has the same value in the original and filtered images
    is a local maximum.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image to process with shape (z, y, x) or (y, x).
    minimum_distance : int, float
        Minimum distance (in number of pixels) between two local peaks.

    Returns
    -------
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_parameter(minimum_distance=(float, int))

    # compute the kernel size (centered around our pixel because it is uneven)
    kernel_size = int(2 * minimum_distance + 1)

    # apply maximum filter to the original image
    image_filtered = ndi.maximum_filter(image, size=kernel_size)

    # we keep the pixels with the same value before and after the filtering
    mask = image == image_filtered

    return mask


def spots_thresholding(image, sigma, mask, threshold):
    """Filter detected spots and get coordinates of the remaining
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
    radius : float or Tuple(float)
        Radius of the detected peaks.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_array(mask,
                      ndim=[2, 3],
                      dtype=[bool],
                      allow_nan=False)
    stack.check_parameter(sigma=(float, int, tuple),
                          threshold=(float, int))

    # remove peak with a low intensity
    if isinstance(threshold, float):
        threshold *= image.max()
    mask_ = (mask & (image > threshold))

    # get peak coordinates
    peak_coordinates = np.nonzero(mask_)
    peak_coordinates = np.column_stack(peak_coordinates)

    # compute radius
    if isinstance(sigma, tuple):
        radius = [np.sqrt(image.ndim) * sigma_ for sigma_ in sigma]
        radius = tuple(radius)
    else:
        radius = np.sqrt(image.ndim) * sigma

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


# ### Utils ###

def get_sigma(resolution_z=300, resolution_yx=103, psf_z=400, psf_yx=200):
    """Compute the standard deviation of the PSF of the spots.

    Parameters
    ----------
    resolution_z : float
        Height of a voxel, along the z axis, in nanometer.
    resolution_yx : float
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
    sigma_xy : float
        Standard deviation of the PSF, along the yx plan, in pixel.
    """
    # compute sigma
    sigma_z = psf_z / resolution_z
    sigma_yx = psf_yx / resolution_yx

    return sigma_z, sigma_yx


def build_reference_spot(image, spots, radius, method="median"):
    """Build a

    Parameters
    ----------
    image : np.ndarray,
        Image with shape (z, y, x) or (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2)
        for 3-d or 2-d images respectively.
    radius : Tuple[float]
        Radius of the detected peaks, one for each dimension.
    method : str
        Method use to compute the reference spot (a 'mean' or 'median' spot).

    Returns
    -------
    reference_spot : np.ndarray
        Reference spot with shape (2*radius_z+1, 2*radius_y+1, 2*radius_x+1) or
        (2*radius_y+1, 2*radius_x+1).

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64],
                      allow_nan=False)
    stack.check_array(spots,
                      ndim=2,
                      dtype=[np.int64],
                      allow_nan=False)
    stack.check_parameter(radius=(float, int, tuple),
                          method=str)
    if method not in ['mean', 'median']:
        raise ValueError("'{0}' is not a valid value for parameter 'method'. "
                         "Use 'mean' or 'median' instead.".format(method))

    # process a 3-d image
    if image.ndim == 3:
        # get a rounded radius for each dimension
        radius_z = int(radius[0]) + 1
        radius_yx = int(radius[1]) + 1
        z_shape = radius_z * 2 + 1
        yx_shape = radius_yx * 2 + 1

        # collect area around each spot
        volume_spot = []
        for i_spot in range(spots.shape[0]):

            # get spot coordinates
            spot_z, spot_y, spot_x = spots[i_spot, :]

            # get boundaries of the volume surrounding the spot
            z_spot_min = max(0, int(spot_z - radius_z))
            z_spot_max = min(image.shape[0], int(spot_z + radius_z))
            y_spot_min = max(0, int(spot_y - radius_yx))
            y_spot_max = min(image.shape[1], int(spot_y + radius_yx))
            x_spot_min = max(0, int(spot_x - radius_yx))
            x_spot_max = min(image.shape[2], int(spot_x + radius_yx))

            # get the volume of the spot
            image_spot = image[z_spot_min:z_spot_max + 1,
                               y_spot_min:y_spot_max + 1,
                               x_spot_min:x_spot_max + 1]

            # remove the cropped images
            if image_spot.shape != (z_shape, yx_shape, yx_shape):
                continue

            volume_spot.append(image_spot)

    # process a 2-d image
    else:
        # get a rounded radius for each dimension
        radius_yx = int(radius[1]) + 1
        yx_shape = radius_yx * 2 + 1

        # collect area around each spot
        volume_spot = []
        for i_spot in range(spots.shape[0]):

            # get spot coordinates
            spot_y, spot_x = spots[i_spot, :]

            # get boundaries of the volume surrounding the spot
            y_spot_min = max(0, int(spot_y - radius_yx))
            y_spot_max = min(image.shape[1], int(spot_y + radius_yx))
            x_spot_min = max(0, int(spot_x - radius_yx))
            x_spot_max = min(image.shape[2], int(spot_x + radius_yx))

            # get the volume of the spot
            image_spot = image[y_spot_min:y_spot_max + 1,
                               x_spot_min:x_spot_max + 1]

            # remove the cropped images
            if image_spot.shape != (yx_shape, yx_shape):
                continue

            volume_spot.append(image_spot)

    # if no spot where detected
    if len(volume_spot) == 0:
        return None

    # project the different spot images
    volume_spot = np.stack(volume_spot, axis=0)
    if method == "mean":
        reference_spot = np.mean(volume_spot, axis=0)
    else:
        reference_spot = np.median(volume_spot, axis=0)

    return reference_spot
