# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Class and functions to detect RNA spots in 2-d and 3-d.
"""

import bigfish.stack as stack

import scipy.ndimage as ndi
import numpy as np


# ### LoG detection ###

def spot_detector(image, sigma, threshold=2):
    """Apply LoG filter followed by a Local Maximum algorithm to detect spots
    in a 2-d or 3-d image.

    1) We smooth the image with a LoG filter.
    2) We apply a multidimensional maximum filter.
    3) A pixel which has the same value in the original and filtered images
    is a local maximum.
    4) We remove local peaks under a threshold.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    sigma : int, float or Tuple(float)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        scalar, the same sigma is applied to every dimensions. It approximates
        the standard deviation (in pixel) of the spots we want to detect.
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs.

    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2)
        for 3-d or 2-d images respectively.

    """
    # check parameters
    stack.check_parameter(threshold=(float, int))

    # cast image in np.float and apply LoG filter
    image_filtered = stack.log_filter(image, sigma)

    # find local maximum
    mask_local_max = local_maximum_detection(image_filtered, sigma)

    # remove spots with a low intensity and return their coordinates
    spots, _ = spots_thresholding(image_filtered, mask_local_max, threshold)

    return spots


def local_maximum_detection(image, sigma):
    """Compute a mask to keep only local maximum, in 2-d and 3-d.

    1) We apply a multidimensional maximum filter.
    2) A pixel which has the same value in the original and filtered images
    is a local maximum.

    Parameters
    ----------
    image : np.ndarray
        Image to process with shape (z, y, x) or (y, x).
    sigma : int, float or Tuple(float)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        scalar, the same sigma is applied to every dimensions. It approximates
        the standard deviation (in pixel) of the spots we want to detect.

    Returns
    -------
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_parameter(sigma=(float, int, tuple))

    # compute the kernel size (centered around our pixel because it is uneven)
    if isinstance(sigma, (int, float)):
        sigma = tuple(sigma) * image.ndim
        sigma = np.round(sigma).astype(image.dtype)
    elif image.ndim != len(sigma):
        raise ValueError("'sigma' should be a scalar or a tuple with one "
                         "value per dimension. Here the image has {0} "
                         "dimensions and sigma {1} elements."
                         .format(image.ndim, len(sigma)))
    else:
        sigma = np.round(sigma).astype(image.dtype)
    kernel_size = 2 * sigma + 1

    # apply maximum filter to the original image
    image_filtered = ndi.maximum_filter(image, size=kernel_size)

    # we keep the pixels with the same value before and after the filtering
    mask = image == image_filtered

    return mask


def spots_thresholding(image, mask_local_max, threshold):
    """Filter detected spots and get coordinates of the remaining spots.

    In order to make the thresholding robust, it should be applied to a
    denoised image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs.

    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the local peaks with shape (nb_peaks, 3) or
        (nb_peaks, 2) for 3-d or 2-d images respectively.
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the spots.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(mask_local_max,
                      ndim=[2, 3],
                      dtype=[bool])
    stack.check_parameter(threshold=(float, int))

    # remove peak with a low intensity
    mask = (mask_local_max & (image > threshold))

    # get peak coordinates
    spots = np.nonzero(mask)
    spots = np.column_stack(spots)

    return spots, mask

