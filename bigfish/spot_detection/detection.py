# -*- coding: utf-8 -*-

"""
Class and functions to detect RNA spots in 2-d and 3-d.
"""

import scipy.ndimage as ndi
import numpy as np

from bigfish import stack


def detection(tensor, r, c, detection_method, **kargs):
    """

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
    # get the smfish image
    image = tensor[r, c, :, :, :]

    # apply spot detection
    peak_coordinates, radius = None, None
    if detection_method == "log_lm":
        peak_coordinates, radius = log_lm(image, **kargs)

    return peak_coordinates, radius


def log_lm(image, sigma, minimum_distance=1, threshold=None):
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
    # cast image in np.float and apply LoG filter
    image_filtered = stack.log_filter(image, sigma)

    # find local maximum
    mask = _non_maximum_suppression_mask(image_filtered, minimum_distance)

    # remove peak with a low intensity
    if isinstance(threshold, float):
        threshold *= image.max()
    mask &= image > threshold

    # get peak coordinates and radius
    peak_coordinates = np.nonzero(mask)
    peak_coordinates = np.column_stack(peak_coordinates)
    radius = np.sqrt(image.ndim) * sigma[-1]

    return peak_coordinates, radius


def local_maximum_detection(image, minimum_distance=1, threshold=0.2):
    """Find local maximum in a 2-d or 3-d image.

    1) We apply a multidimensional maximum filter.
    2) A pixel which has the same value in the original and filtered images
    is a local maximum.
    3) We remove local peaks under a threshold.

    Parameters
    ----------
    image : np.ndarray, np.float
        Image to process with shape (z, y, x) or (y, x).
    minimum_distance : int
        Minimum distance (in number of pixels) between two local peaks.
    threshold : float or int
        A threshold to detect peaks. Considered as a relative threshold if
        float.

    Returns
    -------
    peak_coordinate : np.ndarray, np.int64
        Coordinate of the local peaks with shape (nb_peaks, 3) or
        (nb_peaks, 2).
    """
    mask = _non_maximum_suppression_mask(image, minimum_distance)

    if isinstance(threshold, float):
        threshold *= image.max()
    mask &= image > threshold

    peak_coordinate = np.nonzero(mask)
    peak_coordinate = np.column_stack(peak_coordinate)

    return peak_coordinate


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


def optimize_threshold_log_lm(tensor, sigma, thresholds,
                              r=0, c=2, minimum_distance=1, verbose=False):
    # get the smfish image
    image = tensor[r, c, :, :, :]

    # cast image in np.float and apply LoG filter
    image_filtered = stack.log_filter(image, sigma)

    # find local maximum
    mask = _non_maximum_suppression_mask(image_filtered, minimum_distance)
    if verbose:
        print("{0} local peaks detected.".format(mask.sum()))

    # test different thresholds
    peak_coordinates = []
    for threshold in thresholds:
        if isinstance(threshold, float):
            threshold *= image.max()
        mask_ = (mask & (image > threshold))

        # get peak coordinates
        peak_coordinates_ = np.nonzero(mask_)
        peak_coordinates_ = np.column_stack(peak_coordinates_)
        peak_coordinates.append(peak_coordinates_)

        if verbose:
            print("Threshold {0}: {1} RNA detected."
                  .format(threshold, peak_coordinates_.shape[0]))

        # early stop if we detect zero rna
        if peak_coordinates_.shape[0] == 0:
            break

    # reshape threshold
    thresholds = thresholds[:len(peak_coordinates)]

    # get radius
    radius = np.sqrt(image.ndim) * sigma[-1]

    return peak_coordinates, thresholds, radius


def compute_snr(image, threshold_signal_detection=0.5, neighbor_size=None):
    # TODO add documentation
    # TODO keep only local snr
    # TODO improve local snr with a mean of computed local snr and not a global
    #  snr computed with local noise.
    mask = _non_maximum_suppression_mask(image, minimum_distance=1)

    if isinstance(threshold_signal_detection, float):
        threshold_signal_detection *= image.max()
    mask &= image > threshold_signal_detection

    signal = image.astype(np.float64)
    signal[~mask] = np.nan

    noise = image.astype(np.float64)
    noise[mask] = np.nan

    # global SNR
    snr_1 = np.nanmean(signal) / np.nanstd(noise)
    snr_2 = np.nanmean(signal) / np.nanstd(signal)

    # local SNR
    if neighbor_size is not None:
        mask_filtered = ndi.maximum_filter(mask,
                                           size=neighbor_size,
                                           mode='constant')

        mask_local = mask_filtered & ~mask
        noise_local = image.astype(np.float64)
        noise_local[mask_local] = np.nan

        snr_local = np.nanmean(signal) / np.nanstd(noise_local)

        return snr_1, snr_2, snr_local

    else:
        return snr_1, snr_2
