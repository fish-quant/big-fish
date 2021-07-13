# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to detect spots in 2-d and 3-d.
"""

import warnings
import scipy.ndimage as ndi
import numpy as np

import bigfish.stack as stack

from skimage.measure import regionprops
from skimage.measure import label


# ### Main function ###

def detect_spots(images, threshold=None, remove_duplicate=True,
                 return_threshold=False, voxel_size_z=None, voxel_size_yx=100,
                 psf_z=None, psf_yx=200):
    """Apply LoG filter followed by a Local Maximum algorithm to detect spots
    in a 2-d or 3-d image.

    #. We smooth the image with a LoG filter.
    #. We apply a multidimensional maximum filter.
    #. A pixel which has the same value in the original and filtered images
       is a local maximum.
    #. We remove local peaks under a threshold.
    #. We keep only one pixel coordinate per detected spot.

    Parameters
    ----------
    images : List[np.ndarray] or np.ndarray
        Image (or list of images) with shape (z, y, x) or (y, x). If several
        images are provided, the same threshold is applied.
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs. If None,
        optimal threshold is selected automatically. If several images are
        provided, one optimal threshold is selected for all the images.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.
    return_threshold : bool
        Return the threshold used to detect spots.
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, image is
        considered in 2-d.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer. If None, image is considered in 2-d.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.

    Returns
    -------
    spots : List[np.ndarray] or np.ndarray, np.int64
        Coordinates (or list of coordinates) of the spots with shape
        (nb_spots, 3) or (nb_spots, 2), for 3-d or 2-d images respectively.
    threshold : int or float
        Threshold used to discriminate spots from noisy blobs.

    """
    # check parameters
    stack.check_parameter(threshold=(float, int, type(None)),
                          remove_duplicate=bool,
                          return_threshold=bool,
                          voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float))

    # if one image is provided we enlist it
    if not isinstance(images, list):
        stack.check_array(images,
                          ndim=[2, 3],
                          dtype=[np.uint8, np.uint16,
                                 np.float32, np.float64])
        ndim = images.ndim
        images = [images]
        is_list = False
    else:
        ndim = None
        for i, image in enumerate(images):
            stack.check_array(image,
                              ndim=[2, 3],
                              dtype=[np.uint8, np.uint16,
                                     np.float32, np.float64])
            if i == 0:
                ndim = image.ndim
            else:
                if ndim != image.ndim:
                    raise ValueError("Provided images should have the same "
                                     "number of dimensions.")
        is_list = True

    # check consistency between parameters
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided images has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and psf_z is None:
        raise ValueError("Provided images has {0} dimensions but "
                         "'psf_z' parameter is missing.".format(ndim))
    if ndim == 2:
        voxel_size_z = None
        psf_z = None

    # detect spots
    if return_threshold:
        spots, threshold = _detect_spots_from_images(
            images,
            threshold=threshold,
            remove_duplicate=remove_duplicate,
            return_threshold=return_threshold,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            psf_z=psf_z,
            psf_yx=psf_yx)
    else:
        spots = _detect_spots_from_images(
            images,
            threshold=threshold,
            remove_duplicate=remove_duplicate,
            return_threshold=return_threshold,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            psf_z=psf_z,
            psf_yx=psf_yx)

    # format results
    if not is_list:
        spots = spots[0]

    # return threshold or not
    if return_threshold:
        return spots, threshold
    else:
        return spots


def _detect_spots_from_images(images, threshold=None, remove_duplicate=True,
                              return_threshold=False, voxel_size_z=None,
                              voxel_size_yx=100, psf_z=None, psf_yx=200):
    """Apply LoG filter followed by a Local Maximum algorithm to detect spots
    in a 2-d or 3-d image.

    #. We smooth the image with a LoG filter.
    #. We apply a multidimensional maximum filter.
    #. A pixel which has the same value in the original and filtered images
       is a local maximum.
    #. We remove local peaks under a threshold.
    #. We keep only one pixel coordinate per detected spot.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images with shape (z, y, x) or (y, x). The same threshold is
        applied to every images.
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs. If None,
        optimal threshold is selected automatically. If several images are
        provided, one optimal threshold is selected for all the images.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.
    return_threshold : bool
        Return the threshold used to detect spots.
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, image is
        considered in 2-d.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan, in
        nanometer. If None, image is considered in 2-d.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan, in
        nanometer.

    Returns
    -------
    all_spots : List[np.ndarray], np.int64
        List of spot coordinates with shape (nb_spots, 3) or (nb_spots, 2),
        for 3-d or 2-d images respectively.
    threshold : int or float
        Threshold used to discriminate spots from noisy blobs.

    """
    # initialization
    sigma = stack.get_sigma(voxel_size_z, voxel_size_yx, psf_z, psf_yx)
    n = len(images)

    # apply LoG filter and find local maximum
    images_filtered = []
    pixel_values = []
    masks = []
    for image in images:
        # filter image
        image_filtered = stack.log_filter(image, sigma)
        images_filtered.append(image_filtered)

        # get pixels value
        pixel_values += list(image_filtered.ravel())

        # find local maximum
        mask_local_max = local_maximum_detection(image_filtered, sigma)
        masks.append(mask_local_max)

    # get optimal threshold if necessary based on all the images
    if threshold is None:

        # get threshold values we want to test
        thresholds = _get_candidate_thresholds(pixel_values)

        # get spots count and its logarithm
        all_value_spots = []
        minimum_threshold = float(thresholds[0])
        for i in range(n):
            image_filtered = images_filtered[i]
            mask_local_max = masks[i]
            spots, mask_spots = spots_thresholding(
                image_filtered, mask_local_max,
                threshold=minimum_threshold,
                remove_duplicate=False)
            value_spots = image_filtered[mask_spots]
            all_value_spots.append(value_spots)
        all_value_spots = np.concatenate(all_value_spots)
        thresholds, count_spots = _get_spot_counts(thresholds, all_value_spots)

        # select threshold where the kink of the distribution is located
        if count_spots.size > 0:
            threshold, _, _ = _get_breaking_point(thresholds, count_spots)

    # detect spots
    all_spots = []
    for i in range(n):

        # get images and masks
        image_filtered = images_filtered[i]
        mask_local_max = masks[i]

        # detection
        spots, _ = spots_thresholding(image_filtered, mask_local_max,
                                      threshold, remove_duplicate)
        all_spots.append(spots)

    # return threshold or not
    if return_threshold:
        return all_spots, threshold
    else:
        return all_spots


# ### LoG spot detection ###

def local_maximum_detection(image, min_distance):
    """Compute a mask to keep only local maximum, in 2-d and 3-d.

    #. We apply a multidimensional maximum filter.
    #. A pixel which has the same value in the original and filtered images
       is a local maximum.

    Several connected pixels can have the same value. In such a case, the
    local maximum is not unique.

    In order to make the detection robust, it should be applied to a
    filtered image (using :func:`bigfish.stack.log_filter` for example).

    Parameters
    ----------
    image : np.ndarray
        Image to process with shape (z, y, x) or (y, x).
    min_distance : int, float or Tuple(float)
        Minimum distance (in pixels) between two spots we want to be able to
        detect separately. One value per spatial dimension (zyx or yx
        dimensions). If it's a scalar, the same distance is applied to every
        dimensions.

    Returns
    -------
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_parameter(min_distance=(float, int, tuple))

    # compute the kernel size (centered around our pixel because it is uneven)
    if isinstance(min_distance, (int, float)):
        min_distance = (min_distance,) * image.ndim
        min_distance = np.ceil(min_distance).astype(image.dtype)
    elif image.ndim != len(min_distance):
        raise ValueError("'min_distance' should be a scalar or a tuple with "
                         "one value per dimension. Here the image has {0} "
                         "dimensions and 'min_distance' {1} elements."
                         .format(image.ndim, len(min_distance)))
    else:
        min_distance = np.ceil(min_distance).astype(image.dtype)
    kernel_size = 2 * min_distance + 1

    # apply maximum filter to the original image
    image_filtered = ndi.maximum_filter(image, size=kernel_size)

    # we keep the pixels with the same value before and after the filtering
    mask = image == image_filtered

    return mask


def spots_thresholding(image, mask_local_max, threshold,
                       remove_duplicate=True):
    """Filter detected spots and get coordinates of the remaining spots.

    In order to make the thresholding robust, it should be applied to a
    filtered image (using :func:`bigfish.stack.log_filter`
    for example). If the local maximum is not unique (it can happen if connected
    pixels have the same value), a connected component algorithm is applied to
    keep only one coordinate per spot.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    threshold : float, int or None
        A threshold to discriminate relevant spots from noisy blobs. If None,
        detection is aborted with a warning.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.

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
    stack.check_parameter(threshold=(float, int, type(None)),
                          remove_duplicate=bool)

    if threshold is None:
        mask = np.zeros_like(image, dtype=bool)
        spots = np.array([], dtype=np.int64).reshape((0, image.ndim))
        warnings.warn("No spots were detected (threshold is {0})."
                      .format(threshold),
                      UserWarning)
        return spots, mask

    # remove peak with a low intensity
    mask = (mask_local_max & (image > threshold))
    if mask.sum() == 0:
        spots = np.array([], dtype=np.int64).reshape((0, image.ndim))
        return spots, mask

    # make sure we detect only one coordinate per spot
    if remove_duplicate:
        # when several pixels are assigned to the same spot, keep the centroid
        cc = label(mask)
        local_max_regions = regionprops(cc)
        spots = []
        for local_max_region in local_max_regions:
            spot = np.array(local_max_region.centroid)
            spots.append(spot)
        spots = np.stack(spots).astype(np.int64)

        # built mask again
        mask = np.zeros_like(mask)
        mask[spots[:, 0], spots[:, 1]] = True

    else:
        # get peak coordinates
        spots = np.nonzero(mask)
        spots = np.column_stack(spots)

    # case where no spots were detected
    if spots.size == 0:
        warnings.warn("No spots were detected (threshold is {0})."
                      .format(threshold),
                      UserWarning)

    return spots, mask


# ### Threshold selection ###

def automated_threshold_setting(image, mask_local_max):
    """Automatically set the optimal threshold to detect spots.

    In order to make the thresholding robust, it should be applied to a
    filtered image (using :func:`bigfish.stack.log_filter` for example). The
    optimal threshold is selected based on the spots distribution. The latter
    should have an elbow curve discriminating a fast decreasing stage from a
    more stable one (a plateau).

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.

    Returns
    -------
    optimal_threshold : int
        Optimal threshold to discriminate spots from noisy blobs.

    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(mask_local_max,
                      ndim=[2, 3],
                      dtype=[bool])

    # get threshold values we want to test
    thresholds = _get_candidate_thresholds(image.ravel())

    # get spots count and its logarithm
    first_threshold = float(thresholds[0])
    spots, mask_spots = spots_thresholding(
        image, mask_local_max, first_threshold, remove_duplicate=False)
    value_spots = image[mask_spots]
    thresholds, count_spots = _get_spot_counts(thresholds, value_spots)

    # select threshold where the break of the distribution is located
    if count_spots.size > 0:
        optimal_threshold, _, _ = _get_breaking_point(thresholds, count_spots)

    # case where no spots were detected
    else:
        optimal_threshold = None

    return optimal_threshold


def _get_candidate_thresholds(pixel_values):
    """Choose the candidate thresholds to test for the spot detection.

    Parameters
    ----------
    pixel_values : np.ndarray
        Pixel intensity values of the image.

    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.

    """
    # choose appropriate thresholds candidate
    start_range = 0
    end_range = int(np.percentile(pixel_values, 99.9999))
    if end_range < 100:
        thresholds = np.linspace(start_range, end_range, num=100)
    else:
        thresholds = [i for i in range(start_range, end_range + 1)]
    thresholds = np.array(thresholds)

    return thresholds


def _get_spot_counts(thresholds, value_spots):
    """Compute and format the spots count function for different thresholds.

    Parameters
    ----------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    value_spots : np.ndarray
        Pixel intensity values of all spots.

    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    count_spots : np.ndarray, np.float64
        Spots count function.

    """
    # count spots for each threshold
    count_spots = np.log([np.count_nonzero(value_spots > t)
                          for t in thresholds])
    count_spots = stack.centered_moving_average(count_spots, n=5)

    # the tail of the curve unnecessarily flatten the slop
    count_spots = count_spots[count_spots > 2]
    thresholds = thresholds[:count_spots.size]

    return thresholds, count_spots


def _get_breaking_point(x, y):
    """Select the x-axis value where a L-curve has a kink.

    Assuming a L-curve from A to B, the 'breaking_point' is the more distant
    point to the segment [A, B].

    Parameters
    ----------
    x : np.array, np.float64
        X-axis values.
    y : np.array, np.float64
        Y-axis values.

    Returns
    -------
    breaking_point : float
        X-axis value at the kink location.
    x : np.array, np.float64
        X-axis values.
    y : np.array, np.float64
        Y-axis values.

    """
    # select threshold where curve break
    slope = (y[-1] - y[0]) / len(y)
    y_grad = np.gradient(y)
    m = list(y_grad >= slope)
    j = m.index(False)
    m = m[j:]
    x = x[j:]
    y = y[j:]
    if True in m:
        i = m.index(True)
    else:
        i = -1
    breaking_point = float(x[i])

    return breaking_point, x, y


def get_elbow_values(images, voxel_size_z=None, voxel_size_yx=100, psf_z=None,
                     psf_yx=200):
    """Get values to plot the elbow curve used to automatically set the
    threshold.

    Parameters
    ----------
    images : List[np.ndarray] or np.ndarray
        Image (or list of images) with shape (z, y, x) or (y, x). If several
        images are provided, the same threshold is applied.
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, image is
        considered in 2-d.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer. If None, image is considered in 2-d.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.

    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    count_spots : np.ndarray, np.float64
        Spots count function.
    threshold : float or None
        Threshold automatically set.

    """
    # check parameters
    stack.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float))

    # if one image is provided we enlist it
    if not isinstance(images, list):
        stack.check_array(images,
                          ndim=[2, 3],
                          dtype=[np.uint8, np.uint16,
                                 np.float32, np.float64])
        ndim = images.ndim
        images = [images]
        n = 1
    else:
        ndim = None
        for i, image in enumerate(images):
            stack.check_array(image,
                              ndim=[2, 3],
                              dtype=[np.uint8, np.uint16,
                                     np.float32, np.float64])
            if i == 0:
                ndim = image.ndim
            else:
                if ndim != image.ndim:
                    raise ValueError("Provided images should have the same "
                                     "number of dimensions.")
        n = len(images)

    # check consistency between parameters
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided images has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and psf_z is None:
        raise ValueError("Provided images has {0} dimensions but "
                         "'psf_z' parameter is missing.".format(ndim))
    if ndim == 2:
        voxel_size_z = None
        psf_z = None

    # compute sigma
    sigma = stack.get_sigma(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

    # apply LoG filter and find local maximum
    images_filtered = []
    pixel_values = []
    masks = []
    for image in images:
        # filter image
        image_filtered = stack.log_filter(image, sigma)
        images_filtered.append(image_filtered)

        # get pixels value
        pixel_values += list(image_filtered.ravel())

        # find local maximum
        mask_local_max = local_maximum_detection(
            image_filtered, sigma)
        masks.append(mask_local_max)

    # get threshold values we want to test
    thresholds = _get_candidate_thresholds(pixel_values)

    # get spots count and its logarithm
    all_value_spots = []
    minimum_threshold = float(thresholds[0])
    for i in range(n):
        image_filtered = images_filtered[i]
        mask_local_max = masks[i]
        spots, mask_spots = spots_thresholding(
            image_filtered, mask_local_max,
            threshold=minimum_threshold,
            remove_duplicate=False)
        value_spots = image_filtered[mask_spots]
        all_value_spots.append(value_spots)
    all_value_spots = np.concatenate(all_value_spots)
    thresholds, count_spots = _get_spot_counts(
        thresholds, all_value_spots)

    # select threshold where the kink of the distribution is located
    if count_spots.size > 0:
        threshold, _, _ = _get_breaking_point(thresholds, count_spots)
    else:
        threshold = None

    return thresholds, count_spots, threshold
