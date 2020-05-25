# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Class and functions to detect RNA spots in 2-d and 3-d.
"""

import scipy.ndimage as ndi
import numpy as np

import bigfish.stack as stack
from .utils import get_sigma

from skimage.measure import regionprops
from skimage.measure import label


# ### LoG detection ###

def detect_spots(image, threshold, remove_duplicate=True, voxel_size_z=None,
                 voxel_size_yx=100, psf_z=None, psf_yx=200):
    """Apply LoG filter followed by a Local Maximum algorithm to detect spots
    in a 2-d or 3-d image.

    1) We smooth the image with a LoG filter.
    2) We apply a multidimensional maximum filter.
    3) A pixel which has the same value in the original and filtered images
    is a local maximum.
    4) We remove local peaks under a threshold.
    5) We keep only one pixel coordinate per detected spot.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.
    psf_z : int or float
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer.

    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2)
        for 3-d or 2-d images respectively.

    """
    # check parameters
    stack.check_parameter(threshold=(float, int),
                          remove_duplicate=bool,
                          voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float))

    # compute sigma and radius
    ndim = image.ndim
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and psf_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'psf_z' parameter is missing.".format(ndim))
    if ndim == 2:
        voxel_size_z = None
        psf_z = None
    sigma = get_sigma(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

    # cast image in np.float and apply LoG filter
    image_filtered = stack.log_filter(image, sigma)

    # find local maximum
    mask_local_max = local_maximum_detection(image_filtered, sigma)

    # remove spots with a low intensity and return their coordinates
    spots, _ = spots_thresholding(image_filtered, mask_local_max, threshold,
                                  remove_duplicate)

    return spots


def local_maximum_detection(image, min_distance):
    """Compute a mask to keep only local maximum, in 2-d and 3-d.

    1) We apply a multidimensional maximum filter.
    2) A pixel which has the same value in the original and filtered images
    is a local maximum.

    Several connected pixels can have the same value. In such a case, the
    local maximum is not unique.

    Parameters
    ----------
    image : np.ndarray
        Image to process with shape (z, y, x) or (y, x).
    min_distance : int, float or Tuple(float)
        Minimum distance (in pixels) between two spots we want to be able to
        detect separately. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same distance is applied to
        every dimensions.

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
    filtered image. If the local maximum is not unique (it can happen with
    connected pixels with the same value), connected component algorithm is
    applied to keep only one coordinate per spot.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs.
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
    stack.check_parameter(threshold=(float, int),
                          remove_duplicate=bool)

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

    return spots, mask
