# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""2-d projection functions."""

import numpy as np

from .utils import check_array
from .utils import check_parameter
from .quality import compute_focus


# ### Projections 2-d ###

def maximum_projection(image):
    """Project the z-dimension of an image, keeping the maximum intensity of
    each yx pixel.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with shape (z, y, x).

    Returns
    -------
    projected_image : np.ndarray, np.uint
        A 2-d image with shape (y, x).

    """
    # check parameters
    check_array(image, ndim=3, dtype=[np.uint8, np.uint16])

    # project image along the z axis
    projected_image = image.max(axis=0)

    return projected_image


def mean_projection(image, return_float=False):
    """Project the z-dimension of a image, computing the mean intensity of
    each yx pixel.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d tensor with shape (z, y, x).
    return_float : bool
        Return a (potentially more accurate) float array.

    Returns
    -------
    projected_image : np.ndarray
        A 2-d image with shape (y, x).

    """
    # check parameters
    check_array(image, ndim=3, dtype=[np.uint8, np.uint16])

    # project image along the z axis
    if return_float:
        projected_image = image.mean(axis=0)
    else:
        projected_image = image.mean(axis=0).astype(image.dtype)

    return projected_image


def median_projection(image):
    """Project the z-dimension of a image, computing the median intensity of
    each yx pixel.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with shape (z, y, x).

    Returns
    -------
    projected_image : np.ndarray, np.uint
        A 2-d image with shape (y, x).

    """
    # check parameters
    check_array(image, ndim=3, dtype=[np.uint8, np.uint16])

    # project image along the z axis
    projected_image = np.median(image, axis=0)
    projected_image = projected_image.astype(image.dtype)

    return projected_image


def focus_projection(image, proportion=5, neighborhood_size=7,
                     method="median"):
    """Project the z-dimension of an image.

    Inspired from Samacoits Aubin's thesis (part 5.3, strategy 5). Compare to
    the original algorithm we use the same focus measures to select the
    in-focus z-slices and project our image.

    #. Compute a focus score for each pixel yx with a fixed neighborhood size.
    #. We keep 75% of z-slices with the highest average focus score.
    #. Keep the median/maximum pixel intensity among the top 5 z-slices with
       the highest focus score.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with shape (z, y, x).
    proportion : float or int
        Proportion of z-slices to keep (float between 0 and 1) or number of
        z-slices to keep (positive integer).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.
    method : str
        Projection method applied on the selected pixel values (`median` or
        `max`).

    Returns
    -------
    projected_image : np.ndarray, np.uint
        A 2-d image with shape (y, x).

    """
    # check parameters
    check_array(image, ndim=3, dtype=[np.uint8, np.uint16])
    check_parameter(
        proportion=(float, int),
        neighborhood_size=int)
    if isinstance(proportion, float) and 0 <= proportion <= 1:
        pass
    elif isinstance(proportion, int) and 0 <= proportion:
        pass
    else:
        raise ValueError("'proportion' should be a float between 0 and 1 or a "
                         "positive integer, but not {0}.".format(proportion))

    # compute focus measure for each pixel
    focus = compute_focus(image, neighborhood_size)

    # select and keep best z-slices
    indices_to_keep = get_in_focus_indices(focus, proportion)
    in_focus_image = image[indices_to_keep]
    focus = focus[indices_to_keep]

    # for each yx pixel, get the indices of the 5 best focus values
    top_focus_indices = np.argsort(focus, axis=0)
    n = min(focus.shape[0], 5)
    top_focus_indices = top_focus_indices[-n:, :, :]

    # build a binary matrix with the same shape of our in-focus image to keep
    # the top focus pixels only
    mask = [mask_ for mask_ in map(
        lambda indices: _one_hot_3d(indices, depth=in_focus_image.shape[0]),
        top_focus_indices)]
    mask = np.sum(mask, axis=0, dtype=in_focus_image.dtype)

    # filter top focus pixels in our in-focus image
    in_focus_image = np.multiply(in_focus_image, mask)

    # project image
    in_focus_image = in_focus_image.astype(np.float64)
    in_focus_image[in_focus_image == 0] = np.nan
    if method == "median":
        projected_image = np.nanmedian(in_focus_image, axis=0)
    elif method == "max":
        projected_image = np.nanmax(in_focus_image, axis=0)
    else:
        raise ValueError("Parameter 'method' should be 'median' or 'max', not "
                         "'{0}'.".format(method))
    projected_image = projected_image.astype(image.dtype)

    return projected_image


def _one_hot_3d(indices, depth, return_boolean=False):
    """Build a 3-d one-hot matrix from a 2-d indices matrix.

    Parameters
    ----------
    indices : np.ndarray, int
        A 2-d tensor with integer indices and shape (y, x).
    depth : int
        Depth of the 3-d one-hot matrix.
    return_boolean : bool
        Return a boolean one-hot encoded matrix.

    Returns
    -------
    one_hot : np.ndarray
        A 3-d binary tensor with shape (depth, y, x)

    """
    # check parameters
    check_parameter(depth=int)
    check_array(
        indices,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
               np.int8, np.int16, np.int32, np.int64])

    # initialize the 3-d one-hot matrix
    one_hot = np.zeros((indices.size, depth), dtype=indices.dtype)

    # flatten the matrix to easily one-hot encode it, then reshape it
    one_hot[np.arange(indices.size), indices.ravel()] = 1
    one_hot.shape = indices.shape + (depth,)

    # rearrange the axis
    one_hot = np.moveaxis(one_hot, source=2, destination=0)

    if return_boolean:
        one_hot = one_hot.astype(bool)

    return one_hot


# ### Slice selection ###

def in_focus_selection(image, focus, proportion):
    """Select and keep the 2-d slices with the highest level of focus.

    Helmli and Scherer’s mean method is used as a focus metric.

    Parameters
    ----------
    image : np.ndarray
        A 3-d tensor with shape (z, y, x).
    focus : np.ndarray, np.float64
        A 3-d tensor with a focus metric computed for each pixel of the
        original image. See :func:`bigfish.stack.compute_focus`.
    proportion : float or int
        Proportion of z-slices to keep (float between 0 and 1) or number of
        z-slices to keep (positive integer).

    Returns
    -------
    in_focus_image : np.ndarray
        A 3-d tensor with shape (z_in_focus, y, x), with out-of-focus z-slice
        removed.

    """
    # check parameters
    check_array(
        image,
        ndim=3,
        dtype=[np.uint8, np.uint16, np.float32, np.float64])

    # select and keep best z-slices
    indices_to_keep = get_in_focus_indices(focus, proportion)
    in_focus_image = image[indices_to_keep]

    return in_focus_image


def get_in_focus_indices(focus, proportion):
    """ Select the best in-focus z-slices.

    Helmli and Scherer’s mean method is used as a focus metric.

    Parameters
    ----------
    focus : np.ndarray, np.float64
        A 3-d tensor with a focus metric computed for each pixel of the
        original image. See :func:`bigfish.stack.compute_focus`.
    proportion : float or int
        Proportion of z-slices to keep (float between 0 and 1) or number of
        z-slices to keep (positive integer).

    Returns
    -------
    indices_to_keep : List[int]
        Indices of slices with the best focus score.

    """
    # check parameters
    check_parameter(proportion=(float, int))
    check_array(focus, ndim=3, dtype=np.float64)
    if isinstance(proportion, float) and 0 <= proportion <= 1:
        n = int(focus.shape[0] * proportion)
    elif isinstance(proportion, int) and 0 <= proportion:
        n = int(proportion)
    else:
        raise ValueError("'proportion' should be a float between 0 and 1 or a "
                         "positive integer, but not {0}.".format(proportion))

    # measure focus level per 2-d slices
    focus_levels = np.mean(focus, axis=(1, 2))

    # select the best z-slices
    n = min(n, focus_levels.size)
    indices_to_keep = list(np.argsort(-focus_levels)[:n])
    indices_to_keep = sorted(indices_to_keep)

    return indices_to_keep
