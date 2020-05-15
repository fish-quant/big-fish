# -*- coding: utf-8 -*-

"""2-d projection functions."""

import numpy as np

from .utils import check_array, check_parameter
from .preprocess import cast_img_uint8
from .filter import mean_filter


# ### Projections 2-d ###

def maximum_projection(tensor):
    """Project the z-dimension of a tensor, keeping the maximum intensity of
    each yx pixel.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 3-d tensor with shape (z, y, x).

    Returns
    -------
    projected_tensor : np.ndarray, np.uint
        A 2-d tensor with shape (y, x).

    """
    # check parameters
    check_array(tensor, ndim=3, dtype=[np.uint8, np.uint16], allow_nan=False)

    # project tensor along the z axis
    projected_tensor = tensor.max(axis=0)

    return projected_tensor


def mean_projection(tensor):
    """Project the z-dimension of a tensor, computing the mean intensity of
    each yx pixel.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 3-d tensor with shape (z, y, x).

    Returns
    -------
    projected_tensor : np.ndarray, np.float
        A 2-d tensor with shape (y, x).

    """
    # check parameters
    check_array(tensor, ndim=3, dtype=[np.uint8, np.uint16], allow_nan=False)

    # project tensor along the z axis
    projected_tensor = tensor.mean(axis=0)

    return projected_tensor


def median_projection(tensor):
    """Project the z-dimension of a tensor, computing the median intensity of
    each yx pixel.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 3-d tensor with shape (z, y, x).

    Returns
    -------
    projected_tensor : np.ndarray, np.uint
        A 2-d tensor with shape (y, x).

    """
    # check parameters
    check_array(tensor, ndim=3, dtype=[np.uint8, np.uint16], allow_nan=False)

    # project tensor along the z axis
    projected_tensor = np.median(tensor, axis=0)
    projected_tensor = projected_tensor.astype(tensor.dtype)

    return projected_tensor


def focus_projection(tensor):
    """Project the z-dimension of a tensor as describe in Aubin's thesis
    (part 5.3, strategy 5).

    1) We keep 75% best in-focus z-slices.
    2) Compute a focus value for each voxel zyx with a 7x7 neighborhood window.
    3) Keep the median pixel intensity among the top 5 best focus z-slices.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 3-d tensor with shape (z, y, x).

    Returns
    -------
    projected_tensor : np.ndarray, np.uint
        A 2-d tensor with shape (y, x).

    """
    # check parameters
    check_array(tensor, ndim=3, dtype=[np.uint8, np.uint16], allow_nan=False)

    # remove out-of-focus z-slices
    in_focus_image = in_focus_selection(tensor,
                                        proportion=0.75,
                                        neighborhood_size=30)

    # compute focus value for each voxel with a smaller window.
    local_focus, _ = focus_measurement(in_focus_image, neighborhood_size=7)

    # for each yx pixel, get the indices of the 5 best focus values
    top_local_focus_indices = np.argsort(local_focus, axis=0)
    top_local_focus_indices = top_local_focus_indices[-5:, :, :]

    # build a binary matrix with the same shape of our in-focus image to keep
    # the top focus pixels only
    mask = [mask_ for mask_ in map(
        lambda indices: _one_hot_3d(indices, depth=in_focus_image.shape[0]),
        top_local_focus_indices)]
    mask = np.sum(mask, axis=0, dtype=in_focus_image.dtype)

    # filter top focus pixels in our in-focus image
    in_focus_image = np.multiply(in_focus_image, mask)

    # project tensor
    in_focus_image = in_focus_image.astype(np.float32)
    in_focus_image[in_focus_image == 0] = np.nan
    projected_tensor = np.nanmedian(in_focus_image, axis=0)
    projected_tensor = projected_tensor.astype(tensor.dtype)

    return projected_tensor


def focus_projection_fast(tensor, proportion=0.75, neighborhood_size=7,
                          method="median"):
    """Project the z-dimension of a tensor.

    Inspired from Aubin's thesis (part 5.3, strategy 5). Compare to the
    original algorithm we use the same focus levels to select the in-focus
    z-slices and project our tensor.

    1) Compute a focus value for each voxel zyx with a fixed neighborhood size.
    2) We keep 75% best in-focus z-slices (based on a global focus score).
    3) Keep the median/maximum pixel intensity among the top 5 best
    focus z-slices.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 3-d tensor with shape (z, y, x).
    proportion : float or int
        Proportion of z-slices to keep (float between 0 and 1) or number of
        z-slices to keep (integer above 1).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.
    method : str
        Projection method applied on the selected pixel values.

    Returns
    -------
    projected_tensor : np.ndarray, np.uint
        A 2-d tensor with shape (y, x).

    """
    # TODO case where proportion = {0, 1}
    # check parameters
    check_array(tensor, ndim=3, dtype=[np.uint8, np.uint16], allow_nan=False)
    check_parameter(proportion=(float, int),
                    neighborhood_size=int)
    if isinstance(proportion, float) and 0 <= proportion <= 1:
        pass
    elif isinstance(proportion, int) and 0 <= proportion:
        pass
    else:
        raise ValueError("'proportion' should be a float between 0 and 1 or a "
                         "positive integer, but not {0}.".format(proportion))

    # compute focus value for each voxel.
    local_focus, global_focus = focus_measurement(tensor, neighborhood_size)

    # select and keep best z-slices
    indices_to_keep = get_in_focus_indices(global_focus, proportion)
    in_focus_image = tensor[indices_to_keep]
    local_focus = local_focus[indices_to_keep]

    # for each yx pixel, get the indices of the 5 best focus values
    top_local_focus_indices = np.argsort(local_focus, axis=0)
    n = min(local_focus.shape[0], 5)
    top_local_focus_indices = top_local_focus_indices[-n:, :, :]

    # build a binary matrix with the same shape of our in-focus image to keep
    # the top focus pixels only
    mask = [mask_ for mask_ in map(
        lambda indices: _one_hot_3d(indices, depth=in_focus_image.shape[0]),
        top_local_focus_indices)]
    mask = np.sum(mask, axis=0, dtype=in_focus_image.dtype)

    # filter top focus pixels in our in-focus image
    in_focus_image = np.multiply(in_focus_image, mask)

    # project tensor
    in_focus_image = in_focus_image.astype(np.float32)
    in_focus_image[in_focus_image == 0] = np.nan
    if method == "median":
        projected_tensor = np.nanmedian(in_focus_image, axis=0)
    elif method == "max":
        projected_tensor = np.nanmax(in_focus_image, axis=0)
    else:
        raise ValueError("Parameter 'method' should be 'median' or 'max', not "
                         "'{0}'.".format(method))
    projected_tensor = projected_tensor.astype(tensor.dtype)

    return projected_tensor


# ### Focus selection ###

def in_focus_selection(image, proportion, neighborhood_size=30):
    """Select and keep the slices with the highest level of focus.

    Helmli and Scherer’s mean method used as a focus metric.

    Parameters
    ----------
    image : np.ndarray
        A 3-d tensor with shape (z, y, x).
    proportion : float or int
        Proportion of z-slices to keep (float between 0 and 1) or number of
        z-slices to keep (integer above 1).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.

    Returns
    -------
    in_focus_image : np.ndarray
        A 3-d tensor with shape (z_in_focus, y, x), with out-of-focus z-slice
        removed.

    """
    # check parameters
    check_array(image,
                ndim=3,
                dtype=[np.uint8, np.uint16, np.float32, np.float64],
                allow_nan=False)
    check_parameter(proportion=(float, int),
                    neighborhood_size=int)
    if isinstance(proportion, float) and 0 <= proportion <= 1:
        pass
    elif isinstance(proportion, int) and 0 <= proportion:
        pass
    else:
        raise ValueError("'proportion' should be a float between 0 and 1 or a "
                         "positive integer, but not {0}.".format(proportion))

    # measure focus level
    _, global_focus = focus_measurement(image, neighborhood_size)

    # select and keep best z-slices
    indices_to_keep = get_in_focus_indices(global_focus, proportion)
    in_focus_image = image[indices_to_keep]

    return in_focus_image


def focus_measurement(image, neighborhood_size=30):
    """Helmli and Scherer’s mean method used as a focus metric.

    For each pixel xy in an image, we compute the ratio:

        R(x, y) = mu(x, y) / I(x, y), if mu(x, y) >= I(x, y)
    or
        R(x, y) = I(x, y) / mu(x, y), otherwise

    with I(x, y) the intensity of the pixel xy and mu(x, y) the mean intensity
    of the pixels of its neighborhood.

    Parameters
    ----------
    image : np.ndarray
        A 2-d or 3-d tensor with shape (y, x) or (z, y, x).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.

    Returns
    -------
    ratio : np.ndarray, np.float32
        A 2-d or 3-d tensor with the R(x, y) computed for each pixel of the
        original image.
    global_focus : np.ndarray, np.float32
        Mean value of the ratio computed for every pixels of each 2-d slice.
        Can be used as a metric to quantify the focus level this slice. Shape
        is (z,) for a 3-d image or (,) for a 2-d image.

    """
    # check parameters
    check_array(image,
                ndim=[2, 3],
                dtype=[np.uint8, np.uint16, np.float32, np.float64],
                allow_nan=False)
    check_parameter(neighborhood_size=int)

    # cast image in np.uint8
    image = cast_img_uint8(image)

    if image.ndim == 2:
        ratio, global_focus = _focus_measurement_2d(image, neighborhood_size)
    else:
        ratio, global_focus = _focus_measurement_3d(image, neighborhood_size)

    return ratio, global_focus


def _focus_measurement_2d(image, neighborhood_size):
    """Helmli and Scherer’s mean method used as a focus metric.

    For each pixel xy in an image, we compute the ratio:

        R(x, y) = mu(x, y) / I(x, y), if mu(x, y) >= I(x, y)
    or
        R(x, y) = I(x, y) / mu(x, y), otherwise

    with I(x, y) the intensity of the pixel xy and mu(x, y) the mean intensity
    of the pixels of its neighborhood.

    Parameters
    ----------
    image : np.ndarray, np.np.uint8
        A 2-d tensor with shape (y, x).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.

    Returns
    -------
    ratio : np.ndarray, np.float32
        A 2-d tensor with the R(x, y) computed for each pixel of the
        original image.
    global_focus : np.ndarray, np.float32
        Mean value of the ratio computed for every pixels of each 2-d slice.
        Can be used as a metric to quantify the focus level this slice. Shape
        is () for a 2-d image.

    """
    # filter the image with a mean filter
    image_filtered_mean = mean_filter(image, "square", neighborhood_size)

    # case where mu(x, y) >= I(x, y)
    mask_1 = (image != 0)
    out_1 = np.zeros_like(image_filtered_mean, dtype=np.float32)
    ratio_1 = np.divide(image_filtered_mean, image, out=out_1, where=mask_1)
    ratio_1 = np.where(image_filtered_mean >= image, ratio_1, 0)

    # case where I(x, y) > mu(x, y)
    mask_2 = image_filtered_mean != 0
    out_2 = np.zeros_like(image, dtype=np.float32)
    ratio_2 = np.divide(image, image_filtered_mean, out=out_2, where=mask_2)
    ratio_2 = np.where(image > image_filtered_mean, ratio_2, 0)

    # compute ratio and global focus for the entire image
    ratio = ratio_1 + ratio_2
    ratio = ratio.astype(np.float32)
    global_focus = ratio.mean()

    return ratio, global_focus


def _focus_measurement_3d(image, neighborhood_size):
    """Helmli and Scherer’s mean method used as a focus metric.

    Parameters
    ----------
    image : np.ndarray, np.uint8
        A 3-d tensor with shape (z, y, x).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.

    Returns
    -------
    ratio : np.ndarray, np.float32
        A 3-d tensor with the R(x, y) computed for each pixel of the
        original image.
    global_focus : np.ndarray, np.float32
        Mean value of the ratio computed for every pixels of each 2-d slice.
        Can be used as a metric to quantify the focus level this slice. Shape
        is (z,) for a 3-d image.

    """
    # apply focus_measurement_2d for each z-slice
    l_ratio = []
    l_focus = []
    for z in range(image.shape[0]):
        ratio, global_focus = _focus_measurement_2d(image[z],
                                                    neighborhood_size)
        l_ratio.append(ratio)
        l_focus.append(global_focus)

    # get a 3-d results
    ratio = np.stack(l_ratio)
    global_focus = np.stack(l_focus)

    return ratio, global_focus


def get_in_focus_indices(global_focus, proportion):
    """ Select the best in-focus z-slices.

    Parameters
    ----------
    global_focus : np.ndarray, np.float32
        Mean value of the ratio computed for every pixels of each 2-d slice.
        Can be used as a metric to quantify the focus level this slice. Shape
        is (z,) for a 3-d image or () for a 2-d image.
    proportion : float or int
        Proportion of z-slices to keep (float between 0 and 1) or number of
        z-slices to keep (integer above 1).

    Returns
    -------
    indices_to_keep : List[int]
        Sorted indices of slices with the best focus score (decreasing score).

    """
    # check parameters
    check_parameter(global_focus=(np.ndarray, np.float32),
                    proportion=(float, int))
    if isinstance(global_focus, np.ndarray):
        check_array(global_focus,
                    ndim=[0, 1],
                    dtype=np.float32,
                    allow_nan=False)
    if isinstance(proportion, float) and 0 <= proportion <= 1:
        n = int(len(global_focus) * proportion)
    elif isinstance(proportion, int) and 0 <= proportion:
        n = int(proportion)
    else:
        raise ValueError("'proportion' should be a float between 0 and 1 or a "
                         "positive integer, but not {0}.".format(proportion))

    # select the best z-slices
    n = min(n, global_focus.size)
    indices_to_keep = list(np.argsort(-global_focus)[:n])

    return indices_to_keep


def _one_hot_3d(indices, depth):
    """Build a 3-d one-hot matrix from a 2-d indices matrix.

    Parameters
    ----------
    indices : np.ndarray, int
        A 2-d tensor with integer indices and shape (y, x).
    depth : int
        Depth of the 3-d one-hot matrix.

    Returns
    -------
    one_hot : np.ndarray, np.uint8
        A 3-d binary tensor with shape (depth, y, x)

    """
    # initialize the 3-d one-hot matrix
    one_hot = np.zeros((indices.size, depth), dtype=np.uint8)

    # flatten the matrix to easily one-hot encode it, then reshape it
    one_hot[np.arange(indices.size), indices.ravel()] = 1
    one_hot.shape = indices.shape + (depth,)

    # rearrange the axis
    one_hot = np.moveaxis(one_hot, source=2, destination=0)

    return one_hot
