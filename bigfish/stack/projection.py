# -*- coding: utf-8 -*-

"""2-d projection functions."""

import numpy as np

from .utils import check_array

from skimage import img_as_ubyte, img_as_float32
from skimage.filters import rank
from skimage.morphology.selem import square


# TODO add safety checks

# ### Projections 2-d ###

def projection(tensor, method="mip", r=0, c=0):
    """ Project a tensor along the z-dimension.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 5-d tensor with shape (r, c, z, y, x).
    method : str
        Method used to project ('mip', 'focus').
    r : int
        Index of a specific round to project.
    c : int
        Index of a specific channel to project.

    Returns
    -------
    projected_tensor : np.ndarray
        A 2-d tensor with shape (y, x).

    """
    # check tensor dimensions and its dtype
    check_array(tensor, ndim=5, dtype=[np.uint8, np.uint16])

    # apply projection along the z-dimension
    projected_tensor = tensor[r, c, :, :, :]
    if method == "mip":
        projected_tensor = maximum_projection(projected_tensor)
    elif method == "mean":
        projected_tensor = mean_projection(projected_tensor)
    elif method == "median":
        projected_tensor = median_projection(projected_tensor)
    elif method == "focus":
        # TODO complete focus projection with different strategies
        raise ValueError("Focus projection is not implemented yet.")

    return projected_tensor


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
    # project tensor along the z axis
    projected_tensor = tensor.max(axis=0, keepdims=True)

    return projected_tensor[0]


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
    # project tensor along the z axis
    projected_tensor = tensor.mean(axis=0, keepdims=True)

    return projected_tensor[0]


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
    # project tensor along the z axis
    projected_tensor = tensor.median(axis=0, keepdims=True)

    return projected_tensor[0]


def focus_projection(tensor, channel=0, p=0.75, global_neighborhood_size=30,
                     method="best"):
    """

    Parameters
    ----------
    tensor
    channel
    p
    global_neighborhood_size
    method

    Returns
    -------

    """

    # get 3-d image
    image = tensor[0, channel, :, :, :]

    # measure global focus level for each z-slices
    ratio, l_focus = focus_measurement_3d(image, global_neighborhood_size)

    # remove out-of-focus slices
    indices_to_keep = get_in_focus(l_focus, p)
    in_focus_image = image[indices_to_keep]

    projected_image = None
    if method == "bast":
        # for each pixel, we project the z-slice value with the highest focus
        ratio_2d = np.argmax(ratio[indices_to_keep], axis=0)
        one_hot = one_hot_3d(ratio_2d, depth=len(indices_to_keep))
        projected_image = np.multiply(in_focus_image, one_hot).max(axis=0)
    elif method == "median":
        # for each pixel, we compute the median value of the in-focus z-slices
        projected_image = np.median(in_focus_image, axis=0)
    elif method == "mean":
        # for each pixel, we compute the mean value of the in-focus z-slices
        projected_image = np.median(in_focus_image, axis=0)

    return projected_image, ratio, l_focus


def focus_measurement_2d(image, neighborhood_size):
    """Helmli and Scherer’s mean method used as a focus metric.

    For each pixel xy in an image, we compute the ratio:

        R(x, y) = mu(x, y) / I(x, y), if mu(x, y) >= I(x, y)

    or

        R(x, y) = I(x, y) / mu(x, y), otherwise

    with I(x, y) the intensity of the pixel xy and mu(x, y) the mean intensity
    of the pixels of its neighborhood.

    Parameters
    ----------
    image : np.ndarray, np.float32
        A 2-d tensor with shape (y, x).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.

    Returns
    -------
    global_focus : np.float32
        Mean value of the ratio computed for every pixels of the image. Can be
        used as a metric to quantify the focus level of an 2-d image.
    ratio : np.ndarray, np.float32
        A 2-d tensor with the R(x, y) computed for each pixel of the original
        image.
    image_filtered_mean : np.ndarray, np.float32
        A 2-d tensor with shape (y, x).

    """

    # scikit-image filter use np.uint dtype (so we cast to np.uint8)
    image_2d = img_as_ubyte(image)

    # filter the image with a mean filter
    selem = square(neighborhood_size)
    image_filtered_mean = rank.mean(image_2d, selem)

    # cast again in np.float32
    image_2d = img_as_float32(image_2d)
    image_filtered_mean = img_as_float32(image_filtered_mean)

    # case where mu(x, y) >= I(x, y)
    mask_1 = image_2d != 0
    out_1 = np.zeros_like(image_filtered_mean, dtype=np.float32)
    ratio_1 = np.divide(image_filtered_mean, image_2d, out=out_1, where=mask_1)
    ratio_1 = np.where(image_filtered_mean >= image_2d, ratio_1, 0)

    # case where I(x, y) > mu(x, y)
    mask_2 = image_filtered_mean != 0
    out_2 = np.zeros_like(image_2d, dtype=np.float32)
    ratio_2 = np.divide(image_2d, image_filtered_mean, out=out_2, where=mask_2)
    ratio_2 = np.where(image_2d > image_filtered_mean, ratio_2, 0)

    # compute ratio and global focus for the entire image
    ratio = ratio_1 + ratio_2
    global_focus = ratio.mean()

    return global_focus, ratio, image_filtered_mean


def focus_measurement_3d(image, neighborhood_size):
    """Helmli and Scherer’s mean method used as a focus metric.

    Parameters
    ----------
    image : np.ndarray, np.float32
        A 3-d tensor with shape (z, y, x).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.

    Returns
    -------
    ratio : np.ndarray, np.float32
        A 3-d tensor with the R(x, y) computed for each pixel of the original
        3-d image, for each z-slice.
    l_focus : list
        List of the global focus computed for each z-slice.

    """
    # apply focus_measurement_2d for each z-slice
    l_ratio = []
    l_focus = []
    for z in range(image.shape[0]):
        focus, ratio_2d, _ = focus_measurement_2d(image[z], neighborhood_size)
        l_ratio.append(ratio_2d)
        l_focus.append(focus)

    # get 3-d Helmli and Scherer’s ratio
    ratio = np.stack(l_ratio)

    return ratio, l_focus


def get_in_focus(l_focus, proportion):
    """ Select the best in-focus z-slices.

    Parameters
    ----------
    l_focus : array_like
        List of the global focus computed for each z-slice.
    proportion : float or int
        Proportion of z-slices to keep (float between 0 and 1) or number of
        z-slices to keep (integer above 1).

    Returns
    -------
    indices_to_keep : np.array
    """
    # get the number of z-slices to keep
    if proportion < 1 and isinstance(proportion, float):
        n = int(len(l_focus) * proportion)
    else:
        n = int(proportion)

    # select the best z-slices
    indices_to_keep = np.argsort(l_focus)[-n:]

    return indices_to_keep


def one_hot_3d(tensor_2d, depth):
    """Build a 3-d one-hot matrix from a 2-d indices matrix.

    Parameters
    ----------
    tensor_2d : np.ndarray, int
        A 2-d tensor with integer indices and shape (y, x).
    depth : int
        Depth of the 3-d one-hot matrix.

    Returns
    -------
    one_hot : np.ndarray, np.uint8
        A 3-d binary tensor with shape (depth, y, x)

    """
    # initialize the 3-d one-hot matrix
    one_hot = np.zeros((tensor_2d.size, depth), dtype=np.uint8)

    # flatten the matrix to easily one-hot encode it, then reshape it
    one_hot[np.arange(tensor_2d.size), tensor_2d.ravel()] = 1
    one_hot.shape = tensor_2d.shape + (depth,)

    # rearrange the axis
    one_hot = np.moveaxis(one_hot, source=2, destination=0)

    return one_hot
