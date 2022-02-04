# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions used to detect and clean noisy images.
"""

import numpy as np

from .utils import check_array, check_parameter
from .filter import mean_filter


# ### Focus ###

def compute_focus(image, neighborhood_size=31):
    """Helmli and Scherer’s mean method is used as a focus metric.

    For each pixel yx in a 2-d image, we compute the ratio:

    .. math::

        R(y, x) = \\left \\{ \\begin{array}{rcl} \\frac{I(y, x)}{\\mu(y, x)} &
        \\mbox{if} & I(y, x) \\ge \\mu(y, x) \\\ \\frac{\\mu(y, x)}{I(y, x)} &
        \\mbox{otherwise} & \\end{array} \\right.

    with :math:`I(y, x)` the intensity of the pixel yx and :math:`\\mu(y, x)`
    the mean intensity of the pixels in its neighborhood.

    For a 3-d image, we compute this metric for each z surface.

    Parameters
    ----------
    image : np.ndarray
        A 2-d or 3-d image with shape (y, x) or (z, y, x).
    neighborhood_size : int or tuple or list, default=31
        The size of the square used to define the neighborhood of each pixel.
        An odd value is preferred. To define a rectangular neighborhood, a
        tuple or a list with two elements (height, width) can be provided.

    Returns
    -------
    focus : np.ndarray, np.float64
        A 2-d or 3-d tensor with the R(y, x) computed for each pixel of the
        original image.

    """
    # check parameters
    check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    check_parameter(neighborhood_size=(int, tuple, list))
    if (isinstance(neighborhood_size, (tuple, list))
            and len(neighborhood_size) != 2):
        raise ValueError("Parameter 'neighborhood_size' should be an integer "
                         "(to define a square neighborhood) or a sequence of "
                         "two elements (to define a rectangular "
                         "neighborhood). Not a sequence with {0} elements."
                         .format(len(neighborhood_size)))

    # cast image in float if necessary
    if image.dtype in [np.uint8, np.uint16]:
        image_float = image.astype(np.float64)
    else:
        image_float = image

    # build kernel
    if image.ndim == 3:
        focus = _compute_focus_3d(image_float, neighborhood_size)
    else:
        focus = _compute_focus_2d(image_float, neighborhood_size)

    return focus


def _compute_focus_3d(image_3d, kernel_size):
    """Compute a pixel-wise focus metric for a 3-d image.

    Parameters
    ----------
    image_3d : np.ndarray, np.float
        A 3-d image with shape (z, y, x).
    kernel_size : int or tuple or list
        The size of the square used to define a kernel size. An odd value is
        preferred. To define a rectangular kernel, a tuple or a list with two
        elements (height, width) can be provided.

    Returns
    -------
    focus : np.ndarray, np.float64
        A 3-d tensor with the R(z, y, x) computed for each pixel of the
        original image.

    """
    # compute focus metric for each z surface
    focus = []
    for image_2d in image_3d:
        focus_2d = _compute_focus_2d(image_2d, kernel_size)
        focus.append(focus_2d)

    #  stack focus metrics
    focus = np.stack(focus)

    return focus


def _compute_focus_2d(image_2d, kernel_size):
    """Compute a pixel-wise focus metric for a 2-d image.

    Parameters
    ----------
    image_2d : np.ndarray, np.float
        A 2-d image with shape (y, x).
    kernel_size : int or tuple or list
        The size of the square used to define a kernel size. An odd value is
        preferred. To define a rectangular kernel, a tuple or a list with two
        elements (height, width) can be provided.

    Returns
    -------
    focus : np.ndarray, np.float64
        A 2-d tensor with the R(y, x) computed for each pixel of the original
        image.

    """
    # mean filtered image
    if isinstance(kernel_size, int):
        image_filtered_mean = mean_filter(image_2d, "square", kernel_size)
    else:
        image_filtered_mean = mean_filter(image_2d, "rectangle", kernel_size)

    # compute focus metric
    ratio_default_1 = np.ones_like(image_2d, dtype=np.float64)
    ratio_default_2 = np.ones_like(image_filtered_mean, dtype=np.float64)
    ratio_1 = np.divide(
        image_2d, image_filtered_mean,
        out=ratio_default_1,
        where=image_filtered_mean > 0)
    ratio_2 = np.divide(
        image_filtered_mean, image_2d,
        out=ratio_default_2,
        where=image_2d > 0)
    focus = np.where(image_2d >= image_filtered_mean, ratio_1, ratio_2)

    return focus
