# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""Filtering functions."""

import numpy as np

from .utils import check_array
from .utils import check_parameter

from .preprocess import cast_img_float32
from .preprocess import cast_img_float64
from .preprocess import cast_img_uint8
from .preprocess import cast_img_uint16

from skimage.morphology.selem import square
from skimage.morphology.selem import diamond
from skimage.morphology.selem import rectangle
from skimage.morphology.selem import disk
from skimage.morphology import binary_dilation
from skimage.morphology import dilation
from skimage.morphology import binary_erosion
from skimage.morphology import erosion
from skimage.filters import rank
from skimage.filters import gaussian

from scipy.ndimage import gaussian_laplace
from scipy.ndimage import convolve


# ### Filters ###

def _define_kernel(shape, size, dtype):
    """Build a kernel to apply a filter on images.

    Parameters
    ----------
    shape : str
        Shape of the kernel used to compute the filter (`diamond`, `disk`,
        `rectangle` or `square`).
    size : int, Tuple(int) or List(int)
        The size of the kernel:
            - For the rectangle we expect two values (`height`, `width`).
            - For the square one value (`width`).
            - For the disk and the diamond one value (`radius`).
    dtype : type
        Dtype used for the kernel (the same as the image).

    Returns
    -------
    kernel : skimage.morphology.selem object
        Kernel to use with a skimage filter.

    """
    # build the kernel
    if shape == "diamond":
        kernel = diamond(size, dtype=dtype)
    elif shape == "disk":
        kernel = disk(size, dtype=dtype)
    elif shape == "rectangle" and isinstance(size, (tuple, list)):
        kernel = rectangle(size[0], size[1], dtype=dtype)
    elif shape == "square":
        kernel = square(size, dtype=dtype)
    else:
        raise ValueError("Kernel definition is wrong. Shape of the kernel "
                         "should be 'diamond', 'disk', 'rectangle' or "
                         "'square'. Not {0}.".format(shape))

    return kernel


def mean_filter(image, kernel_shape, kernel_size):
    """Apply a mean filter to a 2-d through convolution filter.

    Parameters
    ----------
    image : np.ndarray, np.uint or np.float
        Image with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter (`diamond`, `disk`,
        `rectangle` or `square`).
    kernel_size : int, Tuple(int) or List(int)
        The size of the kernel. For the rectangle we expect two integers
        (`height`, `width`).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(
        image,
        ndim=2,
        dtype=[np.float32, np.float64, np.uint8, np.uint16])
    check_parameter(
        kernel_shape=str,
        kernel_size=(int, tuple, list))

    # build kernel
    kernel = _define_kernel(
        shape=kernel_shape,
        size=kernel_size,
        dtype=np.float64)
    n = kernel.sum()
    kernel /= n

    # apply convolution filter
    image_filtered = convolve(image, kernel)

    return image_filtered


def median_filter(image, kernel_shape, kernel_size):
    """Apply a median filter to a 2-d image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter (`diamond`, `disk`,
        `rectangle` or `square`).
    kernel_size : int, Tuple(int) or List(int)
        The size of the kernel. For the rectangle we expect two integers
        (`height`, `width`).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16])
    check_parameter(
        kernel_shape=str,
        kernel_size=(int, tuple, list))

    # get kernel
    kernel = _define_kernel(
        shape=kernel_shape,
        size=kernel_size,
        dtype=image.dtype)

    # apply filter
    image_filtered = rank.median(image, kernel)

    return image_filtered


def maximum_filter(image, kernel_shape, kernel_size):
    """Apply a maximum filter to a 2-d image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter (`diamond`, `disk`,
        `rectangle` or `square`).
    kernel_size : int, Tuple(int) or List(int)
        The size of the kernel. For the rectangle we expect two integers
        (`height`, `width`).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16])
    check_parameter(
        kernel_shape=str,
        kernel_size=(int, tuple, list))

    # get kernel
    kernel = _define_kernel(
        shape=kernel_shape,
        size=kernel_size,
        dtype=image.dtype)

    # apply filter
    image_filtered = rank.maximum(image, kernel)

    return image_filtered


def minimum_filter(image, kernel_shape, kernel_size):
    """Apply a minimum filter to a 2-d image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter (`diamond`, `disk`,
        `rectangle` or `square`).
    kernel_size : int, Tuple(int) or List(int)
        The size of the kernel. For the rectangle we expect two integers
        (`height`, `width`).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16])
    check_parameter(
        kernel_shape=str,
        kernel_size=(int, tuple, list))

    # get kernel
    kernel = _define_kernel(
        shape=kernel_shape,
        size=kernel_size,
        dtype=image.dtype)

    # apply filter
    image_filtered = rank.minimum(image, kernel)

    return image_filtered


def log_filter(image, sigma):
    """Apply a Laplacian of Gaussian filter to a 2-d or 3-d image.

    The function returns the inverse of the filtered image such that the pixels
    with the highest intensity from the original (smoothed) image have
    positive values. Those with a low intensity returning a negative value are
    clipped to zero.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    sigma : int, float, Tuple(float, int) or List(float, int)
        Standard deviation used for the gaussian kernel (one for each
        dimension). If it's a scalar, the same standard deviation is applied
        to every dimensions.

    Returns
    -------
    image_filtered : np.ndarray
        Filtered image.

    """
    # check parameters
    check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    check_parameter(sigma=(float, int, tuple, list))

    # we cast the data in np.float to allow negative values
    if image.dtype == np.uint8:
        image_float = cast_img_float32(image)
    elif image.dtype == np.uint16:
        image_float = cast_img_float64(image)
    else:
        image_float = image

    # check sigma
    if isinstance(sigma, (tuple, list)):
        if len(sigma) != image.ndim:
            raise ValueError("'sigma' must be a scalar or a sequence with {0} "
                             "elements.".format(image.ndim))

    # we apply LoG filter
    image_filtered = gaussian_laplace(image_float, sigma=sigma)

    # as the LoG filter makes the peaks in the original image appear as a
    # reversed mexican hat, we inverse the result and clip negative values to 0
    image_filtered = np.clip(-image_filtered, a_min=0, a_max=None)

    # cast filtered image
    if image.dtype == np.uint8:
        image_filtered = cast_img_uint8(image_filtered)
    elif image.dtype == np.uint16:
        image_filtered = cast_img_uint16(image_filtered)
    else:
        pass

    return image_filtered


def gaussian_filter(image, sigma, allow_negative=False):
    """Apply a Gaussian filter to a 2-d or 3-d image.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    sigma : int, float, Tuple(float, int) or List(float, int)
        Standard deviation used for the gaussian kernel (one for each
        dimension). If it's a scalar, the same standard deviation is applied
        to every dimensions.
    allow_negative : bool
        Allow negative values after the filtering or clip them to 0. Not
        compatible with unsigned integer images.

    Returns
    -------
    image_filtered : np.ndarray
        Filtered image.

    """
    # check parameters
    check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    check_parameter(
        sigma=(float, int, tuple, list),
        allow_negative=bool)

    # check parameters consistency
    if image.dtype in [np.uint8, np.uint16] and allow_negative:
        raise ValueError("Negative values are impossible with unsigned "
                         "integer image.")
    # check sigma
    if isinstance(sigma, (tuple, list)):
        if len(sigma) != image.ndim:
            raise ValueError("'sigma' must be a scalar or a sequence with {0} "
                             "elements.".format(image.ndim))

    # we cast the data in np.float to allow negative values
    if image.dtype == np.uint8:
        image_float = cast_img_float32(image)
    elif image.dtype == np.uint16:
        image_float = cast_img_float64(image)
    else:
        image_float = image

    # we apply gaussian filter
    image_filtered = gaussian(image_float, sigma=sigma)

    # we clip negative values to 0
    if not allow_negative:
        image_filtered = np.clip(image_filtered, a_min=0, a_max=1)

    # cast filtered image
    if image.dtype == np.uint8:
        image_filtered = cast_img_uint8(image_filtered)
    elif image.dtype == np.uint16:
        image_filtered = cast_img_uint16(image_filtered)
    else:
        pass

    return image_filtered


def remove_background_mean(image, kernel_shape="disk", kernel_size=200):
    """Remove background noise from a 2-d image, subtracting a mean filtering.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image to process with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter (`diamond`, `disk`,
        `rectangle` or `square`).
    kernel_size : int, Tuple(int) or List(int)
        The size of the kernel. For the rectangle we expect two integers
        (`height`, `width`).

    Returns
    -------
    image_without_back : np.ndarray, np.uint
        Image processed.

    """
    # compute background noise with a large mean filter
    background = mean_filter(
        image,
        kernel_shape=kernel_shape,
        kernel_size=kernel_size)

    # subtract the background from the original image, clipping negative
    # values to 0
    mask = image > background
    image_without_back = np.subtract(
        image, background,
        out=np.zeros_like(image),
        where=mask)

    return image_without_back


def remove_background_gaussian(image, sigma):
    """Remove background noise from a 2-d or 3-d image, subtracting a gaussian
    filtering.

    Parameters
    ----------
    image : np.ndarray
        Image to process with shape (z, y, x) or (y, x).
    sigma : int, float, Tuple(float, int) or List(float, int)
        Standard deviation used for the gaussian kernel (one for each
        dimension). If it's a scalar, the same standard deviation is applied
        to every dimensions.

    Returns
    -------
    image_no_background : np.ndarray
        Image processed with shape (z, y, x) or (y, x).

    """
    # apply a gaussian filter
    image_filtered = gaussian_filter(image, sigma, allow_negative=False)

    # subtract the gaussian filter
    out = np.zeros_like(image)
    image_no_background = np.subtract(
        image, image_filtered,
        out=out,
        where=(image > image_filtered),
        dtype=image.dtype)

    return image_no_background


def dilation_filter(image, kernel_shape=None, kernel_size=None):
    """Apply a dilation to a 2-d image.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter (`diamond`, `disk`,
        `rectangle` or `square`). If None, use cross-shaped structuring
        element (``connectivity=1``).
    kernel_size : int, Tuple(int) or List(int)
        The size of the kernel. For the rectangle we expect two integers
        (`height`, `width`). If None, use cross-shaped structuring element
        (``connectivity=1``).

    Returns
    -------
    image_filtered : np.ndarray
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.float32, np.float64, bool])
    check_parameter(
        kernel_shape=(str, type(None)),
        kernel_size=(int, tuple, list, type(None)))

    # get kernel
    if kernel_shape is None or kernel_size is None:
        kernel = None
    else:
        kernel = _define_kernel(
            shape=kernel_shape,
            size=kernel_size,
            dtype=image.dtype)

    # apply filter
    if image.dtype == bool:
        image_filtered = binary_dilation(image, kernel)
    else:
        image_filtered = dilation(image, kernel)

    return image_filtered


def erosion_filter(image, kernel_shape=None, kernel_size=None):
    """Apply an erosion to a 2-d image.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter (`diamond`, `disk`,
        `rectangle` or `square`). If None, use cross-shaped structuring
        element (``connectivity=1``).
    kernel_size : int, Tuple(int) or List(int)
        The size of the kernel. For the rectangle we expect two integers
        (`height`, `width`). If None, use cross-shaped structuring element
        (``connectivity=1``).

    Returns
    -------
    image_filtered : np.ndarray
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.float32, np.float64, bool])
    check_parameter(
        kernel_shape=(str, type(None)),
        kernel_size=(int, tuple, list, type(None)))

    # get kernel
    if kernel_shape is None or kernel_size is None:
        kernel = None
    else:
        kernel = _define_kernel(
            shape=kernel_shape,
            size=kernel_size,
            dtype=image.dtype)

    # apply filter
    if image.dtype == bool:
        image_filtered = binary_erosion(image, kernel)
    else:
        image_filtered = erosion(image, kernel)

    return image_filtered
