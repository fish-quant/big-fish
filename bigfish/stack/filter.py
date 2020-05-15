# -*- coding: utf-8 -*-

"""Filter functions."""

import numpy as np

from .utils import check_array, check_parameter
from .preprocess import (cast_img_float32, cast_img_float64, cast_img_uint8,
                         cast_img_uint16)

from skimage.morphology.selem import square, diamond, rectangle, disk
from skimage.morphology import (binary_dilation, dilation, binary_erosion,
                                erosion)
from skimage.filters import rank, gaussian

from scipy.ndimage import gaussian_laplace


# ### Filters ###

def _define_kernel(shape, size, dtype):
    """Build a kernel to apply a filter on images.

    Parameters
    ----------
    shape : str
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    size : int, Tuple(int) or List(int)
        The size of the kernel:
            - For the rectangle we expect two values (width, height).
            - For the square one value (width).
            - For the disk and the diamond one value (radius).
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
    elif shape == "rectangle" and isinstance(size, tuple):
        kernel = rectangle(size[0], size[1], dtype=dtype)
    elif shape == "square":
        kernel = square(size, dtype=dtype)
    else:
        raise ValueError("Kernel definition is wrong.")

    return kernel


def mean_filter(image, kernel_shape, kernel_size):
    """Apply a mean filter to a 2-d image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(image,
                ndim=2,
                dtype=[np.uint8, np.uint16])
    check_parameter(kernel_shape=str,
                    kernel_size=(int, tuple, list))

    # get kernel
    kernel = _define_kernel(shape=kernel_shape,
                            size=kernel_size,
                            dtype=image.dtype)

    # apply filter
    image_filtered = rank.mean(image, kernel)

    return image_filtered


def median_filter(image, kernel_shape, kernel_size):
    """Apply a median filter to a 2-d image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (y, x).
    kernel_shape : str
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(image,
                ndim=2,
                dtype=[np.uint8, np.uint16])
    check_parameter(kernel_shape=str,
                    kernel_size=(int, tuple, list))

    # get kernel
    kernel = _define_kernel(shape=kernel_shape,
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
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(image,
                ndim=2,
                dtype=[np.uint8, np.uint16])
    check_parameter(kernel_shape=str,
                    kernel_size=(int, tuple, list))

    # get kernel
    kernel = _define_kernel(shape=kernel_shape,
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
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # check parameters
    check_array(image,
                ndim=2,
                dtype=[np.uint8, np.uint16])
    check_parameter(kernel_shape=str,
                    kernel_size=(int, tuple, list))

    # get kernel
    kernel = _define_kernel(shape=kernel_shape,
                            size=kernel_size,
                            dtype=image.dtype)

    # apply filter
    image_filtered = rank.minimum(image, kernel)

    return image_filtered


def log_filter(image, sigma, keep_dtype=False):
    """Apply a Laplacian of Gaussian filter to a 2-d or 3-d image.

    The function returns the inverse of the filtered image such that the pixels
    with the highest intensity from the original (smoothed) image have
    positive values. Those with a low intensity returning a negative value are
    clipped to zero.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    sigma : float, int, Tuple(float, int) or List(float, int)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        float, the same sigma is applied to every dimensions.
    keep_dtype : bool
        Cast output image as input image.

    Returns
    -------
    image_filtered : np.ndarray
        Filtered image.

    """
    # check parameters
    check_array(image,
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
            raise ValueError("'sigma' must be a scalar or a sequence with the "
                             "same length as 'image.ndim'.")

    # we apply LoG filter
    image_filtered = gaussian_laplace(image_float, sigma=sigma)

    # as the LoG filter makes the peaks in the original image appear as a
    # reversed mexican hat, we inverse the result and clip negative values to 0
    image_filtered = np.clip(-image_filtered, a_min=0, a_max=None)

    # cast filtered image
    if keep_dtype:
        if image.dtype == np.uint8:
            image_filtered = cast_img_uint8(image_filtered)
        elif image.dtype == np.uint16:
            image_filtered = cast_img_uint16(image_filtered)
        else:
            pass

    return image_filtered


def gaussian_filter(image, sigma, allow_negative=False, keep_dtype=False):
    """Apply a Gaussian filter to a 2-d or 3-d image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    sigma : float, int, Tuple(float, int) or List(float, int)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        float, the same sigma is applied to every dimensions.
    allow_negative : bool
        Allow negative values after the filtering or clip them to 0.
    keep_dtype : bool
        Cast output image as input image. Integer output can't allow negative
        values.

    Returns
    -------
    image_filtered : np.ndarray, np.float
        Filtered image.

    """
    # check parameters
    check_array(image,
                ndim=[2, 3],
                dtype=[np.uint8, np.uint16, np.float32, np.float64])
    check_parameter(sigma=(float, int, tuple, list),
                    allow_negative=bool)

    # we cast the data in np.float to allow negative values
    image_float = None
    if image.dtype == np.uint8:
        image_float = cast_img_float32(image)
    elif image.dtype == np.uint16:
        image_float = cast_img_float64(image)

    # we apply gaussian filter
    image_filtered = gaussian(image_float, sigma=sigma)

    # we clip negative values to 0
    if not allow_negative:
        image_filtered = np.clip(image_filtered, a_min=0, a_max=None)

    # cast filtered image
    if keep_dtype and not allow_negative:
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
    image : np.ndarray, np.uint8
        Image to process with shape (y, x). Casting in np.uint8 makes the
        computation faster.
    kernel_shape : str
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).

    Returns
    -------
    image_without_back : np.ndarray, np.uint
        Image processed.

    """
    # check parameters
    check_array(image,
                ndim=2,
                dtype=[np.uint8])
    # TODO allow np.uint16 ?
    check_parameter(kernel_shape=str,
                    kernel_size=(int, tuple, list))

    # compute background noise with a large mean filter
    background = mean_filter(image,
                             kernel_shape=kernel_shape,
                             kernel_size=kernel_size)

    # subtract the background from the original image, clipping negative
    # values to 0
    mask = image > background
    image_without_back = np.subtract(image, background,
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
    sigma : float, int, Tuple(float, int) or List(float, int)
        Sigma used for the gaussian filter (one for each dimension). If it's a
        float, the same sigma is applied to every dimensions.

    Returns
    -------
    image_no_background : np.ndarray
        Image processed with shape (z, y, x) or (y, x).

    """
    # check parameters
    check_array(image,
                ndim=[2, 3],
                dtype=[np.uint8, np.uint16, np.float32, np.float64])
    check_parameter(sigma=(float, int, tuple, list))

    # apply a gaussian filter
    image_filtered = gaussian_filter(image, sigma,
                                     allow_negative=False,
                                     keep_dtype=True)

    # substract the gaussian filter
    out = np.zeros_like(image)
    image_no_background = np.subtract(image, image_filtered,
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
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # TODO check dtype
    # check parameters
    check_array(image,
                ndim=2,
                dtype=[np.uint8, np.uint16, bool])
    check_parameter(kernel_shape=(str, type(None)),
                    kernel_size=(int, tuple, list, type(None)))

    # get kernel
    if kernel_shape is None or kernel_size is None:
        kernel = None
    else:
        kernel = _define_kernel(shape=kernel_shape,
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
        Shape of the kernel used to compute the filter ('diamond', 'disk',
        'rectangle' or 'square').
    kernel_size : int or Tuple(int)
        The size of the kernel. For the rectangle we expect two integers
        (width, height).

    Returns
    -------
    image_filtered : np.ndarray, np.uint
        Filtered 2-d image with shape (y, x).

    """
    # TODO check dtype
    # check parameters
    check_array(image,
                ndim=2,
                dtype=[np.uint8, np.uint16, bool])
    check_parameter(kernel_shape=(str, type(None)),
                    kernel_size=(int, tuple, list, type(None)))

    # get kernel
    if kernel_shape is None or kernel_size is None:
        kernel = None
    else:
        kernel = _define_kernel(shape=kernel_shape,
                                size=kernel_size,
                                dtype=image.dtype)

    # apply filter
    if image.dtype == bool:
        image_filtered = binary_erosion(image, kernel)
    else:
        image_filtered = erosion(image, kernel)

    return image_filtered
