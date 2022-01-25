# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions used to format and clean any input loaded in bigfish.
"""

import warnings

import numpy as np

from .utils import check_array
from .utils import check_parameter
from .utils import check_range_value

from skimage import img_as_ubyte
from skimage import img_as_float32
from skimage import img_as_float64
from skimage import img_as_uint
from skimage.exposure import rescale_intensity
from skimage.transform import resize


# TODO replace 'tensor' by 'image'

# ### Image normalization ###

def compute_image_standardization(image):
    """Normalize image by computing its z score.

    Parameters
    ----------
    image : np.ndarray
        Image to normalize with shape (y, x).

    Returns
    -------
    normalized_image : np.ndarray
        Normalized image with shape (y, x).

    """
    # check parameters
    check_array(image, ndim=2, dtype=[np.uint8, np.uint16, np.float32])

    # check image is in 2D
    if len(image.shape) != 2:
        raise ValueError("'image' should be a 2-d array. Not {0}-d array"
                         .format(len(image.shape)))

    # compute mean and standard deviation
    m = np.mean(image)
    adjusted_stddev = max(np.std(image), 1.0 / np.sqrt(image.size))

    # normalize image
    normalized_image = (image - m) / adjusted_stddev

    return normalized_image


def rescale(tensor, channel_to_stretch=None, stretching_percentile=99.9):
    """Rescale tensor values up to its dtype range (unsigned/signed integers)
    or between 0 and 1 (float).

    Each round and each channel is rescaled independently. Tensor has between
    2 to 5 dimensions, in the following order: (round, channel, z, y, x).

    By default, we rescale the tensor intensity range to its dtype range (or
    between 0 and 1 for float tensor). We can improve the contrast by
    stretching a smaller range of pixel intensity: between the minimum value
    of a channel and percentile value of the channel (cf.
    ``stretching_percentile``).

    To be consistent with skimage, 64-bit (unsigned) integer images are not
    supported.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to rescale.
    channel_to_stretch : int, List[int] or Tuple[int]
        Channel to stretch. If None, minimum and maximum of each channel are
        used as the intensity range to rescale.
    stretching_percentile : float or int
        Percentile to determine the maximum intensity value used to rescale
        the image. If 1, the maximum pixel intensity is used to rescale the
        image.

    Returns
    -------
    tensor : np.ndarray
        Tensor rescaled.

    """
    # check parameters
    check_parameter(
        tensor=np.ndarray,
        channel_to_stretch=(int, list, tuple, type(None)),
        stretching_percentile=(int, float))
    check_array(
        tensor,
        ndim=[2, 3, 4, 5],
        dtype=[np.uint8, np.uint16, np.uint32,
               np.int8, np.int16, np.int32,
               np.float16, np.float32, np.float64])
    check_range_value(tensor, min_=0)

    # enlist 'channel_to_stretch' if necessary
    if channel_to_stretch is None:
        channel_to_stretch = []
    elif isinstance(channel_to_stretch, int):
        channel_to_stretch = [channel_to_stretch]

    # wrap tensor in 5-d if necessary
    tensor_5d, original_ndim = _wrap_5d(tensor)

    # rescale
    tensor_5d = _rescale_5d(
        tensor_5d,
        channel_to_stretch=channel_to_stretch,
        stretching_percentile=stretching_percentile)

    # rebuild the original tensor shape
    tensor = _unwrap_5d(tensor_5d, original_ndim)

    return tensor


def _wrap_5d(tensor):
    """Increases the number of dimensions of a tensor up to 5.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to wrap.

    Returns
    -------
    tensor_5d : np.ndarray
        Tensor with shape (round, channel, z, y, x).
    original_ndim : int
        Original number of dimensions.

    """
    # wrap tensor in 5-d if necessary
    original_ndim = tensor.ndim
    if original_ndim == 2:
        tensor_5d = tensor[np.newaxis, np.newaxis, np.newaxis, ...]
    elif original_ndim == 3:
        tensor_5d = tensor[np.newaxis, np.newaxis, ...]
    elif original_ndim == 4:
        tensor_5d = tensor[np.newaxis, ...]
    else:
        tensor_5d = tensor

    return tensor_5d, original_ndim


def _unwrap_5d(tensor_5d, original_ndim):
    """Remove useless dimensions from a 5-d tensor.

    Parameters
    ----------
    tensor_5d : np.ndarray
        Tensor with shape (round, channel, z, y, x).
    original_ndim : int
        Original number of dimensions.

    Returns
    -------
    tensor : np.ndarray
        Unwrapped tensor.

    """
    # rebuild the original tensor shape
    if original_ndim == 2:
        tensor = tensor_5d[0, 0, 0, :, :]
    elif original_ndim == 3:
        tensor = tensor_5d[0, 0, :, :, :]
    elif original_ndim == 4:
        tensor = tensor_5d[0, :, :, :, :]
    else:
        tensor = tensor_5d

    return tensor


def _rescale_5d(tensor, channel_to_stretch, stretching_percentile):
    """Rescale tensor values up to its dtype range (unsigned/signed integers)
    or between 0 and 1 (float).

    Each round and each channel is rescaled independently. Tensor has between
    2 to 5 dimensions, in the following order: (round, channel, z, y, x).

    By default, we rescale the tensor intensity range to its dtype range (or
    between 0 and 1 for float tensor). We can improve the contrast by
    stretching a smaller range of pixel intensity: between the minimum value
    of a channel and percentile value of the channel (cf.
    'stretching_percentile').

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to rescale.
    channel_to_stretch : int, List[int] or Tuple[int]
        Channel to stretch. If None, minimum and maximum of each channel are
        used as the intensity range to rescale.
    stretching_percentile : float
        Percentile to determine the maximum intensity value used to rescale
        the image. If 1, the maximum pixel intensity is used to rescale the
        image.

    Returns
    -------
    tensor : np.ndarray
        Tensor rescaled.

    """
    # target intensity range
    target_range = 'dtype'
    if tensor.dtype in [np.float16, np.float32, np.float64]:
        target_range = (0, 1)

    # rescale each round independently
    rounds = []
    for r in range(tensor.shape[0]):

        # rescale each channel independently
        channels = []
        for c in range(tensor.shape[1]):

            # get channel
            channel = tensor[r, c, :, :, :]

            # rescale channel
            if c in channel_to_stretch:
                pa, pb = np.percentile(channel, (0, stretching_percentile))
                channel_rescaled = rescale_intensity(
                    channel,
                    in_range=(pa, pb),
                    out_range=target_range)
            else:
                channel_rescaled = rescale_intensity(
                    channel,
                    out_range=target_range)
            channels.append(channel_rescaled)

        # stack channels
        tensor_4d = np.stack(channels, axis=0)
        rounds.append(tensor_4d)

    # stack rounds
    tensor_5d = np.stack(rounds, axis=0)

    return tensor_5d


def cast_img_uint8(tensor, catch_warning=False):
    """Cast the image in np.uint8 and scale values between 0 and 255.

    Negative values are not allowed as the skimage method ``img_as_ubyte``
    would clip them to 0. Positives values are scaled between 0 and 255,
    excepted if they fit directly in 8 bit (in this case values are not
    modified).

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.
    catch_warning : bool
        Catch and ignore `UserWarning` about possible precision or sign loss.

    Returns
    -------
    tensor : np.ndarray, np.uint8
        Image cast.

    """
    # check tensor dtype
    check_array(
        tensor,
        ndim=[2, 3, 4, 5],
        dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
               np.int8, np.int16, np.int32, np.int64,
               np.float16, np.float32, np.float64])
    if tensor.dtype in [np.float16, np.float32, np.float64]:
        check_range_value(tensor, min_=0, max_=1)
    elif tensor.dtype in [np.int8, np.int16, np.int32, np.int64]:
        check_range_value(tensor, min_=0)

    if tensor.dtype == np.uint8:
        return tensor

    if (tensor.dtype in [np.uint16, np.uint32, np.uint64,
                         np.int16, np.int32, np.int64]
            and tensor.max() <= 255):
        raise ValueError("Tensor values are between {0} and {1}. It fits in 8 "
                         "bits and won't be scaled between 0 and 255. Use "
                         "'tensor.astype(np.uint8)' instead."
                         .format(tensor.min(), tensor.max()))

    # cast tensor
    if catch_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor = img_as_ubyte(tensor)
    else:
        tensor = img_as_ubyte(tensor)

    return tensor


def cast_img_uint16(tensor, catch_warning=False):
    """Cast the data in np.uint16.

    Negative values are not allowed as the skimage method ``img_as_uint`` would
    clip them to 0. Positives values are scaled between 0 and 65535, excepted
    if they fit directly in 16 bit (in this case values are not modified).

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.
    catch_warning : bool
        Catch and ignore `UserWarning` about possible precision or sign loss.
    Returns
    -------
    tensor : np.ndarray, np.uint16
        Image cast.

    """
    # check tensor dtype
    check_array(
        tensor,
        ndim=[2, 3, 4, 5],
        dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
               np.int8, np.int16, np.int32, np.int64,
               np.float16, np.float32, np.float64])
    if tensor.dtype in [np.float16, np.float32, np.float64]:
        check_range_value(tensor, min_=0, max_=1)
    elif tensor.dtype in [np.int8, np.int16, np.int32, np.int64]:
        check_range_value(tensor, min_=0)

    if tensor.dtype == np.uint16:
        return tensor

    if (tensor.dtype in [np.uint32, np.uint64, np.int32, np.int64]
            and tensor.max() <= 65535):
        raise ValueError("Tensor values are between {0} and {1}. It fits in "
                         "16 bits and won't be scaled between 0 and 65535. "
                         "Use 'tensor.astype(np.uint16)' instead."
                         .format(tensor.min(), tensor.max()))

    # cast tensor
    if catch_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor = img_as_uint(tensor)
    else:
        tensor = img_as_uint(tensor)

    return tensor


def cast_img_float32(tensor, catch_warning=False):
    """Cast the data in np.float32.

    If the input data is in (unsigned) integer, the values are scaled between
    0 and 1. When converting from a np.float dtype, values are not modified.

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.
    catch_warning : bool
        Catch and ignore `UserWarning` about possible precision or sign loss.

    Returns
    -------
    tensor : np.ndarray, np.float32
        image cast.

    """
    # check tensor dtype
    check_array(
        tensor,
        ndim=[2, 3, 4, 5],
        dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
               np.int8, np.int16, np.int32, np.int64,
               np.float16, np.float32, np.float64])

    # cast tensor
    if catch_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor = img_as_float32(tensor)
    else:
        tensor = img_as_float32(tensor)

    return tensor


def cast_img_float64(tensor):
    """Cast the data in np.float64.

    If the input data is in (unsigned) integer, the values are scaled between
    0 and 1. When converting from a np.float dtype, values are not modified.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to cast.

    Returns
    -------
    tensor : np.ndarray, np.float64
        Tensor cast.

    """
    # check tensor dtype
    check_array(
        tensor,
        ndim=[2, 3, 4, 5],
        dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
               np.int8, np.int16, np.int32, np.int64,
               np.float16, np.float32, np.float64])

    # cast tensor
    tensor = img_as_float64(tensor)

    return tensor


# ### Format and crop images ###

def resize_image(image, output_shape, method="bilinear"):
    """Resize an image with bilinear interpolation or nearest neighbor method.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    output_shape : Tuple[int]
        Shape of the resized image.
    method : str
        Interpolation method to use.

    Returns
    -------
    image_resized : np.ndarray
        Resized image.

    """
    # check parameters
    check_parameter(output_shape=tuple, method=str)
    check_array(image, ndim=[2, 3], dtype=[np.uint8, np.uint16, np.float32])

    # resize image
    if method == "bilinear":
        image_resized = resize(
            image,
            output_shape,
            mode="reflect",
            preserve_range=True,
            order=1,
            anti_aliasing=True)
    elif method == "nearest":
        image_resized = resize(
            image,
            output_shape,
            mode="reflect",
            preserve_range=True,
            order=0,
            anti_aliasing=False)
    else:
        raise ValueError("Method {0} is not available. Choose between "
                         "'bilinear' or 'nearest' instead.".format(method))

    # cast output dtype
    image_resized = image_resized.astype(image.dtype)

    return image_resized


def get_marge_padding(height, width, x):
    """Pad image to make its shape a multiple of `x`.

    Parameters
    ----------
    height : int
        Original height of the image.
    width : int
        Original width of the image.
    x : int
        Padded image have a `height` and `width` multiple of `x`.

    Returns
    -------
    marge_padding : List[List]
        List of lists with the format
        [[`marge_height_t`, `marge_height_b`], [`marge_width_l`,
        `marge_width_r`]].

    """
    # check parameters
    check_parameter(height=int, width=int, x=int)

    # pad height and width to make it multiple of x
    marge_sup_height = x - (height % x)
    marge_sup_height_l = int(marge_sup_height / 2)
    marge_sup_height_r = marge_sup_height - marge_sup_height_l
    marge_sup_width = x - (width % x)
    marge_sup_width_l = int(marge_sup_width / 2)
    marge_sup_width_r = marge_sup_width - marge_sup_width_l
    marge_padding = [[marge_sup_height_l, marge_sup_height_r],
                     [marge_sup_width_l, marge_sup_width_r]]

    return marge_padding
