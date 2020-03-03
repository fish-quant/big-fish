# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Function used to read data from various sources and store them in a
multidimensional tensor (numpy.ndarray).
"""

import mrc
import warnings

import numpy as np

from skimage import io
from .utils import check_array
from .utils import check_parameter


# ### Read ###

def read_image(path, sanity_check=False):
    """Read an image with the .png, .jpg, .jpeg, .tif or .tiff extension.

    Parameters
    ----------
    path : str
        Path of the image to read.
    sanity_check : bool
        Check if the array returned fit with bigfish pipeline.

    Returns
    -------
    image : ndarray, np.uint or np.int
        Image read.

    """
    # check path
    check_parameter(path=str,
                    sanity_check=bool)

    # read image
    image = io.imread(path)

    # check the output image
    if sanity_check:
        check_array(image,
                    dtype=[np.uint8, np.uint16, np.uint32,
                           np.int8, np.int16, np.int32,
                           np.float16, np.float32, np.float64,
                           bool],
                    ndim=[2, 3, 4, 5],
                    allow_nan=False)

    return image


def read_dv(path, sanity_check=False):
    """Read a video file with the .dv extension.

    Parameters
    ----------
    path : str
        Path of the file to read.
    sanity_check : bool
        Check if the array returned fit with bigfish pipeline.

    Returns
    -------
    video : ndarray, np.uint or np.int
        Video read.

    """
    # check path
    check_parameter(path=str,
                    sanity_check=bool)

    # read video file
    video = mrc.imread(path)

    # check the output video
    # metadata can be read running 'tensor.Mrc.info()'
    if sanity_check:
        check_array(video,
                    dtype=[np.uint16, np.int16, np.int32, np.float32],
                    allow_nan=False)

    return video


def read_array(path):
    """Read a numpy array with 'npy' extension.

    Parameters
    ----------
    path : str
        Path of the array to read.

    Returns
    -------
    array : ndarray
        Array read.

    """
    # check path
    check_parameter(path=str)
    if ".npy" not in path:
        path += ".npy"

    # read array file
    array = np.load(path)

    return array


def read_compressed(path, verbose=False):
    """Read a NpzFile object with 'npz' extension.

    Parameters
    ----------
    path : str
        Path of the file to read.
    verbose : bool
        Return names of the different compressed objects.

    Returns
    -------
    data : NpzFile object
        NpzFile read.

    """
    # check path
    check_parameter(path=str,
                    verbose=bool)
    if ".npz" not in path:
        path += ".npz"

    # read array file
    data = np.load(path)
    if verbose:
        print("Compressed objects: {0} \n".format(", ".join(data.files)))

    return data


# ### Write ###

def save_image(image, path, extension="tif"):
    """Save an image.

    The input image should have between 2 and 5 dimensions, with boolean,
    8-bit, 16-bit or 32-bit (unsigned) integer, 16-bit, 32-bit or 64-bit float.

    The dimensions should be in the following order: (round, channel, z, y, x).

    Additional notes:
    - If the image has more than 2 dimensions, 'tif' and 'tiff' extensions are
    required ('png' extension does not handle 3-d images other than (M, N, 3)
    or (M, N, 4) shapes).
    - A 2-d boolean image can be saved in 'png', 'jpg' or 'jpeg'
    (cast in np.uint8).
    - A multidimensional boolean image should be saved with
    bigfish.stack.save_array function or as a 0-1 images with 'tif'/'tiff'
    extension.

    Parameters
    ----------
    image : np.ndarray
        Image to save.
    path : str
        Path of the saved image.
    extension : str
        Default extension to save the image (among 'png', 'jpg', 'jpeg', 'tif'
        or 'tiff').

    Returns
    -------

    """
    # check image and parameters
    check_parameter(image=np.ndarray,
                    path=str,
                    extension=str)
    check_array(image,
                dtype=[np.uint8, np.uint16, np.uint32,
                       np.int8, np.int16, np.int32,
                       np.float16, np.float32, np.float64,
                       bool],
                ndim=[2, 3, 4, 5],
                allow_nan=False)

    # check extension and build path
    if "." in path:
        extension = path.split(".")[-1]
    else:
        path += ".{0}".format(extension)
    if extension not in ["png", "jpg", "jpeg", "tif", "tiff"]:
        raise ValueError("{0} extension is not supported, please choose among "
                         "'png', 'jpg', 'jpeg', 'tif' or 'tiff'."
                         .format(extension))

    # warn about extension
    if (extension in ["png", "jpg", "jpeg"] and len(image.shape) > 2
            and image.dtype != bool):
        raise ValueError("Extension {0} is not fitted with multidimensional "
                         "images. Use 'tif' or 'tiff' extension instead."
                         .format(extension))
    if (extension in ["png", "jpg", "jpeg"] and len(image.shape) == 2
            and image.dtype != bool):
        warnings.warn("Extension {0} is not consistent with dtype. To prevent "
                      "'image' from being cast you should use 'tif' or 'tiff' "
                      "extension.".format(extension), UserWarning)
    if (extension in ["png", "jpg", "jpeg"] and len(image.shape) == 2
            and image.dtype == bool):
        warnings.warn("Extension {0} is not consistent with dtype. To prevent "
                      "'image' from being cast you should use "
                      "'bigfish.stack.save_array' function instead."
                      .format(extension), UserWarning)
    if (extension in ["tif", "tiff"] and len(image.shape) == 2
            and image.dtype == bool):
        raise ValueError("Extension {0} is not fitted with boolean images. "
                         "Use 'png', 'jpg' or 'jpeg' extension instead."
                         .format(extension))
    if (extension in ["png", "jpg", "jpeg", "tif", "tiff"]
            and len(image.shape) > 2 and image.dtype == bool):
        raise ValueError("Extension {0} is not fitted with multidimensional "
                         "boolean images. Use 'bigfish.stack.save_array' "
                         "function instead.".format(extension))

    # save image without warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning)
        io.imsave(path, image)

    return


def save_array(array, path):
    """Save an array.

    The input array should have between 2 and 5 dimensions, with boolean,
    8-bit, 16-bit or 32-bit (unsigned) integer, 64-bit integer, 16-bit, 32-bit
    or 64-bit float.

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    path : str
        Path of the saved array.

    Returns
    -------

    """
    # check array and path
    check_parameter(array=np.ndarray,
                    path=str)
    check_array(array,
                dtype=[np.uint8, np.uint16, np.uint32,
                       np.int8, np.int16, np.int32, np.int64,
                       np.float16, np.float32, np.float64,
                       bool],
                ndim=[2, 3, 4, 5],
                allow_nan=True)
    if "." in path and "npy" not in path:
        path_ = path.split(".")[0]
        path = path_ + ".npy"

    # save array
    np.save(path, array)

    return
