# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions used to read data from various sources and store them in a
multidimensional tensor (numpy.ndarray).
"""

import mrc
import warnings

import numpy as np
import pandas as pd

from skimage import io
from .utils import check_array
from .utils import check_parameter


# TODO add general read function with mime types
# TODO saving data in csv does not preserve dtypes

# ### Read ###

def read_image(path, sanity_check=False):
    """Read an image with ``png``, ``jpg``, ``jpeg``, ``tif`` or ``tiff``
    extension.

    Parameters
    ----------
    path : str
        Path of the image to read.
    sanity_check : bool
        Check if the array returned fits with bigfish pipeline.

    Returns
    -------
    image : ndarray, np.uint or np.int
        Image read.

    """
    # check path
    check_parameter(
        path=str,
        sanity_check=bool)

    # read image
    image = io.imread(path)

    # check the output image
    if sanity_check:
        check_array(
            image,
            dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
                   np.int8, np.int16, np.int32, np.int64,
                   np.float16, np.float32, np.float64,
                   bool],
            ndim=[2, 3, 4, 5],
            allow_nan=False)

    return image


def read_dv(path, sanity_check=False):
    """Read a video file with ``dv`` extension.

    Parameters
    ----------
    path : str
        Path of the file to read.
    sanity_check : bool
        Check if the array returned fits with bigfish pipeline.

    Returns
    -------
    video : ndarray
        Video read.

    """
    # check path
    check_parameter(
        path=str,
        sanity_check=bool)

    # read video file
    video = mrc.imread(path)

    # check the output video
    # metadata can be read running 'tensor.mrc.info()'
    if sanity_check:
        check_array(
            video,
            dtype=[np.uint16, np.int16, np.int32, np.float32],
            allow_nan=False)

    return video


def read_array(path):
    """Read a numpy array with ``npy`` extension.

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

    # read array file
    array = np.load(path)

    return array


def read_array_from_csv(path, dtype=None, delimiter=";", encoding="utf-8"):
    """Read a numpy array saved in a ``csv`` file.

    Parameters
    ----------
    path : str
        Path of the ``csv`` file to read.
    dtype : type or None
        Expected dtype to cast the array.
    delimiter : str
        Delimiter used to separate columns.
    encoding : str
        Encoding to use.

    Returns
    -------
    array : ndarray
        Array read.

    """
    # check parameters
    check_parameter(
        path=str,
        dtype=(type, type(None)),
        delimiter=str,
        encoding=str)

    # read csv file
    array = np.loadtxt(path, delimiter=delimiter, encoding=encoding)

    # cast array dtype
    if dtype is not None:
        array = array.astype(dtype)

    return array


def read_dataframe_from_csv(path, delimiter=";", encoding="utf-8"):
    """Read a numpy array or a pandas object saved in a ``csv`` file.

    Parameters
    ----------
    path : str
        Path of the ``csv`` file to read.
    delimiter : str
        Delimiter used to separate columns.
    encoding : str
        Encoding to use.

    Returns
    -------
    df : pd.DataFrame
        Pandas object read.

    """
    # check parameters
    check_parameter(
        path=str,
        delimiter=str,
        encoding=str)

    # read csv file
    df = pd.read_csv(path, sep=delimiter, encoding=encoding)

    return df


def read_uncompressed(path, verbose=False):
    """Read a NpzFile object with ``npz`` extension.

    Parameters
    ----------
    path : str
        Path of the file to read.
    verbose : bool
        Return names of the different objects.

    Returns
    -------
    data : NpzFile object
        NpzFile read.

    """
    # check parameters
    check_parameter(
        path=str,
        verbose=bool)

    # read array file
    data = np.load(path)
    if verbose:
        print("Available keys: {0} \n".format(", ".join(data.files)))

    return data


def read_cell_extracted(path, verbose=False):
    """Read a NpzFile object with ``npz`` extension, previously written with
    :func:`bigfish.stack.save_cell_extracted`.

    Parameters
    ----------
    path : str
        Path of the file to read.
    verbose : bool
        Return names of the different objects.

    Returns
    -------
    cell_results : Dict
        Dictionary including information about the cell (image, masks,
        coordinates arrays). Minimal information are :

        * `cell_id`: Unique id of the cell.
        * `bbox`: bounding box coordinates with the order (`min_y`, `min_x`,
          `max_y`, `max_x`).
        * `cell_coord`: boundary coordinates of the cell.
        * `cell_mask`: mask of the cell.

    """
    # read compressed file
    data = read_uncompressed(path, verbose)

    # store data in a dictionary
    cell_results = {}
    for key in data.files:
        cell_results[key] = data[key]

    return cell_results


# ### Write ###

def save_image(image, path, extension="tif"):
    """Save an image.

    The input image should have between 2 and 5 dimensions, with boolean,
    (unsigned) integer, or float.

    The dimensions should be in the following order: (round, channel, z, y, x).

    Parameters
    ----------
    image : np.ndarray
        Image to save.
    path : str
        Path of the saved image.
    extension : str
        Default extension to save the image (among ``png``, ``jpg``, ``jpeg``,
        ``tif`` or ``tiff``).

    Notes
    -----
    * If the image has more than 2 dimensions, ``tif`` and ``tiff`` extensions
      are required (``png`` extension does not handle 3-d images other than
      (M, N, 3) or (M, N, 4) shapes).
    * A 2-d boolean image can be saved in ``png``, ``jpg`` or ``jpeg`` (cast in
      np.uint8).
    * A multidimensional boolean image should be saved with
      :func:`bigfish.stack.save_array` or as a boolean images with ``tif``/
      ``tiff`` extension.

    """
    # check image and parameters
    check_parameter(
        path=str,
        extension=str)
    check_array(
        image,
        dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
               np.int8, np.int16, np.int32, np.int64,
               np.float16, np.float32, np.float64,
               bool],
        ndim=[2, 3, 4, 5],
        allow_nan=False)

    # check extension and build path
    if "/" in path:
        path_ = path.split("/")
        filename = path_[-1]

        if "." in filename:
            extension = filename.split(".")[-1]
        else:
            filename += ".{0}".format(extension)
        path_[-1] = filename
        path = "/".join(path_)
    else:
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


def save_array(array, path):
    """Save an array in a ``npy`` extension file.

    The input array should have between 2 and 5 dimensions, with boolean,
    (unsigned) integer, or float.

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    path : str
        Path of the saved array.

    """
    # check array and path
    check_parameter(path=str)
    check_array(
        array,
        dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
               np.int8, np.int16, np.int32, np.int64,
               np.float16, np.float32, np.float64,
               bool],
        ndim=[2, 3, 4, 5])

    # add extension if necessary
    if ".npy" not in path:
        path += ".npy"

    # save array
    np.save(path, array)


def save_data_to_csv(data, path, delimiter=";"):
    """Save a numpy array or a pandas object into a ``csv`` file.

    The input should be a pandas object (`Series` or `DataFrame`) or a
    numpy array with 2 dimensions and (unsigned) integer or float.

    Parameters
    ----------
    data : np.ndarray, pd.Series or pd.DataFrame
        Data to save.
    path : str
        Path of the saved ``csv`` file.
    delimiter : str
        Delimiter used to separate columns.

    """
    # check parameters
    check_parameter(
        data=(pd.DataFrame, pd.Series, np.ndarray),
        path=str,
        delimiter=str)

    # add extension if necessary
    if ".csv" not in path:
        path += ".csv"

    # save numpy ndarray in a csv file
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        check_array(
            data,
            dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
                   np.int8, np.int16, np.int32, np.int64,
                   np.float16, np.float32, np.float64],
            ndim=2)

        if data.dtype == np.float16:
            fmt = "%.4f"
        elif data.dtype == np.float32:
            fmt = "%.7f"
        elif data.dtype == np.float64:
            fmt = "%.16f"
        else:
            fmt = "%.1i"
        np.savetxt(path, data, fmt=fmt, delimiter=delimiter, encoding="utf-8")

    # save pandas object in a csv file
    elif isinstance(data, pd.Series):
        data = data.to_frame()
        data.to_csv(
            path,
            sep=delimiter,
            header=True,
            index=False,
            encoding="utf-8")
    else:
        data.to_csv(
            path,
            sep=delimiter,
            header=True,
            index=False,
            encoding="utf-8")


def save_cell_extracted(cell_results, path):
    """Save cell-level results from :func:`bigfish.stack.extract_cell` in a
    NpzFile object with ``npz`` extension.

    Parameters
    ----------
    cell_results : Dict
        Dictionary including information about the cell (image, masks,
        coordinates arrays). Minimal information are:

        * `cell_id`: Unique id of the cell.
        * `bbox`: bounding box coordinates with the order (`min_y`, `min_x`,
          `max_y`, `max_x`).
        * `cell_coord`: boundary coordinates of the cell.
        * `cell_mask`: mask of the cell.
    path : str
        Path of the saved array.

    """
    # check parameters
    check_parameter(
        cell_results=dict,
        path=str)

    # add extension if necessary
    if ".npz" not in path:
        path += ".npz"

    # save compressed file
    np.savez(path, **cell_results)
