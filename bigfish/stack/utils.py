# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.stack subpackage.
"""

import os
import inspect
import hashlib

import numpy as np
import pandas as pd

from urllib.request import urlretrieve


# ### Sanity checks dataframe ###

def check_df(df, features=None, features_without_nan=None):
    """Full safety check of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Dataframe or Series to check.
    features : List[str]
        Names of the expected features.
    features_without_nan : List[str]
        Names of the features to check for the missing values

    Returns
    -------
    _ : bool
        Assert if the dataframe is well formatted.

    """
    # check parameters
    check_parameter(
        df=(pd.DataFrame, pd.Series),
        features=(list, type(None)),
        features_without_nan=(list, type(None)))

    # check features
    if features is not None:
        _check_features_df(df, features)

    # check NaN values
    if features_without_nan is not None:
        _check_features_df(df, features_without_nan)
        _check_nan_df(df, features_without_nan)

    return True


def _check_features_df(df, features):
    """Check that the dataframe contains expected features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check.
    features : List[str]
        Names of the expected features.

    """
    # check columns
    if not set(features).issubset(df.columns):
        raise ValueError("The dataframe does not seem to have the right "
                         "features. {0} instead of {1}"
                         .format(list(df.columns.values), features))


def _check_nan_df(df, features_to_check=None):
    """Check specific columns of the dataframe do not have any missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check.
    features_to_check : List[str]
        Names of the checked features.

    """
    # count NaN
    nan_count = df.isnull().sum()

    # for the full dataframe...
    if features_to_check is None:
        x = nan_count.sum()
        if x > 0:
            raise ValueError("The dataframe has {0} NaN values.".format(x))

    # ...or for some features
    else:
        nan_count = nan_count[features_to_check]
        x = nan_count.sum()
        if x > 0:
            raise ValueError("The dataframe has {0} NaN values for the "
                             "requested features: \n{1}.".format(x, nan_count))


# ### Sanity checks array ###

def check_array(array, ndim=None, dtype=None, allow_nan=True):
    """Full safety check of an array.

    Parameters
    ----------
    array : np.ndarray
        Array to check.
    ndim : int or List[int]
        Number of dimensions expected.
    dtype : type or List[type]
        Types expected.
    allow_nan : bool
        Allow NaN values or not.

    Returns
    -------
    _ : bool
        Assert if the array is well formatted.

    """
    # check parameters
    check_parameter(
        array=np.ndarray,
        ndim=(int, list, type(None)),
        dtype=(type, list, type(None)),
        allow_nan=bool)

    # check the dtype
    if dtype is not None:
        _check_dtype_array(array, dtype)

    # check the number of dimension
    if ndim is not None:
        _check_dim_array(array, ndim)

    # check NaN
    if not allow_nan:
        _check_nan_array(array)

    return True


def _check_dtype_array(array, dtype):
    """Check that a np.ndarray has the right dtype.

    Parameters
    ----------
    array : np.ndarray
        Array to check
    dtype : type or List[type]
        Type expected.

    """
    # enlist the dtype expected
    if isinstance(dtype, type):
        dtype = [dtype]

    # TODO simplify
    # check the dtype of the array
    error = True
    for dtype_expected in dtype:
        if array.dtype == dtype_expected:
            error = False
            break

    if error:
        raise TypeError("{0} is not supported yet. Use one of those dtypes "
                        "instead: {1}.".format(array.dtype, dtype))


def _check_dim_array(array, ndim):
    """Check that the array has the right number of dimensions.

    Parameters
    ----------
    array : np.ndarray
        Array to check.
    ndim : int or List[int]
        Number of dimensions expected

    """
    # enlist the number of expected dimensions
    if isinstance(ndim, int):
        ndim = [ndim]

    # check the number of dimensions of the array
    if array.ndim not in ndim:
        raise ValueError("Array can't have {0} dimension(s). Expected "
                         "dimensions are: {1}.".format(array.ndim, ndim))


def _check_nan_array(array):
    """Check that the array does not have missing values.

    Parameters
    ----------
    array : np.ndarray
        Array to check.

    """
    # count nan
    mask = np.isnan(array)
    x = mask.sum()

    # check the NaN values of the array
    if x > 0:
        raise ValueError("Array has {0} NaN values.".format(x))


def check_range_value(array, min_=None, max_=None):
    """Check the support of the array.

    Parameters
    ----------
    array : np.ndarray
        Array to check.
    min_ : int
        Minimum value allowed.
    max_ : int
        Maximum value allowed.

    Returns
    -------
    _ : bool
        Assert if the array has the right range of values.

    """
    # check lowest and highest bounds
    if min_ is not None and array.min() < min_:
        raise ValueError("The array should have a lower bound of {0}, but its "
                         "minimum value is {1}.".format(min_, array.min()))
    if max_ is not None and array.max() > max_:
        raise ValueError("The array should have an upper bound of {0}, but "
                         "its maximum value is {1}.".format(max_, array.max()))

    return True


# ### Sanity checks parameters ###

def check_parameter(**kwargs):
    """Check dtype of the function's parameters.

    Parameters
    ----------
    kwargs : Type or Tuple[Type]
        Map of each parameter with its expected dtype.

    Returns
    -------
    _ : bool
        Assert if the array is well formatted.

    """
    # get the frame and the parameters of the function
    frame = inspect.currentframe().f_back
    _, _, _, values = inspect.getargvalues(frame)

    # compare each parameter with its expected dtype
    for arg in kwargs:
        expected_dtype = kwargs[arg]
        parameter = values[arg]
        if not isinstance(parameter, expected_dtype):
            actual = "'{0}'".format(type(parameter).__name__)
            if isinstance(expected_dtype, tuple):
                target = ["'{0}'".format(x.__name__) for x in expected_dtype]
                target = "(" + ", ".join(target) + ")"
            else:
                target = expected_dtype.__name__
            raise TypeError("Parameter {0} should be a {1}. It is a {2} "
                            "instead.".format(arg, target, actual))

    return True


# ### Constants ###

def get_margin_value():
    """Return the margin pixel around a cell coordinate used to define its
    bounding box.

    Returns
    -------
    _ : int
        Margin value (in pixels).

    """
    # should be greater or equal to 2 (maybe 1 is enough)
    return 5


def get_eps_float32():
    """Return the epsilon value for a 32 bit float.

    Returns
    -------
    _ : np.float32
        Epsilon value.

    """
    return np.finfo(np.float32).eps


# ### Fetch data ###

def load_and_save_url(remote_url, directory, filename=None):
    """Download remote data and save them

    Parameters
    ----------
    remote_url : str
        Remote url of the data to download.
    directory : str
        Directory to save the download content.
    filename : str
        Filename of the object to save.

    Returns
    -------
    path : str
        Path of the downloaded file.

    """
    # check parameters
    check_parameter(
        remote_url=str,
        directory=str,
        filename=(str, type(None)))

    # get output path
    if filename is None:
        filename = remote_url.split("/")[-1]
    path = os.path.join(directory, filename)

    # download and save data
    urlretrieve(remote_url, path)

    return path


def check_hash(path, expected_hash):
    """Check hash value of a file.

    Parameters
    ----------
    path : str
        Path of the file to check.
    expected_hash : str
        Expected hash value.

    Returns
    -------
    _ : bool
        True if hash values match.

    """
    # check parameter
    check_parameter(
        path=str,
        expected_hash=str)

    # compute hash value
    hash_value = compute_hash(path)

    # compare checksum
    if hash_value != expected_hash:
        raise IOError("File {0} has an SHA256 checksum ({1}) differing from "
                      "expected ({2}). File may be corrupted."
                      .format(path, hash_value, expected_hash))

    return True


def compute_hash(path):
    """Compute sha256 hash of a file.

    Parameters
    ----------
    path : str
        Path to read the file.

    Returns
    -------
    sha256 : str
        Hash value of the file.

    """
    # check parameters
    check_parameter(path=str)

    # initialization
    sha256hash = hashlib.sha256()
    chunk_size = 8192

    # open and read file
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)

    # compute hash
    sha256 = sha256hash.hexdigest()

    return sha256


def check_input_data(input_directory, input_segmentation=False):
    """Check input images exists and download them if necessary.

    Parameters
    ----------
    input_directory : str
        Path of the image directory.
    input_segmentation : bool
        Check 2-d example images for segmentation.

    """
    # parameters
    filename_input_dapi = "experiment_1_dapi_fov_1.tif"
    url_input_dapi = "https://github.com/fish-quant/big-fish-examples/releases/download/data/experiment_1_dapi_fov_1.tif"
    hash_input_dapi = "3ce6dcfbece75da41326943432ada4cc9bacd06750e59dc2818bb253b6e7fdcd"
    filename_input_smfish = "experiment_1_smfish_fov_1.tif"
    url_input_smfish = "https://github.com/fish-quant/big-fish-examples/releases/download/data/experiment_1_smfish_fov_1.tif"
    hash_input_smfish = "bc6aec1f3da4c25f3c6b579c274584ce1e88112c7f980e5437b5ad5223bc8ff6"
    filename_input_nuc_full = "example_nuc_full.tif"
    url_input_nuc_full = "https://github.com/fish-quant/big-fish-examples/releases/download/data/example_nuc_full.tif"
    hash_input_nuc_full = "3bf70c7b5a02c60725baba3dfddff3010e0957de9ab78f0f65166248ead84ec4"
    filename_input_cell_full = "example_cell_full.tif"
    url_input_cell_full = "https://github.com/fish-quant/big-fish-examples/releases/download/data/example_cell_full.tif"
    hash_input_cell_full = "36981955ed97e9cab8a69241140a9aac3bdcf32dc157d6957fd37edcb16b34bd"

    # check if input dapi image exists
    path = os.path.join(input_directory, filename_input_dapi)
    if os.path.isfile(path):

        # check that image is not corrupted
        try:
            check_hash(path, hash_input_dapi)
            print("{0} is already in the directory"
                  .format(filename_input_dapi))

        # otherwise download it
        except IOError:
            print("{0} seems corrupted".format(filename_input_dapi))
            print("downloading {0}...".format(filename_input_dapi))
            load_and_save_url(url_input_dapi,
                              input_directory,
                              filename_input_dapi)
            check_hash(path, hash_input_dapi)

    # if file does not exist we directly download it
    else:
        print("downloading {0}...".format(filename_input_dapi))
        load_and_save_url(url_input_dapi,
                          input_directory,
                          filename_input_dapi)
        check_hash(path, hash_input_dapi)

    # check if input smfish image exists
    path = os.path.join(input_directory, filename_input_smfish)
    if os.path.isfile(path):

        # check that image is not corrupted
        try:
            check_hash(path, hash_input_smfish)
            print("{0} is already in the directory"
                  .format(filename_input_smfish))

        # otherwise download it
        except IOError:
            print("{0} seems corrupted".format(filename_input_smfish))
            print("downloading {0}...".format(filename_input_smfish))
            load_and_save_url(url_input_smfish,
                              input_directory,
                              filename_input_smfish)
            check_hash(path, hash_input_smfish)

    # if file does not exist we directly download it
    else:
        print("downloading {0}...".format(filename_input_smfish))
        load_and_save_url(url_input_smfish,
                          input_directory,
                          filename_input_smfish)
        check_hash(path, hash_input_smfish)

    # stop here or check segmentation examples
    if input_segmentation:

        # check if example nucleus exists
        path = os.path.join(input_directory, filename_input_nuc_full)
        if os.path.isfile(path):

            # check that image is not corrupted
            try:
                check_hash(path, hash_input_nuc_full)
                print("{0} is already in the directory"
                      .format(filename_input_nuc_full))

            # otherwise download it
            except IOError:
                print("{0} seems corrupted".format(filename_input_nuc_full))
                print("downloading {0}...".format(filename_input_nuc_full))
                load_and_save_url(url_input_nuc_full,
                                  input_directory,
                                  filename_input_nuc_full)
                check_hash(path, hash_input_nuc_full)

        # if file does not exist we directly download it
        else:
            print("downloading {0}...".format(filename_input_nuc_full))
            load_and_save_url(url_input_nuc_full,
                              input_directory,
                              filename_input_nuc_full)
            check_hash(path, hash_input_nuc_full)

        # check if example cell exists
        path = os.path.join(input_directory, filename_input_cell_full)
        if os.path.isfile(path):

            # check that image is not corrupted
            try:
                check_hash(path, hash_input_cell_full)
                print("{0} is already in the directory"
                      .format(filename_input_cell_full))

            # otherwise download it
            except IOError:
                print("{0} seems corrupted".format(filename_input_cell_full))
                print("downloading {0}...".format(filename_input_cell_full))
                load_and_save_url(url_input_cell_full,
                                  input_directory,
                                  filename_input_cell_full)
                check_hash(path, hash_input_cell_full)

        # if file does not exist we directly download it
        else:
            print("downloading {0}...".format(filename_input_cell_full))
            load_and_save_url(url_input_cell_full,
                              input_directory,
                              filename_input_cell_full)
            check_hash(path, hash_input_cell_full)


# ### Computation ###

def moving_average(array, n):
    """Compute a trailing average.

    Parameters
    ----------
    array : np.ndarray
        Array used to compute moving average.
    n : int
        Window width of the moving average.

    Returns
    -------
    results : np.ndarray
        Moving average values.

    """
    # check parameter
    check_parameter(n=int)
    check_array(array, ndim=1)

    # compute moving average
    cumsum = [0]
    results = []
    for i, x in enumerate(array, 1):
        cumsum.append(cumsum[i-1] + x)
        if i >= n:
            ma = (cumsum[i] - cumsum[i - n]) / n
            results.append(ma)
    results = np.array(results)

    return results


def centered_moving_average(array, n):
    """Compute a centered moving average.

    Parameters
    ----------
    array : np.ndarray
        Array used to compute moving average.
    n : int
        Window width of the moving average.

    Returns
    -------
    results : np.ndarray
        Centered moving average values.

    """
    # check parameter
    check_parameter(n=int)
    check_array(array, ndim=1)

    # pad array to keep the same length and centered the outcome
    if n % 2 == 0:
        r = int(n / 2)
        n += 1
    else:
        r = int((n - 1) / 2)
    array_padded = np.pad(array, pad_width=r, mode="reflect")

    # compute centered moving average
    results = moving_average(array_padded, n)

    return results
