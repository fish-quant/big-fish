# -*- coding: utf-8 -*-

"""
Utility functions for bigfish.stack submodule.
"""

import numpy as np


# TODO complete the checks for the dataframe (dtype, missing values).
# ### Sanity checks ###

def check_features_df(df, features):
    """Check that the dataframe has the right features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check.
    features : List[str]
        Names of the features expected.

    Returns
    -------

    """
    # get dataframe's features
    col_names = df.columns

    # sort the two lists
    col_names = sorted(col_names)
    features = sorted(features)

    if col_names == features:
        return
    else:
        raise ValueError("The file does not seem to have the right features. "
                         "{0} instead of {1}".format(col_names, features))


def check_array(array, ndim=None, dtype=None):
    """Full safety check of an array.

    Parameters
    ----------
    array : np.ndarray
        Array to check.
    ndim : int or List[int]
        Number of dimensions expected.
    dtype : type or List[type]
        Types expected.

    Returns
    -------

    """
    # check the array itself
    if not isinstance(array, np.ndarray):
        raise ValueError("Data should be a np.ndarray instead of {0}."
                         .format(type(array)))

    # check the dtype
    if dtype is not None:
        _check_dtype_array(array, dtype)

    # check the number of dimension
    if ndim is not None:
        _check_dim_array(array, ndim)

    # TODO check the order of the dimensions

    # TODO check nan

    return


def _check_dtype_array(array, dtype):
    """Check that a np.ndarray has the right dtype.

    Parameters
    ----------
    array : np.ndarray
        Array to check
    dtype : type or List[type]
        Type expected.

    Returns
    -------

    """
    # enlist the dtype expected
    if isinstance(dtype, type):
        dtype = [dtype]

    # check the dtype of the array
    for dtype_expected in dtype:
        if array.dtype == dtype_expected:
            return
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

    Returns
    -------

    """

    # enlist the number of expected dimensions
    if isinstance(ndim, int):
        ndim = [ndim]

    # check the number of dimensions of the array
    if array.ndim not in ndim:
        raise ValueError("Array can't have {0} dimension(s). Expected "
                         "dimensions are: {1}.".format(array.ndim, ndim))


def check_range_value(array, min_, max_):
    """

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
        Assert if the array is within the requested bound.

    """
    if array.min() < min_ or array.max() > max_:
        return False
    else:
        return True
