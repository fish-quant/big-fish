# -*- coding: utf-8 -*-

"""
Utility functions for bigfish.stack submodule.
"""

import inspect
import re

import numpy as np
import pandas as pd

from skimage.draw import polygon_perimeter


# ### Sanity checks dataframe ###

def check_df(df, features=None, features_nan=None):
    """Full safety check of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check.
    features : List[str]
        Names of the expected features.
    features_nan : List[str]
        Names of the features to check for the missing values

    Returns
    -------
    _ : bool
        Assert if the dataframe is well formatted.

    """
    # check parameters
    check_parameter(features=(list, type(None)),
                    features_nan=(list, type(None)))

    # check the dataframe itself
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Data should be a pd.DataFrame instead of {0}."
                         .format(type(df)))

    # check features
    if features is not None:
        _check_features_df(df, features)

    # check NaN values
    if features_nan is not None:
        _check_features_df(df, features_nan)
        _check_nan_df(df, features_nan)

    # TODO complete the checks for the dataframe (dtype).

    return True


def _check_features_df(df, features):
    """Check that the dataframe contains expected features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check.
    features : List[str]
        Names of the expected features.

    Returns
    -------

    """
    # check columns
    if not set(features).issubset(df.columns):
        raise ValueError("The dataframe does not seem to have the right "
                         "features. {0} instead of {1}"
                         .format(df.columns, features))

    return


def _check_nan_df(df, features_nan=None):
    """

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check.
    features_nan : List[str]
        Names of the checked features.

    Returns
    -------

    """
    # count NaN
    nan_count = df.isnull().sum()

    # for the full dataframe...
    if features_nan is None:
        x = nan_count.sum()
        if x > 0:
            raise ValueError("The dataframe has {0} NaN values.".format(x))

    # ...or for some features
    else:
        nan_count = nan_count[features_nan]
        x = nan_count.sum()
        if x > 0:
            raise ValueError("The dataframe has {0} NaN values for the "
                             "requested features: \n{1}.".format(x, nan_count))

    return


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
    check_parameter(array=np.ndarray,
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


def _check_nan_array(array):
    """Check that the array does not have NaN values.

    Parameters
    ----------
    array : np.ndarray
        Array to check.

    Returns
    -------

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

def check_recipe(recipe):
    """Check and validate a recipe.

    Checking a recipe consist in validating its filename pattern and the
    content of the dictionary.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        'pattern', 'fov', 'r', 'c', 'z', 'ext' or 'opt'.

    Returns
    -------

    """
    # TODO check files exists
    # check recipe is a dictionary
    if not isinstance(recipe, dict):
        raise Exception("The recipe is not valid. It should be a dictionary.")

    # check the filename pattern
    if "pattern" not in recipe:
        raise ValueError("A recipe should have a filename pattern "
                         "('pattern' keyword).")
    recipe_pattern = recipe["pattern"]
    if not isinstance(recipe_pattern, str):
        raise ValueError("'pattern' should be a string, not a {0}."
                         .format(type(recipe_pattern)))

    # count the different dimensions to combinate in the recipe (among
    # 'fov', 'r', 'c' and 'z')
    dimensions = re.findall("fov|r|c|z", recipe_pattern)

    # each dimension can only appear once in the filename pattern
    if len(dimensions) != len(set(dimensions)):
        raise ValueError("The pattern used in recipe is wrong, a dimension "
                         "appears several times: {0}".format(recipe_pattern))

    # check keys and values of the recipe
    for key, value in recipe.items():
        if key not in ['fov', 'r', 'c', 'z', 'ext', 'opt', 'pattern']:
            raise ValueError("The recipe can only contain the keys 'fov', "
                             "'r', 'c', 'z', 'ext', 'opt' or 'pattern'. "
                             "Not '{0}'.".format(key))
        if not isinstance(value, (list, str)):
            raise TypeError("A recipe can only contain lists or strings, "
                            "not {0}.".format(type(value)))

    return


def check_parameter(**kwargs):
    """Check dtype of the function's parameters.

    Parameters
    ----------
    kwargs : dict
        Map of each parameter with its expected dtype.

    Returns
    -------

    """
    # get the frame and the parameters of the function
    frame = inspect.currentframe().f_back
    _, _, _, values = inspect.getargvalues(frame)

    # compare each parameter with its expected dtype
    for arg in kwargs:
        expected_dtype = kwargs[arg]
        parameter = values[arg]
        if not isinstance(parameter, expected_dtype):
            raise ValueError("Parameter {0} should be cast in {1}. It is a {2}"
                             "instead."
                             .format(arg, expected_dtype, type(parameter)))

    return


# ### Coordinate utilities ###

def complete_coordinates_2d(list_coord):
    """Complete a 2-d coordinates array, by generating/interpolating missing
    points.

    Parameters
    ----------
    list_coord : List[np.array]
        List of the coordinates arrays to complete, with shape (nb_points, 2).

    Returns
    -------

    """
    # check parameter
    check_parameter(list_coord=list)

    # for each array in the list, complete its coordinates using the scikit
    # image method 'polygon_perimeter'
    list_coord_completed = []
    for coord in list_coord:
        coord_x, coord_y = polygon_perimeter(coord[:, 0], coord[:, 1])
        coord_x = coord_x[:, np.newaxis]
        coord_y = coord_y[:, np.newaxis]
        new_coord = np.concatenate((coord_x, coord_y), axis=-1)
        list_coord_completed.append(new_coord)

    return list_coord_completed


def from_coord_to_image(coord, image_shape=None):
    """Convert an array of coordinates into a binary matrix.

    Parameters
    ----------
    coord : np.ndarray, np.uint64
        Array of coordinate with shape (nb_points, 2) or (nb_points, 3).
    image_shape:

    Returns
    -------
    image : np.ndarray, np.float32
        Binary matrix plotting the coordinates values.

    """
    # build matrices
    if image_shape is None:
        max_x = coord[:, 0].max() + 5
        max_y = coord[:, 1].max() + 5
        image_shape = (max_x, max_y)
    image = np.zeros(image_shape, dtype=np.float32)
    image[coord[:, 0], coord[:, 1]] = 1.0

    return image
