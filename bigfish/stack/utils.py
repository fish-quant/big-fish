# -*- coding: utf-8 -*-

"""
Utility functions for bigfish.stack submodule.
"""

import inspect
import re
import os
import copy

import numpy as np
import pandas as pd


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
# TODO fix the problem with _check_nan_array (too many calls, too slow)
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


# ### Recipe management (sanity checks, fitting) ###

def check_recipe(recipe, data_directory=None):
    """Check and validate a recipe.

    Checking a recipe consist in validating its filename pattern and the
    content of the dictionary.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        'pattern', 'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    data_directory : str
        Path of the directory with the files describes in the recipe. If it is
        provided, the function check that the files exist.

    Returns
    -------

    """
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

    # check that requested files exist
    if data_directory is not None:
        if not os.path.isdir(data_directory):
            raise ValueError("Directory does not exist: {0}"
                             .format(data_directory))
        recipe = fit_recipe(recipe)
        nb_r, nb_c, nb_z = get_nb_element_per_dimension(recipe)
        nb_fov = count_nb_fov(recipe)
        for fov in range(nb_fov):
            for r in range(nb_r):
                for c in range(nb_c):
                    for z in range(nb_z):
                        path = get_path_from_recipe(recipe, data_directory,
                                                    fov=fov, r=r, c=c, z=z)
                        if not os.path.isfile(path):
                            raise ValueError("File does not exist: {0}"
                                             .format(path))

    return


def fit_recipe(recipe):
    """Fit a recipe.

    Fitting a recipe consists in wrapping every values of 'fov', 'r', 'c' and
    'z' in a list (an empty one if necessary). Values for 'ext' and 'opt' are
    also initialized.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        'pattern', 'fov', 'r', 'c', 'z', 'ext' or 'opt'.

    Returns
    -------
    new_recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Contain the keys
        'pattern', 'fov', 'r', 'c', 'z', 'ext' and 'opt', initialized if
        necessary.

    """
    # initialize recipe
    new_recipe = copy.deepcopy(recipe)

    # initialize and fit the dimensions 'fov', 'r', 'c' and 'z'
    for key in ['fov', 'r', 'c', 'z']:
        if key not in new_recipe:
            new_recipe[key] = [None]
        value = new_recipe[key]
        if isinstance(value, str):
            new_recipe[key] = [value]

    # initialize the dimensions 'ext', 'opt'
    for key in ['ext', 'opt']:
        if key not in new_recipe:
            new_recipe[key] = ""

    return new_recipe


def get_path_from_recipe(recipe, input_folder, fov=0, r=0, c=0, z=0):
    """Build the path of a file from a recipe and the indices of specific
    elements.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        'pattern', 'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    input_folder : str
        Path of the folder containing the images.
    fov : int
        Index of the 'fov' element in the recipe to use in the filename.
    r : int
        Index of the 'r' element in the recipe to use in the filename.
    c : int
        Index of the 'c' element in the recipe to use in the filename.
    z : int
        Index of the 'z' element in the recipe to use in the filename.

    Returns
    -------
    path : str
        Path of the file to load.

    """
    # build a map of the elements' indices
    map_element_index = {"fov": fov, "r": r, "c": c, "z": z}

    # get filename pattern and decompose it
    recipe_pattern = recipe["pattern"]
    path_elements = re.findall("fov|r|c|z|ext|opt", recipe_pattern)
    path_separators = re.split("fov|r|c|z|ext|opt", recipe_pattern)

    # get filename recombining elements of the recipe
    filename = path_separators[0]  # usually an empty string
    for (element_name, separator) in zip(path_elements, path_separators[1:]):
        # if we need an element from a list of elements of the same dimension
        # (eg. to pick a specific channel 'c' among a list of channels)
        if element_name in map_element_index:
            element_index = map_element_index[element_name]
            element = recipe[element_name][element_index]
        # if this element is unique for all the recipe (eg. 'fov')
        else:
            element = recipe[element_name]
        # the filename is built ensuring the order of apparition of the
        # different morphemes and their separators
        filename += element
        filename += separator

    # get path
    path = os.path.join(input_folder, filename)

    return path


def get_nb_element_per_dimension(recipe):
    """Count the number of element to stack for each dimension ('r', 'c'
    and 'z').

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        'fov', 'r', 'c', 'z', 'ext' or 'opt'.

    Returns
    -------
    nb_r : int
        Number of rounds to be stacked.
    nb_c : int
        Number of channels to be stacked.
    nb_z : int
        Number of z layers to be stacked.

    """
    return len(recipe["r"]), len(recipe["c"]), len(recipe["z"])


def count_nb_fov(recipe):
    """Count the number of different fields of view that can be defined from
    the recipe.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        'pattern', 'fov', 'r', 'c', 'z', 'ext' or 'opt'.

    Returns
    -------
    nb_fov : int
        Number of different fields of view in the recipe.

    """
    # check recipe is a dictionary
    if not isinstance(recipe, dict):
        raise Exception("The recipe is not valid. It should be a dictionary.")

    # check the fov key exists
    if "fov" not in recipe:
        return 1

    # case where fov is directly a string
    elif isinstance(recipe["fov"], str):
        return 1

    # case where fov is a list of strings
    elif isinstance(recipe["fov"], list):
        return len(recipe["fov"])

    # non valid cases
    else:
        raise ValueError("'fov' should be a List or a str, not {0}"
                         .format(type(recipe["fov"])))


# ### Sanity checks parameters ###

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
            # TODO improve the error: raise 'Parameter array' when it comes from 'check_array'.
            raise ValueError("Parameter {0} should be cast in {1}. It is a {2}"
                             "instead."
                             .format(arg, expected_dtype, type(parameter)))

    return


# ### Others ###

def get_offset_value():
    """Return the margin pixel around a cell coordinate used to define its
    bounding box.

    Returns
    -------
    _ : int
        Margin value (in pixels).

    """
    # TODO rename it 'get_margin_value'
    # should be greater than 2 (maybe 1 is enough)
    return 5


def get_eps_float32():
    """Return the epsilon value for a 32 bit float.

    Returns
    -------
    _ : np.float32
        Epsilon value.

    """

    return np.finfo(np.float32).eps
