# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.stack subpackage.
"""

import os
import re
import copy
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
    check_parameter(df=(pd.DataFrame, pd.Series),
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


# ### Recipe management (sanity checks, fitting) ###

def check_recipe(recipe, data_directory=None):
    """Check and validate a recipe.

    Checking a recipe consists in validating its filename pattern and the
    content of the dictionary.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        `pattern`, `fov`, `r`, `c`, `z`, `ext` or `opt`.
    data_directory : str
        Path of the directory with the files describes in the recipe. If it is
        provided, the function check that the files exist.

    Returns
    -------
    _ : bool
        Assert if the recipe is well formatted.

    """
    # check parameters
    check_parameter(recipe=dict,
                    data_directory=(str, type(None)))

    # check the filename pattern
    if "pattern" not in recipe:
        raise KeyError("A recipe should have a filename pattern "
                       "('pattern' keyword).")
    recipe_pattern = recipe["pattern"]
    if not isinstance(recipe_pattern, str):
        raise TypeError("'pattern' should be a string, not a {0}."
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
            raise KeyError("The recipe can only contain the keys 'fov', 'r', "
                           "'c', 'z', 'ext', 'opt' or 'pattern'. Not '{0}'."
                           .format(key))
        if not isinstance(value, (list, str)):
            raise TypeError("A recipe can only contain lists or strings, "
                            "not {0}.".format(type(value)))

    # check that requested files exist
    if data_directory is not None:
        if not os.path.isdir(data_directory):
            raise NotADirectoryError("Directory does not exist: {0}"
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
                            raise FileNotFoundError("File does not exist: {0}"
                                                    .format(path))

    return True


def fit_recipe(recipe):
    """Fit a recipe.

    Fitting a recipe consists in wrapping every values of `fov`, `r`, `c` and
    `z` in a list (an empty one if necessary). Values for `ext` and `opt` are
    also initialized.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        `pattern`, `fov`, `r`, `c`, `z`, `ext` or `opt`.

    Returns
    -------
    new_recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Contain the keys
        `pattern`, `fov`, `r`, `c`, `z`, `ext` and `opt`, initialized if
        necessary.

    """
    # check parameters
    check_parameter(recipe=dict)

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


def _is_recipe_fitted(recipe):
    """Check if a recipe is ready to be used.

    Fitting a recipe consists in wrapping every values of `fov`, `r`, `c` and
    `z` in a list (an empty one if necessary). Values for `ext` and `opt` are
    also initialized.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        `pattern`, `fov`, `r`, `c`, `z`, `ext` or `opt`.

    Returns
    -------
    _ : bool
        Indicates if the recipe is fitted or not

    """
    # all keys should be initialized in the new recipe, with a list or a string
    for key in ['fov', 'r', 'c', 'z']:
        if key not in recipe or not isinstance(recipe[key], list):
            return False
    for key in ['ext', 'opt']:
        if key not in recipe or not isinstance(recipe[key], str):
            return False
    if 'pattern' not in recipe or not isinstance(recipe['pattern'], str):
        return False

    return True


def get_path_from_recipe(recipe, input_folder, fov=0, r=0, c=0, z=0):
    """Build the path of a file from a recipe and the indices of specific
    elements.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        `pattern`, `fov`, `r`, `c`, `z`, `ext` or `opt`.
    input_folder : str
        Path of the folder containing the images.
    fov : int
        Index of the `fov` element in the recipe to use in the filename.
    r : int
        Index of the `r` element in the recipe to use in the filename.
    c : int
        Index of the `c` element in the recipe to use in the filename.
    z : int
        Index of the `z` element in the recipe to use in the filename.

    Returns
    -------
    path : str
        Path of the file to load.

    """
    # check parameters
    check_parameter(recipe=dict,
                    input_folder=str,
                    fov=int,
                    r=int,
                    c=int,
                    z=int)

    # check if the recipe is fitted
    if not _is_recipe_fitted(recipe):
        recipe = fit_recipe(recipe)

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
    """Count the number of element to stack for each dimension (`r`, `c`
    and `z`).

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        `fov`, `r`, `c`, `z`, `ext` or `opt`.

    Returns
    -------
    nb_r : int
        Number of rounds to be stacked.
    nb_c : int
        Number of channels to be stacked.
    nb_z : int
        Number of z layers to be stacked.

    """
    # check parameters
    check_parameter(recipe=dict)

    # check if the recipe is fitted
    if not _is_recipe_fitted(recipe):
        recipe = fit_recipe(recipe)

    return len(recipe["r"]), len(recipe["c"]), len(recipe["z"])


def count_nb_fov(recipe):
    """Count the number of different fields of view that can be defined from
    the recipe.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        `pattern`, `fov`, `r`, `c`, `z`, `ext` or `opt`.

    Returns
    -------
    nb_fov : int
        Number of different fields of view in the recipe.

    """
    # check parameters
    check_parameter(recipe=dict)

    # check if the recipe is fitted
    if not _is_recipe_fitted(recipe):
        recipe = fit_recipe(recipe)

    # a good recipe should have a list in the 'fov' key
    if not isinstance(recipe["fov"], list):
        raise TypeError("'fov' should be a List or a str, not {0}"
                        .format(type(recipe["fov"])))
    else:
        return len(recipe["fov"])


def check_datamap(data_map):
    """Check and validate a data map.

    Checking a data map consists in validating the recipe-folder pairs.

    Parameters
    ----------
    data_map : List[tuple]
        Map between input directories and recipes.

    Returns
    -------
    _ : bool
        Assert if the data map is well formatted.

    """
    check_parameter(data_map=list)
    for pair in data_map:
        if not isinstance(pair, (tuple, list)):
            raise TypeError("A data map is a list with tuples or lists. "
                            "Not {0}".format(type(pair)))
        if len(pair) != 2:
            raise ValueError("Elements of a data map are tuples or lists that "
                             "map a recipe (dict) to an input directory "
                             "(string). Here {0} elements are given {1}"
                             .format(len(pair), pair))
        (recipe, input_folder) = pair
        if not isinstance(input_folder, str):
            raise TypeError("A data map map a recipe (dict) to an input "
                            "directory (string). Not ({0}, {1})"
                            .format(type(recipe), type(input_folder)))
        check_recipe(recipe, data_directory=input_folder)

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
    check_parameter(remote_url=str,
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
    check_parameter(path=str,
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


# ### Spot utilities ###

def get_sigma(voxel_size_z=None, voxel_size_yx=100, psf_z=None, psf_yx=200):
    """Compute the standard deviation of the PSF of the spots.

    Parameters
    ----------
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we consider
        a 2-d PSF.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan, in
        nanometer. If None, we consider a 2-d PSF.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan, in
        nanometer.

    Returns
    -------
    sigma : Tuple[float]
        Standard deviations in pixel of the PSF, one element per dimension.

    """
    # check parameters
    check_parameter(voxel_size_z=(int, float, type(None)),
                    voxel_size_yx=(int, float),
                    psf_z=(int, float, type(None)),
                    psf_yx=(int, float))

    # compute sigma
    sigma_yx = psf_yx / voxel_size_yx

    if voxel_size_z is None or psf_z is None:
        return sigma_yx, sigma_yx

    else:
        sigma_z = psf_z / voxel_size_z
        return sigma_z, sigma_yx, sigma_yx


def get_radius(voxel_size_z=None, voxel_size_yx=100, psf_z=None, psf_yx=200):
    """Approximate the radius of the detected spot.

    We use the formula:

    .. math::

        \\mbox{radius} = \\mbox{sqrt(ndim)} * \\sigma

    with :math:`\\mbox{ndim}` the number of dimension of the image and
    :math:`\\sigma` the standard deviation (in pixel) of the detected spot.

    Parameters
    ----------
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we consider
        a 2-d spot.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan, in
        nanometer. If None, we consider a 2-d spot.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan, in
        nanometer.

    Returns
    -------
    radius : Tuple[float]
        Radius in pixels of the detected spots, one element per dimension.

    """
    # compute sigma
    sigma = get_sigma(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

    # compute radius
    radius = [np.sqrt(len(sigma)) * sigma_ for sigma_ in sigma]
    radius = tuple(radius)

    return radius
