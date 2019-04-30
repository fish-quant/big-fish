# -*- coding: utf-8 -*-

"""
Functions used to format and clean any input loaded in bigfish.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

from .loader import read_image, read_cell_json, read_rna_json
from .utils import (check_array, check_parameter, check_recipe,
                    check_range_value)

from sklearn.preprocessing import LabelEncoder

from skimage import img_as_ubyte, img_as_float32, img_as_float64, img_as_uint
from skimage.exposure import rescale_intensity

from scipy.sparse import coo_matrix

from scipy import ndimage as ndi


# TODO add safety checks
# TODO add a stack builder without recipe

# ### Simulated data ###

def build_simulated_dataset(path_cell, path_rna, path_output=None):
    """Build a dataset from the simulated coordinates of the nucleus, the
    cytoplasm and the RNA.

    Parameters
    ----------
    path_cell : str
        Path of the json file with the 2D nucleus and cytoplasm coordinates
        used by FishQuant to simulate the data.
    path_rna : str
        Path of the json file with the 3D RNA localization simulated by
        FishQuant. If it is the path of a folder, all its json files will be
        aggregated.
    path_output : str
        Path of the output file with the merged dataset. The final dataframe is
        serialized and store in a pickle file.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with all the simulated cells, the coordinates of their
        different elements and the localization pattern used to simulate them.
    df_cell : pandas.DataFrame
        Dataframe with the 2D coordinates of the nucleus and the cytoplasm of
        actual cells used to simulate data.
    df_rna : pandas.DataFrame
        Dataframe with 3D coordinates of the simulated RNA, localization
        pattern used to simulate them and its strength.

    """
    # TODO this function should be updated as soon as we change the simulation
    #  framework
    # check parameters
    check_parameter(path_cell=str, path_rna=str, path_output=(str, type(None)))

    # read the cell data (nucleus + cytoplasm)
    df_cell = read_cell_json(path_cell)

    # read the RNA data
    if os.path.isdir(path_rna):
        # we concatenate all the json file in the folder
        simulations = []
        for filename in os.listdir(path_rna):
            if ".json" in filename:
                path = os.path.join(path_rna, filename)
                df_ = read_rna_json(path)
                simulations.append(df_)
        df_rna = pd.concat(simulations)
        df_rna.reset_index(drop=True, inplace=True)

    else:
        # we directly read the json file
        df_rna = read_rna_json(path_rna)

    # merge the dataframe
    df = pd.merge(df_rna, df_cell, on="name_img_BGD")

    # save output
    if path_output is not None:
        df.to_pickle(path_output)

    return df, df_cell, df_rna


# ### Real data ###

def build_stacks(data_map, input_dimension=None, check=False, normalize=False,
                 channel_to_stretch=None, stretching_percentile=99.9,
                 cast_8bit=False, return_origin=False):
    """Generator to build several stacks from recipe-folder pairs.

    To build a stack, a recipe should be linked to a directory including all
    the files needed to build the stack. The content of the recipe allows to
    reorganize the different files stored in the directory in order to build
    a 5-d tensor. If several fields of view (fov) are store in the recipe,
    several tensors are generated.

    The list 'data_map' takes the form:

        [
         (recipe_1, path_input_directory_1),
         (recipe_2, path_input_directory_1),
         (recipe_3, path_input_directory_1),
         (recipe_4, path_input_directory_2),
         ...
        ]

    The recipe dictionary for one field of view takes the form:

        {
         "fov": str,     (optional)
         "z": List[str], (optional)
         "c": List[str], (optional)
         "r": List[str], (optional)
         "ext": str,     (optional)
         "opt": str,     (optional)
         "pattern"
         }

    - A field of view is defined by an ID common to every images belonging to
    the field of view ("fov").
    - At least every images are in 2-d with x and y dimensions. So we need to
    mention the round-dimension, the channel-dimension and the z-dimension to
    add ("r", "c" and "z"). For these keys, we provide a list of
    strings to identify the images to stack.
    - An extra information to identify the files to stack in the input folder
    can be provided with the file extension "ext" (usually 'tif' or 'tiff') or
    an optional morpheme ("opt").
    - A pattern used to get the filename ("pattern").

    Example 1. Let us assume 3-d images (zyx dimensions) saved as
    "r03c03f01_405.tif", "r03c03f01_488.tif" and "r03c03f01_561.tif". The first
    morpheme "r03c03f01" uniquely identifies a 3-d field of view. The second
    morphemes "405", "488" and "561" identify three different channels we
    want to stack. There is no round in this experiment. We need to return a
    tensor with shape (1, 3, z, y, x). Thus, a valid recipe would be:

        {
         "fov": "r03c03f01",
         "c": ["405", "488", "561"],
         "ext": "tif"
         "pattern": "fov_c.ext"
         }

    Example 2. Let us assume 2-d images (yx dimensions) saved as
    "dapi_1.TIFF", "cy3_1.TIFF", "GFP_1.TIFF", "dapi_2.TIFF", "cy3_2.TIFF" and
    "GFP_2.TIFF". The first morphemes "dapi", "cy3" and "GFP" identify
    channels. The second morphemes "1" and "2" identify two different fields of
    view. There is no round and no z dimension in this experiment. We can
    build two tensors with shape (1, 3, 1, y, x). Thus, a valid recipe would
    be:

        {
         "fov": ["1", "2"],
         "c": ["dapi", "cy3", "GFP"],
         "ext": "TIFF"
         "pattern": "c_fov.ext"
         }

    Parameters
    ----------
    data_map : List[tuple]
        Map between input directories and recipes.
    input_dimension : int
        Number of dimensions of the loaded files.
    check : bool
        Check the validity of the loaded tensor.
    normalize : bool
        Normalize the different channels of the loaded stack (rescaling).
    channel_to_stretch : int or List[int]
        Channel to stretch.
    stretching_percentile : float
        Percentile to determine the maximum intensity value used to rescale
        the image.
    return_origin : bool
        Return the input directory and the recipe used to build the stack.
    cast_8bit : bool
        Cast tensor in np.uint8.

    Returns
    -------
    tensor : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).
    input_directory : str
        Path of the input directory from where the tensor is built.
    recipe : dict
        Recipe used to build the tensor.

    """
    # check parameters
    check_parameter(data_map=list,
                    return_origin=bool)

    # load and generate tensors for each recipe-folder pair
    for recipe, input_folder in data_map:

        # load and generate tensors for each fov stored in a recipe
        nb_fov = count_nb_fov(recipe)
        for i_fov in range(nb_fov):
            tensor = build_stack(recipe, input_folder, input_dimension, i_fov,
                                 check, normalize, channel_to_stretch,
                                 stretching_percentile, cast_8bit)
            if return_origin:
                yield tensor, input_folder, recipe, i_fov
            else:
                yield tensor


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


def build_stack(recipe, input_folder, input_dimension=None, i_fov=0,
                check=False, normalize=False, channel_to_stretch=None,
                stretching_percentile=99.9, cast_8bit=False):
    """Build 5-d stack and normalize it.

    The recipe dictionary for one field of view takes the form:

        {
         "fov": str,     (optional)
         "z": List[str], (optional)
         "c": List[str], (optional)
         "r": List[str], (optional)
         "ext": str,     (optional)
         "opt": str,     (optional)
         "pattern"
         }

    - A field of view is defined by an ID common to every images belonging to
    the field of view ("fov").
    - At least every images are in 2-d with x and y dimensions. So we need to
    mention the round-dimension, the channel-dimension and the z-dimension to
    add ("r", "c" and "z"). For these keys, we provide a list of
    strings to identify the images to stack.
    - An extra information to identify the files to stack in the input folder
    can be provided with the file extension "ext" (usually 'tif' or 'tiff') or
    an optional morpheme ("opt").
    - A pattern used to get the filename ("pattern").

    Example 1. Let us assume 3-d images (zyx dimensions) saved as
    "r03c03f01_405.tif", "r03c03f01_488.tif" and "r03c03f01_561.tif". The first
    morpheme "r03c03f01" uniquely identifies a 3-d field of view. The second
    morphemes "405", "488" and "561" identify three different channels we
    want to stack. There is no round in this experiment. We need to return a
    tensor with shape (1, 3, z, y, x). Thus, a valid recipe would be:

        {
         "fov": "r03c03f01",
         "c": ["405", "488", "561"],
         "ext": "tif"
         "pattern": "fov_c.ext"
         }

    Example 2. Let us assume 2-d images (yx dimensions) saved as
    "dapi_1.TIFF", "cy3_1.TIFF", "GFP_1.TIFF", "dapi_2.TIFF", "cy3_2.TIFF" and
    "GFP_2.TIFF". The first morphemes "dapi", "cy3" and "GFP" identify
    channels. The second morphemes "1" and "2" identify two different fields of
    view. There is no round and no z dimension in this experiment. We can
    build two tensors with shape (1, 3, 1, y, x). Thus, a valid recipe would
    be:

        {
         "fov": ["1", "2"],
         "c": ["dapi", "cy3", "GFP"],
         "ext": "TIFF"
         "pattern": "c_fov.ext"
         }

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        'pattern', 'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    input_folder : str
        Path of the folder containing the images.
    input_dimension : int
        Number of dimensions of the loaded files.
    i_fov : int
        Index of the fov to build.
    check : bool
        Check the validity of the loaded tensor.
    normalize : bool
        Normalize the different channels of the loaded stack (rescaling).
    channel_to_stretch : int or List[int]
        Channel to stretch.
    stretching_percentile : float
        Percentile to determine the maximum intensity value used to rescale
        the image.
    cast_8bit : bool
        Cast the tensor in np.uint8.

    Returns
    -------
    tensor : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).

    """
    # check parameters
    check_parameter(normalize=bool,
                    channel_to_stretch=(int, list, type(None)),
                    stretching_percentile=float,
                    cast_8bit=bool,
                    return_origin=bool)

    # build stack from recipe and tif files
    tensor = load_stack(recipe, input_folder, input_dimension, i_fov)

    # check the validity of the loaded tensor
    if check:
        check_array(tensor,
                    ndim=5,
                    dtype=[np.uint8, np.uint16],
                    allow_nan=False)

    # rescale data and improve contrast
    if normalize:
        tensor = rescale(tensor, channel_to_stretch, stretching_percentile)

    # cast in np.uint8 if necessary, in order to reduce memory allocation
    if tensor.dtype == np.uint16 and cast_8bit:
        tensor = cast_img_uint8(tensor)

    return tensor


def load_stack(recipe, input_folder, input_dimension=None, i_fov=0):
    """Build a 5-d tensor from the same fields of view (fov).

    The function stacks a set of images using a recipe mapping the
    different images with the dimensions they represent. Each stacking step
    add a new dimension to the original tensors (eg. we stack 2-d images with
    the same xy coordinates, but different depths to get a 3-d image). If the
    files we need to build a new dimension are not included in the
    recipe, an empty dimension is added. This operation is repeated until we
    get a 5-d tensor. We first operate on the z dimension, then the
    channels and eventually the rounds.

    The recipe dictionary for one field of view takes the form:

        {
         "fov": str,     (optional)
         "z": List[str], (optional)
         "c": List[str], (optional)
         "r": List[str], (optional)
         "ext": str,     (optional)
         "opt": str,     (optional)
         "pattern"
         }

    - A field of view is defined by an ID common to every images belonging to
    the field of view ("fov").
    - At least every images are in 2-d with x and y dimensions. So we need to
    mention the round-dimension, the channel-dimension and the z-dimension to
    add ("r", "c" and "z"). For these keys, we provide a list of
    strings to identify the images to stack.
    - An extra information to identify the files to stack in the input folder
    can be provided with the file extension "ext" (usually 'tif' or 'tiff') or
    an optional morpheme ("opt").
    - A pattern used to get the filename ("pattern").

    Example 1. Let us assume 3-d images (zyx dimensions) saved as
    "r03c03f01_405.tif", "r03c03f01_488.tif" and "r03c03f01_561.tif". The first
    morpheme "r03c03f01" uniquely identifies a 3-d field of view. The second
    morphemes "405", "488" and "561" identify three different channels we
    want to stack. There is no round in this experiment. We need to return a
    tensor with shape (1, 3, z, y, x). Thus, a valid recipe would be:

        {
         "fov": "r03c03f01",
         "c": ["405", "488", "561"],
         "ext": "tif"
         "pattern": "fov_c.ext"
         }

    Example 2. Let us assume 2-d images (yx dimensions) saved as
    "dapi_1.TIFF", "cy3_1.TIFF", "GFP_1.TIFF", "dapi_2.TIFF", "cy3_2.TIFF" and
    "GFP_2.TIFF". The first morphemes "dapi", "cy3" and "GFP" identify
    channels. The second morphemes "1" and "2" identify two different fields of
    view. There is no round and no z dimension in this experiment. We can
    build two tensors with shape (1, 3, 1, y, x). Thus, a valid recipe would
    be:

        {
         "fov": ["1", "2"],
         "c": ["dapi", "cy3", "GFP"],
         "ext": "TIFF"
         "pattern": "c_fov.ext"
         }

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Can only contain the keys
        'pattern', 'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    input_folder : str
        Path of the folder containing the images.
    input_dimension : int
        Number of dimensions of the loaded files.
    i_fov : int
        Index of the fov to build.

    Returns
    -------
    stack : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).

    """
    # check parameters
    check_recipe(recipe)
    check_parameter(input_folder=str,
                    input_dimension=(int, type(None)),
                    i_fov=int)

    # complete the recipe with unused morphemes
    recipe = fit_recipe(recipe)

    # if the initial dimension of the files is unknown, we read one of them
    if input_dimension is None:
        input_dimension = get_input_dimension(recipe, input_folder)

    # get the number of elements to stack per dimension
    nb_r, nb_c, nb_z = get_nb_element_per_dimension(recipe)

    # we stack our files according to their initial dimension
    if input_dimension == 2:
        stack = _build_stack_from_2d(recipe, input_folder, fov=i_fov,
                                     nb_r=nb_r, nb_c=nb_c, nb_z=nb_z)
    elif input_dimension == 3:
        stack = _build_stack_from_3d(recipe, input_folder, fov=i_fov,
                                     nb_r=nb_r, nb_c=nb_c)
    elif input_dimension == 4:
        stack = _build_stack_from_4d(recipe, input_folder, fov=i_fov,
                                     nb_r=nb_r)
    elif input_dimension == 5:
        stack = _build_stack_from_5d(recipe, input_folder, fov=i_fov)
    else:
        raise ValueError("Files do not have the right number of dimensions: "
                         "{0}. The files we stack should be in 2-d, 3-d, 4-d "
                         "or 5-d.".format(input_dimension))

    return stack


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
    # initialize and fit the dimensions 'fov', 'r', 'c' and 'z'
    for key in ['fov', 'r', 'c', 'z']:
        if key not in recipe:
            recipe[key] = list("")
        value = recipe[key]
        if isinstance(value, str):
            recipe[key] = [value]

    # initialize the dimensions 'ext', 'opt'
    for key in ['ext', 'opt']:
        if key not in recipe:
            recipe[key] = ""

    return recipe


def _build_stack_from_2d(recipe, input_folder, fov=0, nb_r=1, nb_c=1, nb_z=1):
    """Load and stack 2-d tensors.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    input_folder : str
        Path of the folder containing the images.
    fov : int
        Index of the fov to build.
    nb_r : int
        Number of round file to stack in order to get a 5-d tensor.
    nb_c : int
        Number of channel file to stack in order to get a 4-d tensor.
    nb_z : int
        Number of z file to stack in order to get a 3-d tensor.

    Returns
    -------
    tensor_5d : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).

    """

    # load and stack successively z, channel then round elements
    tensors_4d = []
    for r in range(nb_r):

        # load and stack channel elements (3-d tensors)
        tensors_3d = []
        for c in range(nb_c):

            # load and stack z elements (2-d tensors)
            tensors_2d = []
            for z in range(nb_z):
                path = get_path_from_recipe(recipe, input_folder, fov=fov,
                                            r=r, c=c, z=z)
                tensor_2d = read_image(path)
                tensors_2d.append(tensor_2d)

            # stack 2-d tensors in 3-d
            tensor_3d = np.stack(tensors_2d, axis=0)
            tensors_3d.append(tensor_3d)

        # stack 3-d tensors in 4-d
        tensor_4d = np.stack(tensors_3d, axis=0)
        tensors_4d.append(tensor_4d)

    # stack 4-d tensors in 5-d
    tensor_5d = np.stack(tensors_4d, axis=0)

    return tensor_5d


def _build_stack_from_3d(recipe, input_folder, fov=0, nb_r=1, nb_c=1):
    """Load and stack 3-d tensors.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    input_folder : str
        Path of the folder containing the images.
    fov : int
        Index of the fov to build.
    nb_r : int
        Number of round file to stack in order to get a 5-d tensor.
    nb_c : int
        Number of channel file to stack in order to get a 4-d tensor.

    Returns
    -------
    tensor_5d : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).

    """
    # load and stack successively channel elements then round elements
    tensors_4d = []
    for r in range(nb_r):

        # load and stack channel elements (3-d tensors)
        tensors_3d = []
        for c in range(nb_c):
            path = get_path_from_recipe(recipe, input_folder, fov=fov, r=r,
                                        c=c)
            tensor_3d = read_image(path)
            tensors_3d.append(tensor_3d)

            # stack 3-d tensors in 4-d
            tensor_4d = np.stack(tensors_3d, axis=0)
            tensors_4d.append(tensor_4d)

    # stack 4-d tensors in 5-d
    tensor_5d = np.stack(tensors_4d, axis=0)

    return tensor_5d


def _build_stack_from_4d(recipe, input_folder, fov=0, nb_r=1):
    """Load and stack 4-d tensors.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    input_folder : str
        Path of the folder containing the images.
    fov : int
        Index of the fov to build.
    nb_r : int
        Number of round file to stack in order to get a 5-d tensor.

    Returns
    -------
    tensor_5d : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).

    """
    # load each file from a new round element and stack them
    tensors_4d = []
    for r in range(nb_r):
        path = get_path_from_recipe(recipe, input_folder, fov=fov, r=r)
        tensor_4d = read_image(path)
        tensors_4d.append(tensor_4d)

    # stack 4-d tensors in 5-d
    tensor_5d = np.stack(tensors_4d, axis=0)

    return tensor_5d


def _build_stack_from_5d(recipe, input_folder, fov=0):
    """Load directly a 5-d tensor.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    input_folder : str
        Path of the folder containing the images.
    fov : int
        Index of the fov to build.

    Returns
    -------
    tensor_5d : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).

    """
    # the recipe can only contain one file with a 5-d tensor per fov
    path = get_path_from_recipe(recipe, input_folder, fov=fov)
    tensor_5d = read_image(path)

    return tensor_5d


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
    for (element_name, separator) in zip(path_elements, path_separators):
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


def get_input_dimension(recipe, input_folder):
    """ Load an arbitrary image to get the original dimension of the files.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions. Only contain the keys
        'fov', 'r', 'c', 'z', 'ext' or 'opt'.
    input_folder : str
        Path of the folder containing the images.

    Returns
    -------
    nb_dim : int
        Number of dimensions of the original file.

    """
    # get a valid path from the recipe
    path = get_path_from_recipe(recipe, input_folder)

    # load the image and return the number of dimensions
    image = read_image(path)
    nb_dim = image.ndim

    return nb_dim


def build_stack_no_recipe(paths, input_dimension=None, check=False,
                          normalize=False, channel_to_stretch=None,
                          stretching_percentile=99.9, cast_8bit=False):
    """Build 5-d stack and normalize it, without recipe.

    Parameters
    ----------
    paths : List[str]
        List of the paths to stack.
    input_dimension : str
        Number of dimensions of the loaded files.
    check : bool
        Check the validity of the loaded tensor.
    normalize : bool
        Normalize the different channels of the loaded stack (rescaling).
    channel_to_stretch : int or List[int]
        Channel to stretch.
    stretching_percentile : float
        Percentile to determine the maximum intensity value used to rescale
        the image.
    cast_8bit : bool
        Cast the tensor in np.uint8.

    Returns
    -------
    tensor : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).

    """
    # check parameters
    check_parameter(normalize=bool,
                    channel_to_stretch=(int, list, type(None)),
                    stretching_percentile=float,
                    cast_8bit=bool)

    # build stack from tif files
    tensor = load_stack_no_recipe(paths, input_dimension)

    # check the validity of the loaded tensor
    if check:
        check_array(tensor,
                    ndim=5,
                    dtype=[np.uint8, np.uint16],
                    allow_nan=False)

    # rescale data and improve contrast
    if normalize:
        tensor = rescale(tensor, channel_to_stretch, stretching_percentile)

    # cast in np.uint8 if necessary, in order to reduce memory allocation
    if tensor.dtype == np.uint16 and cast_8bit:
        tensor = cast_img_uint8(tensor)

    return tensor


def load_stack_no_recipe(paths, input_dimension=None):
    """Build a 5-d tensor from the same field of view (fov), without recipe.

    Files with a path listed are stacked together, then empty dimensions are
    added up to 5.

    Parameters
    ----------
    paths : List[str]
        List of the file to stack.
    input_dimension : str
        Number of dimensions of the loaded files.

    Returns
    -------
    tensor_5d : np.ndarray, np.uint
        Tensor with shape (r, c, z, y, x).

    """
    # check parameters
    check_parameter(paths=str,
                    input_dimension=(int, type(None)))

    # load an image and get the number of dimensions
    if input_dimension is None:
        testfile = read_image(paths[0])
        input_dimension = testfile.ndim

    # get stacks
    stacks = []
    for path in paths:
        s = read_image(path)
        stacks.append(s)

    # we stack our files according to their initial dimension
    if input_dimension == 2:
        tensor_3d = np.stack(stacks, axis=0)
        tensor_5d = tensor_3d[np.newaxis, np.newaxis, :, :, :]
    elif input_dimension == 3:
        tensor_4d = np.stack(stacks, axis=0)
        tensor_5d = tensor_4d[np.newaxis, :, :, :, :]
    elif input_dimension == 4:
        tensor_5d = np.stack(stacks, axis=0)
    elif input_dimension == 5 and len(stacks) == 1:
        tensor_5d = stacks[0]
    else:
        raise ValueError("Files do not have the right number of dimensions: "
                         "{0}. The files we stack should be in 2-d, 3-d, 4-d "
                         "or 5-d.".format(input_dimension))

    return tensor_5d


# ### Normalization ###

def rescale(tensor, channel_to_stretch=None, stretching_percentile=99.9):
    """Rescale tensor values up to its dtype range.

    Each round and each channel is rescaled independently.

    We can improve the contrast of the image by stretching its range of
    intensity values. To do that we provide a smaller range of pixel intensity
    to rescale, spreading out the information contained in the original
    histogram. Usually, we apply such normalization to smFish channels. Other
    channels are simply rescale from the minimum and maximum intensity values
    of the image to those of its dtype.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        Tensor to rescale with shape (r, c, z, y, x), (c, z, y, x), (z, y, x)
        or (y, x).
    channel_to_stretch : int or List[int]
        Channel to stretch.
    stretching_percentile : float
        Percentile to determine the maximum intensity value used to rescale
        the image.

    Returns
    -------
    tensor : np.ndarray, np.uint
        Tensor to rescale with shape (r, c, z, y, x).

    """
    # check parameters
    check_array(tensor, ndim=[2, 3, 4, 5], dtype=[np.uint8, np.uint16])
    check_parameter(channel_to_stretch=(int, list, type(None)),
                    stretching_percentile=float)

    # format 'channel_to_stretch'
    if channel_to_stretch is None:
        channel_to_stretch = []
    elif isinstance(channel_to_stretch, int):
        channel_to_stretch = [channel_to_stretch]

    # get a 5-d tensor
    original_ndim = tensor.ndim
    if original_ndim == 2:
        tensor_5d = tensor[np.newaxis, np.newaxis, np.newaxis, ...]
    elif original_ndim == 3:
        tensor_5d = tensor[np.newaxis, np.newaxis, ...]
    elif original_ndim == 4:
        tensor_5d = tensor[np.newaxis, ...]
    else:
        tensor_5d = tensor

    # rescale
    tensor_5d = _rescale_5d(tensor_5d, channel_to_stretch,
                            stretching_percentile)

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
    """Rescale tensor values up to its dtype range.

    Each round and each channel is rescaled independently.

    We can improve the contrast of the image by stretching its range of
    intensity values. To do that we provide a smaller range of pixel intensity
    to rescale, spreading out the information contained in the original
    histogram. Usually, we apply such normalization to smFish channels. Other
    channels are simply rescale from the minimum and maximum intensity values
    of the image to those of its dtype.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        Tensor to rescale with shape (r, c, z, y, x).
    channel_to_stretch : List[int]
        Channel to stretch.
    stretching_percentile : float
        Percentile to determine the maximum intensity value used to rescale
        the image.

    Returns
    -------
    tensor : np.ndarray, np.uint
        Tensor to rescale with shape (r, c, z, y, x).

    """
    # rescale each round independently
    rounds = []
    for r in range(tensor.shape[0]):

        # rescale each channel independently
        channels = []
        for i in range(tensor.shape[1]):
            channel = tensor[r, i, :, :, :]
            if i in channel_to_stretch:
                pa, pb = np.percentile(channel, (0, stretching_percentile))
                channel_rescaled = rescale_intensity(channel,
                                                     in_range=(pa, pb))
            else:
                channel_rescaled = rescale_intensity(channel)
            channels.append(channel_rescaled)
        tensor_4d = np.stack(channels, axis=0)
        rounds.append(tensor_4d)

    tensor_5d = np.stack(rounds, axis=0)

    return tensor_5d


def cast_img_uint8(tensor):
    """Cast the image in np.uint8.

    Negative values (from np.float tensors) are not allowed as the skimage
    method 'img_as_ubyte' would clip them to 0. Positives values are scaled
    between 0 and 255.

    Casting image to np.uint8 reduce the memory needed to process it and
    accelerate computations.

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.

    Returns
    -------
    tensor : np.ndarray, np.uint8
        Image cast.

    """
    # check tensor dtype
    check_array(tensor, dtype=[np.uint8, np.uint16,
                               np.float32, np.float64,
                               np.bool])

    if tensor.dtype == np.uint8:
        return tensor

    # check the range value for float tensors
    if tensor.dtype in [np.float32, np.float64]:
        if not check_range_value(tensor, 0, 1):
            raise ValueError("To cast a tensor from {0} to np.uint8, its "
                             "values must be between 0 and 1, and not {1} "
                             "and {2}."
                             .format(tensor.dtype, tensor.min(), tensor.max()))

    # cast tensor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor = img_as_ubyte(tensor)

    return tensor


def cast_img_uint16(tensor):
    """Cast the data in np.uint16.

    Negative values (from np.float tensors) are not allowed as the skimage
    method 'img_as_uint' would clip them to 0. Positives values are scaled
    between 0 and 65535.

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.

    Returns
    -------
    tensor : np.ndarray, np.uint16
        Image cast.

    """
    # check tensor dtype
    check_array(tensor, dtype=[np.uint8, np.uint16,
                               np.float32, np.float64,
                               np.bool])

    if tensor.dtype == np.uint16:
        return tensor

    # check the range value for float tensors
    if tensor.dtype in [np.float32, np.float64]:
        if not check_range_value(tensor, 0, 1):
            raise ValueError("To cast a tensor from {0} to np.uint16, its "
                             "values must be between 0 and 1, and not {1} "
                             "and {2}."
                             .format(tensor.dtype, tensor.min(), tensor.max()))

    # cast tensor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor = img_as_uint(tensor)

    return tensor


def cast_img_float32(tensor):
    """Cast the data in np.float32.

    If the input data is in np.uint8 or np.uint16, the values are scale
    between 0 and 1. When converting from a np.float dtype, values are not
    modified.

    Casting image to np.float32 reduce the memory needed to process it and
    accelerate computations (compare to np.float64).

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.

    Returns
    -------
    tensor : np.ndarray, np.float32
        image cast.

    """
    # check tensor dtype
    check_array(tensor, dtype=[np.uint8, np.uint16,
                               np.float32, np.float64,
                               np.bool])

    # cast tensor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor = img_as_float32(tensor)

    return tensor


def cast_img_float64(tensor):
    """Cast the data in np.float64.

    If the input data is in np.uint8 or np.uint16, the values are scale
    between 0 and 1. When converting from a np.float dtype, values are not
    modified.

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
    check_array(tensor, dtype=[np.uint8, np.uint16,
                               np.float32, np.float64,
                               np.bool])

    # cast tensor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor = img_as_float64(tensor)

    return tensor


# ### Coordinates data cleaning ###
# TODO add safety check for these cleaning functions
def clean_simulated_data(data, data_cell, path_output=None):
    """Clean simulated dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with all the simulated cells, the coordinates of their
        different elements and the localization pattern used to simulate them.
    data_cell : pandas.DataFrame
        Dataframe with the 2D coordinates of the nucleus and the cytoplasm of
        actual cells used to simulate data.
    path_output : str
        Path to save the cleaned dataset.

    Returns
    -------
    data_final : pandas.DataFrame
        Cleaned dataset.
    background_to_remove : List[str]
        Invalid background.
    id_volume : List[int]
        Background id from 'data_cell' to remove.
    id_rna : List[int]
        Cell id to remove from data.

    """
    # TODO remove the 'SettingWithCopyWarning'
    # filter invalid simulated cell backgrounds
    data_clean, background_to_remove, id_volume = clean_volume(data, data_cell)

    # filter invalid simulated rna spots
    data_clean, id_rna = clean_rna(data_clean)

    # make the feature 'n_rna' consistent
    data_clean["nb_rna"] = data_clean.apply(
        lambda row: len(row["RNA_pos"]),
        axis=1)

    # remove useless features
    data_final = data_clean[
        ['RNA_pos', 'cell_ID', 'pattern_level', 'pattern_name', 'pos_cell',
         'pos_nuc', "nb_rna"]]

    # encode the label
    le = LabelEncoder()
    data_final["label"] = le.fit_transform(data_final["pattern_name"])

    # reset index
    data_final.reset_index(drop=True, inplace=True)

    # save cleaned dataset
    if path_output is not None:
        data_final.to_pickle(path_output)

    return data_final, background_to_remove, id_volume, id_rna


def clean_volume(data, data_cell):
    """Remove misaligned simulated cells from the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with all the simulated cells, the coordinates of their
        different elements and the localization pattern used to simulate them.
    data_cell : pandas.DataFrame
        Dataframe with the 2D coordinates of the nucleus and the cytoplasm of
        actual cells used to simulate data.

    Returns
    -------
    data_clean : pandas.DataFrame
        Cleaned dataframe.
    background_to_remove : List[str]
        Invalid background.
    id_to_remove : List[int]
        Background id from 'data_cell' to remove.

    """
    # for each cell, check if the volume is valid or not
    data_cell["valid_volume"] = data_cell.apply(
        lambda row: _check_volume(row["pos_cell"], row["pos_nuc"]),
        axis=1)

    # get the invalid backgrounds
    background_to_remove = []
    id_to_remove = []
    for i in data_cell.index:
        if np.logical_not(data_cell.loc[i, "valid_volume"]):
            background_to_remove.append(data_cell.loc[i, "name_img_BGD"])
            id_to_remove.append(i)

    # remove invalid simulated cells
    data_clean = data[~data["name_img_BGD"].isin(background_to_remove)]

    return data_clean, background_to_remove, id_to_remove


def _check_volume(cyto_coord, nuc_coord):
    """Check nucleus coordinates are not outside the boundary of the cytoplasm.

    Parameters
    ----------
    cyto_coord : pandas.Series
        Coordinates of the cytoplasm membrane.
    nuc_coord : pandas.Series
        Coordinates of the nucleus border.

    Returns
    -------
    _ : bool
        Tell if the cell volume is valid or not.

    """
    # get coordinates
    cyto = np.array(cyto_coord)
    nuc = np.array(nuc_coord)

    max_x = max(cyto[:, 0].max() + 5, nuc[:, 0].max() + 5)
    max_y = max(cyto[:, 1].max() + 5, nuc[:, 1].max() + 5)

    # build the dense representation for the cytoplasm
    values = [1] * cyto.shape[0]
    cyto = coo_matrix((values, (cyto[:, 0], cyto[:, 1])),
                      shape=(max_x, max_y)).todense()

    # build the dense representation for the nucleus
    values = [1] * nuc.shape[0]
    nuc = coo_matrix((values, (nuc[:, 0], nuc[:, 1])),
                     shape=(max_x, max_y)).todense()

    # check if the volume is valid
    mask_cyto = ndi.binary_fill_holes(cyto)
    mask_nuc = ndi.binary_fill_holes(nuc)
    frame = np.zeros((max_x, max_y))
    diff = frame - mask_cyto + mask_nuc
    diff = (diff > 0).sum()

    if diff > 0:
        return False
    else:
        return True


def clean_rna(data):
    """Remove cells with misaligned simulated rna spots from the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with all the simulated cells, the coordinates of their
        different elements and the localization pattern used to simulate them.

    Returns
    -------
    data_clean : pandas.DataFrame
        Cleaned dataframe.
    id_to_remove : List[int]
        Cell id to remove from data.

    """
    # for each cell we check if the rna spots are valid or not
    data["valid_rna"] = data.apply(
        lambda row: _check_rna(row["pos_cell"], row["RNA_pos"]),
        axis=1)

    # get id of the invalid cells
    id_to_remove = []
    for i in data.index:
        if np.logical_not(data.loc[i, "valid_rna"]):
            id_to_remove.append(i)

    # remove invalid simulated cells
    data_clean = data[data["valid_rna"]]

    return data_clean, id_to_remove


def _check_rna(cyto_coord, rna_coord):
    """Check rna spots coordinates are not outside the boundary of the
    cytoplasm.

    Parameters
    ----------
    cyto_coord : pandas.Series
        Coordinates of the cytoplasm membrane.
    rna_coord : pandas.Series
        Coordinates of the rna spots.

    Returns
    -------
    _ : bool
        Tell if the rna spots are valid or not.

    """
    # get coordinates
    cyto = np.array(cyto_coord)
    if not isinstance(rna_coord[0], list):
        # it means we have only one spot
        return False
    rna = np.array(rna_coord)

    # check if the coordinates are positive
    if rna.min() < 0:
        return False

    max_x = int(max(cyto[:, 0].max() + 5, rna[:, 0].max() + 5))
    max_y = int(max(cyto[:, 1].max() + 5, rna[:, 1].max() + 5))

    # build the dense representation for the cytoplasm
    values = [1] * cyto.shape[0]
    cyto = coo_matrix((values, (cyto[:, 0], cyto[:, 1])),
                      shape=(max_x, max_y)).todense()

    # build the dense representation for the rna
    values = [1] * rna.shape[0]
    rna = coo_matrix((values, (rna[:, 0], rna[:, 1])),
                     shape=(max_x, max_y)).todense()
    rna = (rna > 0)

    # check if the coordinates are valid
    mask_cyto = ndi.binary_fill_holes(cyto)
    frame = np.zeros((max_x, max_y))
    diff = frame - mask_cyto + rna
    diff = (diff > 0).sum()

    if diff > 0:
        return False
    else:
        return True
