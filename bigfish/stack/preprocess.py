# -*- coding: utf-8 -*-

"""
Functions used to format and clean any input loaded in bigfish.
"""

import os
import warnings

import numpy as np
import pandas as pd

from .io import read_image, read_cell_json, read_rna_json
from .utils import (check_array, check_parameter, check_recipe,
                    check_range_value, check_df, fit_recipe,
                    get_path_from_recipe, get_nb_element_per_dimension,
                    count_nb_fov)

from sklearn.preprocessing import LabelEncoder

from skimage import img_as_ubyte, img_as_float32, img_as_float64, img_as_uint
from skimage.exposure import rescale_intensity

from scipy import ndimage as ndi


# TODO be able to build only one channel

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
         "fov": List[str], (optional)
         "z": List[str],   (optional)
         "c": List[str],   (optional)
         "r": List[str],   (optional)
         "ext": str,       (optional)
         "opt": str,       (optional)
         "pattern"
         }

    - A field of view is defined by an ID common to every images belonging to
    the same field of view ("fov").
    - At least every images are in 2-d with x and y dimensions. So we need to
    mention the round-dimension, the channel-dimension and the z-dimension to
    add ("r", "c" and "z"). For these keys, we provide a list of
    strings to identify the images to stack.
    - An extra information to identify the files to stack in the input folder
    can be provided with the file extension "ext" (usually 'tif' or 'tiff') or
    an optional morpheme ("opt").
    - A pattern used to get the filename ("pattern").
    - The fields "fov", "z", "c" and "r" can be strings instead of lists.

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


def build_stack(recipe, input_folder, input_dimension=None, i_fov=0,
                check=False, normalize=False, channel_to_stretch=None,
                stretching_percentile=99.9, cast_8bit=False):
    """Build 5-d stack and normalize it.

    The recipe dictionary for one field of view takes the form:

        {
         "fov": List[str], (optional)
         "z": List[str],   (optional)
         "c": List[str],   (optional)
         "r": List[str],   (optional)
         "ext": str,       (optional)
         "opt": str,       (optional)
         "pattern"
         }

    - A field of view is defined by an ID common to every images belonging to
    the same field of view ("fov").
    - At least every images are in 2-d with x and y dimensions. So we need to
    mention the round-dimension, the channel-dimension and the z-dimension to
    add ("r", "c" and "z"). For these keys, we provide a list of
    strings to identify the images to stack.
    - An extra information to identify the files to stack in the input folder
    can be provided with the file extension "ext" (usually 'tif' or 'tiff') or
    an optional morpheme ("opt").
    - A pattern used to get the filename ("pattern").
    - The fields "fov", "z", "c" and "r" can be strings instead of lists.

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
    check_recipe(recipe)
    check_parameter(input_folder=str,
                    input_dimension=(int, type(None)),
                    i_fov=int,
                    check=bool,
                    normalize=bool,
                    channel_to_stretch=(int, list, type(None)),
                    stretching_percentile=float,
                    cast_8bit=bool)

    # build stack from recipe and tif files
    tensor = _load_stack(recipe, input_folder, input_dimension, i_fov)

    # check the validity of the loaded tensor
    if check:
        check_array(tensor,
                    ndim=5,
                    dtype=[np.uint8, np.uint16])

    # rescale data and improve contrast
    if normalize:
        tensor = rescale(tensor, channel_to_stretch, stretching_percentile)

    # cast in np.uint8 if necessary, in order to reduce memory allocation
    if tensor.dtype == np.uint16 and cast_8bit:
        tensor = cast_img_uint8(tensor)

    return tensor


def _load_stack(recipe, input_folder, input_dimension=None, i_fov=0):
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
         "fov": List[str], (optional)
         "z": List[str],   (optional)
         "c": List[str],   (optional)
         "r": List[str],   (optional)
         "ext": str,       (optional)
         "opt": str,       (optional)
         "pattern"
         }

    - A field of view is defined by an ID common to every images belonging to
    the same field of view ("fov").
    - At least every images are in 2-d with x and y dimensions. So we need to
    mention the round-dimension, the channel-dimension and the z-dimension to
    add ("r", "c" and "z"). For these keys, we provide a list of
    strings to identify the images to stack.
    - An extra information to identify the files to stack in the input folder
    can be provided with the file extension "ext" (usually 'tif' or 'tiff') or
    an optional morpheme ("opt").
    - A pattern used to get the filename ("pattern").
    - The fields "fov", "z", "c" and "r" can be strings instead of lists.

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
    # complete the recipe with unused morphemes
    recipe = fit_recipe(recipe)

    # if the initial dimension of the files is unknown, we read one of them
    if input_dimension is None:
        input_dimension = _get_input_dimension(recipe, input_folder)

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


def _get_input_dimension(recipe, input_folder):
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
    check_parameter(paths=(str, list),
                    input_dimension=(int, type(None)),
                    normalize=bool,
                    channel_to_stretch=(int, list, type(None)),
                    stretching_percentile=float,
                    cast_8bit=bool)

    # build stack from tif files
    tensor = _load_stack_no_recipe(paths, input_dimension)

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


def _load_stack_no_recipe(paths, input_dimension=None):
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
        Tensor to rescale with shape (r, c, z, y, x), (c, z, y, x), (z, y, x)
        or (y, x).

    """
    # check parameters
    check_array(tensor,
                ndim=[2, 3, 4, 5],
                dtype=[np.uint8, np.uint16],
                allow_nan=False)
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
    check_array(tensor,
                dtype=[np.uint8, np.uint16, np.float32, np.float64, np.bool],
                allow_nan=False)

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
    check_array(tensor,
                dtype=[np.uint8, np.uint16, np.float32, np.float64, np.bool],
                allow_nan=False)

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
    check_array(tensor,
                dtype=[np.uint8, np.uint16, np.float32, np.float64, np.bool],
                allow_nan=False)

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
    check_array(tensor,
                dtype=[np.uint8, np.uint16, np.float32, np.float64, np.bool],
                allow_nan=False)

    # cast tensor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tensor = img_as_float64(tensor)

    return tensor


# ### Resize and rescale ###
# TODO debug
def deconstruct_image(image, target_size):
    """Deconstruct an image in a sequence of smaller or larger images in order
    to fit with a segmentation method, while preserving image scale.

    If the image need to be enlarged to reach the target size, we pad it. If
    the current size is a multiple of the target size, image is cropped.
    Otherwise, it is padded (to multiply the target size) then cropped.
    Information about the deconstruction process are returned in order to
    easily reconstruct the original image after transformation.

    Parameters
    ----------
    image : np.ndarray
        Image to deconstruct with shape (y, x).
    target_size : int
        Size of the elements to return.

    Returns
    -------
    images : List[np.ndarray]
        List of images to analyse independently.
    deconstruction : dict
        Dictionary with deconstruction information to help the reconstruction
        of the original image.

    """
    # TODO adapt to non squared images
    # TODO add an overlap in the crop
    # check parameters
    check_array(image,
                ndim=2,
                dtype=[np.uint8, np.uint16,
                       np.float32, np.float64,
                       bool],
                allow_nan=False)
    check_parameter(target_size=int)

    # initialize metadata
    (width, height) = image.shape
    deconstruction = {"cropped": False, "padded": False,
                      "original_width": width, "original_height": height}

    # check if the image is squared
    if width != height:
        raise ValueError("Non-squared image are not supported yet.")

    # case where the image is too small
    if width < target_size:

        # padding
        to_add = target_size - width
        right = int(to_add / 2)
        left = to_add - right
        pad_width = ((left, right), (left, right))
        images = [np.pad(image, pad_width, mode="symmetric")]
        deconstruction["padded"] = True
        deconstruction["pad_left"] = left
        deconstruction["pad_right"] = right

    # case where the image is too large
    elif width > target_size:

        # current size is not a multiple of the target size
        if width % target_size != 0:

            # padding
            to_add = target_size * (1 + width // target_size) - width
            right = int(to_add / 2)
            left = to_add - right
            pad_width = ((left, right), (left, right))
            image = np.pad(image, pad_width, mode="symmetric")
            deconstruction["padded"] = True
            deconstruction["pad_left"] = left
            deconstruction["pad_right"] = right
            (width, height) = image.shape

        # cropping
        nb_row = height // target_size
        nb_col = width // target_size
        images = []
        for i_row in range(nb_row):
            row_start = i_row * target_size
            row_end = (i_row + 1) * target_size
            for i_col in range(nb_col):
                col_start = i_col * target_size
                col_end = (i_col + 1) * target_size
                image_ = image[row_start:row_end, col_start:col_end]
                images.append(image_)
        deconstruction["cropped"] = True
        deconstruction["nb_row"] = nb_row
        deconstruction["nb_col"] = nb_col

    else:
        images = [image.copy()]

    # store number of images created from the original one
    deconstruction["nb_images"] = len(images)

    return images, deconstruction


def reconstruct_image(images, deconstruction):
    """Reconstruct an image based on the information stored during the
    deconstruction process (padding and cropping).

    Parameters
    ----------
    images : List[np.ndarray] or np.ndarray
        Images used to reconstruct an image with the original width and height.
    deconstruction : dict
        Information of the deconstruction process.

    Returns
    -------
    reconstructed_image : np.ndarray
        Image with the original width and height.

    """
    # TODO adapt to non squared images
    # TODO add an overlap in the crop
    # TODO handle the different overlapped label values
    # check parameters
    check_parameter(images=(np.ndarray, list),
                    deconstruction=dict)
    if isinstance(images, np.ndarray):
        images = [images]
    for image_ in images:
        check_array(image_,
                    ndim=2,
                    dtype=[np.uint8, np.uint16,
                           np.float32, np.float64,
                           bool],
                    allow_nan=False)

    # case where the original image was padded then cropped
    if deconstruction["padded"] and deconstruction["cropped"]:

        # reconstruct the padded image (cropped => padded - original)
        nb_row = deconstruction["nb_row"]
        nb_col = deconstruction["nb_col"]
        image_ = images[0]
        (cropped_width, cropped_height) = image_.shape
        reconstructed_image = np.zeros(
            (nb_row * cropped_height, nb_col * cropped_width),
            dtype=image_.dtype)
        i = 0
        for i_row in range(nb_row):
            row_ = i_row * cropped_height
            _row = (i_row + 1) * cropped_height
            for i_col in range(nb_col):
                col_ = i_col * cropped_width
                _col = (i_col + 1) * cropped_width
                reconstructed_image[row_:_row, col_:_col] = images[i]
                i += 1

        # reconstruct the original image (cropped - padded => original)
        left = deconstruction["pad_left"]
        right = deconstruction["pad_right"]
        reconstructed_image = reconstructed_image[left:-right, left:-right]

    # case where the original image was padded only
    elif deconstruction["padded"] and not deconstruction["cropped"]:

        # reconstruct the original image from a padding (padded => original)
        left = deconstruction["pad_left"]
        right = deconstruction["pad_right"]
        reconstructed_image = images[0][left:-right, left:-right]

    # case where the original image was cropped only
    elif not deconstruction["padded"] and deconstruction["cropped"]:

        # reconstruct the original image from a cropping (cropped => original)
        nb_row = deconstruction["nb_row"]
        nb_col = deconstruction["nb_col"]
        image_ = images[0]
        (cropped_width, cropped_height) = image_.shape
        reconstructed_image = np.zeros(
            (nb_row * cropped_height, nb_col * cropped_width),
            dtype=image_.dtype)
        i = 0
        for i_row in range(nb_row):
            row_ = i_row * cropped_height
            _row = (i_row + 1) * cropped_height
            for i_col in range(nb_col):
                col_ = i_col * cropped_width
                _col = (i_col + 1) * cropped_width
                reconstructed_image[row_:_row, col_:_col] = images[i]
                i += 1

    # case where no deconstruction happened
    else:
        reconstructed_image = images[0].copy()

    return reconstructed_image


# ### Coordinates data cleaning ###

def clean_simulated_data(data, data_cell, label_encoder=None,
                         path_output=None):
    """Clean simulated dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with all the simulated cells, the coordinates of their
        different elements and the localization pattern used to simulate them.
    data_cell : pandas.DataFrame
        Dataframe with the 2D coordinates of the nucleus and the cytoplasm of
        actual cells used to simulate data.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Label encoder from string to integer.
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
        Cell id to remove from data because of rna coordinates
    label_encoder : sklearn.preprocessing.LabelEncoder
        Label encoder from string to integer.

    """
    # check dataframes and parameters
    check_parameter(label_encoder=(type(LabelEncoder()), type(None)),
                    path_output=(str, type(None)))
    check_df(data,
             features=["name_img_BGD", "pos_cell", "RNA_pos", "cell_ID",
                       "pattern_level", "pattern_name"],
             features_nan=["name_img_BGD", "pos_cell", "RNA_pos", "cell_ID",
                           "pattern_level", "pattern_name"])
    check_df(data_cell,
             features=["name_img_BGD", "pos_cell", "pos_nuc"],
             features_nan=["name_img_BGD", "pos_cell", "pos_nuc"])

    # filter invalid simulated cell backgrounds
    data_clean, background_to_remove, id_volume = _clean_volume(data, data_cell)

    # filter invalid simulated rna spots
    data_clean, id_rna = _clean_rna(data_clean)

    # make the feature 'n_rna' consistent
    data_clean["nb_rna"] = data_clean.apply(
        lambda row: len(row["RNA_pos"]),
        axis=1)

    # remove useless features
    data_final = data_clean.loc[:, ['RNA_pos', 'cell_ID', 'pattern_level',
                                    'pattern_name', 'pos_cell', 'pos_nuc',
                                    "nb_rna"]]

    # encode the label
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_str = set(data_final.loc[:, "pattern_name"])
        label_encoder.fit(label_str)
    data_final.loc[:, "label"] = label_encoder.transform(
        data_final.loc[:, "pattern_name"])

    # reset index
    data_final.reset_index(drop=True, inplace=True)

    # save cleaned dataset
    if path_output is not None:
        data_final.to_pickle(path_output)

    return data_final, background_to_remove, id_volume, id_rna, label_encoder


def _clean_volume(data, data_cell):
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
    data_cell.loc[:, "valid_volume"] = data_cell.apply(
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
    invalid_rows = data.loc[:, "name_img_BGD"].isin(background_to_remove)
    data_clean = data.loc[~invalid_rows, :]

    return data_clean, background_to_remove, id_to_remove


def _check_volume(cyt_coord, nuc_coord):
    """Check nucleus coordinates are not outside the boundary of the cytoplasm.

    Parameters
    ----------
    cyt_coord : pandas.Series
        Coordinates of the cytoplasm membrane.
    nuc_coord : pandas.Series
        Coordinates of the nucleus border.

    Returns
    -------
    _ : bool
        Tell if the cell volume is valid or not.

    """
    # get coordinates
    cyt_coord = np.array(cyt_coord)
    nuc_coord = np.array(nuc_coord)

    # complete coordinates
    list_coord = complete_coordinates_2d([cyt_coord, nuc_coord])
    cyt_coord, nuc_coord = list_coord[0], list_coord[1]

    # get image shape
    max_x = max(cyt_coord[:, 0].max() + 5, nuc_coord[:, 0].max() + 5)
    max_y = max(cyt_coord[:, 1].max() + 5, nuc_coord[:, 1].max() + 5)
    image_shape = (max_x, max_y)

    # build the dense representation for the cytoplasm and the nucleus
    cyt = from_coord_to_image(cyt_coord, image_shape=image_shape)
    nuc = from_coord_to_image(nuc_coord, image_shape=image_shape)

    # check if the volume is valid
    mask_cyt = ndi.binary_fill_holes(cyt)
    mask_nuc = ndi.binary_fill_holes(nuc)
    frame = np.zeros(image_shape)
    diff = frame - mask_cyt + mask_nuc
    diff = (diff > 0).sum()

    if diff > 0:
        return False
    else:
        return True


def _clean_rna(data):
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
    data.loc[:, "valid_rna"] = data.apply(
        lambda row: _check_rna(row["pos_cell"], row["RNA_pos"]),
        axis=1)

    # get id of the invalid cells
    id_to_remove = []
    for i in data.index:
        if np.logical_not(data.loc[i, "valid_rna"]):
            id_to_remove.append(i)

    # remove invalid simulated cells
    data_clean = data.loc[data.loc[:, "valid_rna"], :]

    return data_clean, id_to_remove


def _check_rna(cyt_coord, rna_coord):
    """Check rna spots coordinates are not outside the boundary of the
    cytoplasm.

    Parameters
    ----------
    cyt_coord : pandas.Series
        Coordinates of the cytoplasm membrane.
    rna_coord : pandas.Series
        Coordinates of the rna spots.

    Returns
    -------
    _ : bool
        Tell if the rna spots are valid or not.

    """
    # get coordinates
    cyt_coord = np.array(cyt_coord)
    if not isinstance(rna_coord[0], list):
        # it means we have only one spot
        return False
    rna_coord = np.array(rna_coord)

    # check if the coordinates are positive
    if rna_coord.min() < 0:
        return False

    # complete coordinates
    cyt_coord = complete_coordinates_2d([cyt_coord])[0]

    # get image shape
    max_x = int(max(cyt_coord[:, 0].max() + 5, rna_coord[:, 0].max() + 5))
    max_y = int(max(cyt_coord[:, 1].max() + 5, rna_coord[:, 1].max() + 5))
    image_shape = (max_x, max_y)

    # build the dense representation for the cytoplasm and the rna
    cyt = from_coord_to_image(cyt_coord, image_shape=image_shape)
    rna = from_coord_to_image(rna_coord, image_shape=image_shape)

    # check if the coordinates are valid
    mask_cyt = ndi.binary_fill_holes(cyt)
    frame = np.zeros(image_shape)
    diff = frame - mask_cyt + rna
    diff = (diff > 0).sum()

    if diff > 0:
        return False
    else:
        return True
