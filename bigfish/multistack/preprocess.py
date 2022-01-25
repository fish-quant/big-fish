# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions used to build 4D or 5D images.
"""

import numpy as np

import bigfish.stack as stack

from .utils import check_recipe
from .utils import check_datamap
from .utils import fit_recipe
from .utils import get_path_from_recipe
from .utils import get_nb_element_per_dimension
from .utils import count_nb_fov


# TODO only read in memory one or several channels (and not the entire image)
# TODO allow new keys to define a recipe

# ### Building stack ###

def build_stacks(data_map, input_dimension=None, sanity_check=False,
                 return_origin=False):
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
         "fov": str or List[str], (optional)
         "z": str or List[str],   (optional)
         "c": str or List[str],   (optional)
         "r": str or List[str],   (optional)
         "ext": str,              (optional)
         "opt": str,              (optional)
         "pattern": str
         }

    - A field of view is defined by a string common to every images belonging
    to the same field of view ("fov").
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
        Number of dimensions of the loaded files. Can speed up the function if
        provided.
    sanity_check : bool
        Check the validity of the loaded tensor. Can slow down the function.
    return_origin : bool
        Return the input directory and the recipe used to build the stack.

    Returns
    -------
    tensor : np.ndarray
        Tensor with shape (round, channel, z, y, x).
    input_directory : str
        Path of the input directory from where the tensor is built.
    recipe : dict
        Recipe used to build the tensor.
    i_fov : int
        Index of the fov to build (for a specific recipe).

    """
    # check parameters
    stack.check_parameter(
        data_map=list,
        input_dimension=(int, type(None)),
        sanity_check=bool,
        return_origin=bool)
    check_datamap(data_map)

    # load and generate tensors for each recipe-folder pair
    for recipe, input_folder in data_map:

        # load and generate tensors for each fov stored in a recipe
        nb_fov = count_nb_fov(recipe)
        for i_fov in range(nb_fov):
            tensor = build_stack(
                recipe,
                input_folder,
                input_dimension=input_dimension,
                sanity_check=sanity_check,
                i_fov=i_fov)
            if return_origin:
                yield tensor, input_folder, recipe, i_fov
            else:
                yield tensor


def build_stack(recipe, input_folder, input_dimension=None, sanity_check=False,
                i_fov=0):
    """Build a 5-d stack from the same field of view (fov).

    The recipe dictionary for one field of view takes the form:

        {
         "fov": str or List[str], (optional)
         "z": str or List[str],   (optional)
         "c": str or List[str],   (optional)
         "r": str or List[str],   (optional)
         "ext": str,              (optional)
         "opt": str,              (optional)
         "pattern": str
         }

    - A field of view is defined by a string common to every images belonging
    to the same field of view ("fov").
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
        Number of dimensions of the loaded files. Can speed up the function if
        provided.
    i_fov : int
        Index of the fov to build.
    sanity_check : bool
        Check the validity of the loaded tensor. Can slow down the function.

    Returns
    -------
    tensor : np.ndarray
        Tensor with shape (round, channel, z, y, x).

    """
    # check parameters
    check_recipe(recipe)
    stack.check_parameter(
        input_folder=str,
        input_dimension=(int, type(None)),
        i_fov=int,
        sanity_check=bool)

    # build stack from recipe and tif files
    tensor = _load_stack(recipe, input_folder, input_dimension, i_fov)

    # check the validity of the loaded tensor
    if sanity_check:
        stack.check_array(
            tensor,
            dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
                   np.int8, np.int16, np.int32, np.int64,
                   np.float16, np.float32, np.float64,
                   bool],
            ndim=5,
            allow_nan=False)

    return tensor


def _load_stack(recipe, input_folder, input_dimension=None, i_fov=0):
    """Build a 5-d tensor from the same field of view (fov).

    The function stacks a set of images using a recipe mapping the
    different images with the dimensions they represent. Each stacking step
    add a new dimension to the original tensors (eg. we stack 2-d images with
    the same xy coordinates to get a 3-d image). If the files we need to build
    a new dimension are not included in the recipe, an empty dimension is
    added. This operation is repeated until we get a 5-d tensor. We first
    operate on the z dimension, then the channels and eventually the rounds.

    The recipe dictionary for one field of view takes the form:

        {
         "fov": str or List[str], (optional)
         "z": str or List[str],   (optional)
         "c": str or List[str],   (optional)
         "r": str or List[str],   (optional)
         "ext": str,              (optional)
         "opt": str,              (optional)
         "pattern": str
         }

    - A field of view is defined by a string common to every images belonging
    to the same field of view ("fov").
    - At least every images are in 2-d with x and y dimensions. So we need to
    mention the round-dimension, the channel-dimension and the z-dimension to
    add ("r", "c" and "z"). For these keys, we provide a list of
    strings to identify the images to stack.
    - An extra information to identify the files to stack in the input folder
    can be provided with the file extension "ext" (usually 'tif' or 'tiff') or
    an optional morpheme ("opt").
    - A pattern used to get the filename ("pattern").

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
    stack : np.ndarray
        Tensor with shape (round, channel, z, y, x).

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
        stack_ = _build_stack_from_2d(
            recipe,
            input_folder,
            fov=i_fov,
            nb_r=nb_r,
            nb_c=nb_c,
            nb_z=nb_z)
    elif input_dimension == 3:
        stack_ = _build_stack_from_3d(
            recipe,
            input_folder,
            fov=i_fov,
            nb_r=nb_r,
            nb_c=nb_c)
    elif input_dimension == 4:
        stack_ = _build_stack_from_4d(
            recipe,
            input_folder,
            fov=i_fov,
            nb_r=nb_r)
    elif input_dimension == 5:
        stack_ = _build_stack_from_5d(recipe, input_folder, fov=i_fov)
    else:
        raise ValueError("Files do not have the right number of dimensions: "
                         "{0}. The files we stack should have between 2 and "
                         "5 dimensions.".format(input_dimension))

    return stack_


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
    tensor_5d : np.ndarray
        Tensor with shape (round, channel, z, y, x).

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
                path = get_path_from_recipe(
                    recipe,
                    input_folder,
                    fov=fov,
                    r=r,
                    c=c,
                    z=z)
                tensor_2d = stack.read_image(path)
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
    tensor_5d : np.ndarray
        Tensor with shape (round, channel, z, y, x).

    """
    # load and stack successively channel elements then round elements
    tensors_4d = []
    for r in range(nb_r):

        # load and stack channel elements (3-d tensors)
        tensors_3d = []
        for c in range(nb_c):
            path = get_path_from_recipe(
                recipe,
                input_folder,
                fov=fov,
                r=r,
                c=c)
            tensor_3d = stack.read_image(path)
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
    tensor_5d : np.ndarray
        Tensor with shape (round, channel, z, y, x).

    """
    # load each file from a new round element and stack them
    tensors_4d = []
    for r in range(nb_r):
        path = get_path_from_recipe(recipe, input_folder, fov=fov, r=r)
        tensor_4d = stack.read_image(path)
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
    tensor_5d : np.ndarray
        Tensor with shape (round, channel, z, y, x).

    """
    # the recipe can only contain one file with a 5-d tensor per fov
    path = get_path_from_recipe(recipe, input_folder, fov=fov)
    tensor_5d = stack.read_image(path)

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
    image = stack.read_image(path)
    nb_dim = image.ndim

    return nb_dim


def build_stack_no_recipe(paths, input_dimension=None, sanity_check=False):
    """Build 5-d stack without recipe.

    Parameters
    ----------
    paths : List[str]
        List of the paths to stack.
    input_dimension : str
        Number of dimensions of the loaded files. Can speed up the function if
        provided.
    sanity_check : bool
        Check the validity of the loaded tensor. Can slow down the function.

    Returns
    -------
    tensor : np.ndarray
        Tensor with shape (round, channel, z, y, x).

    """
    # check parameters
    stack.check_parameter(
        paths=(str, list),
        input_dimension=(int, type(None)),
        sanity_check=bool)

    # build stack from tif files
    tensor = _load_stack_no_recipe(paths, input_dimension)

    # check the validity of the loaded tensor
    if sanity_check:
        stack.check_array(
            tensor,
            dtype=[np.uint8, np.uint16, np.uint32,
                   np.int8, np.int16, np.int32,
                   np.float16, np.float32, np.float64,
                   bool],
            ndim=5,
            allow_nan=False)

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
        Tensor with shape (round, channel, z, y, x).

    """
    # load an image and get the number of dimensions
    if input_dimension is None:
        testfile = stack.read_image(paths[0])
        input_dimension = testfile.ndim

    # get stacks
    stacks = []
    for path in paths:
        s = stack.read_image(path)
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
                         "{0}. The files we stack should have between 2 and "
                         "5 dimensions.".format(input_dimension))

    return tensor_5d

