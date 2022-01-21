# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.multistack subpackage.
"""

import os
import re
import copy

import bigfish.stack as stack


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
    stack.check_parameter(
        recipe=dict,
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
                        path = get_path_from_recipe(
                            recipe,
                            data_directory,
                            fov=fov,
                            r=r,
                            c=c,
                            z=z)
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
    stack.check_parameter(recipe=dict)

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
    stack.check_parameter(
        recipe=dict,
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
    stack.check_parameter(recipe=dict)

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
    stack.check_parameter(recipe=dict)

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
    stack.check_parameter(data_map=list)
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

