# -*- coding: utf-8 -*-

"""
Functions used to format and clean any input loaded in bigfish.
"""

import os

import numpy as np
import pandas as pd

from bigfish.stack.loader import read_tif, read_cell_json, read_rna_json

from skimage import img_as_ubyte, img_as_float32
from skimage.morphology.selem import square
from skimage.filters import rank


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
    # read the cell data (nucleus + cytoplasm)
    df_cell = read_cell_json(path_cell)
    print("data cell: {0}".format(df_cell.shape))

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
    print("data rna: {0}".format(df_rna.shape))

    # merge the dataframe
    df = pd.merge(df_rna, df_cell, on="name_img_BGD")
    print("data: {0}".format(df.shape))

    # save output
    if path_output is not None:
        df.to_pickle(path_output)

    return df, df_cell, df_rna






def build_stack(recipe, input_folder, input_dimension=None):
    """

    Parameters
    ----------
    recipe
    input_folder
    input_dimension

    Returns
    -------

    """
    if input_dimension is None:
        fov_str = recipe["fov"]
        ext_str = "." + recipe["ext"]
        filenames = [filename
                     for filename in os.listdir(input_folder)
                     if fov_str in filename and ext_str in filename]
        path = os.path.join(input_folder, filenames[0])
        test = read_tif(path)
        input_dimension = test.ndim

    if input_dimension == 2:
        stack = _build_stack_from_2d(recipe, input_folder)
    elif input_dimension == 3:
        stack = _build_stack_from_3d(recipe, input_folder)
    elif input_dimension == 4:
        stack = _build_stack_from_4d(recipe, input_folder)
    else:
        # TODO Error message
        raise ValueError("Blablabla")

    return stack


def check_recipe(recipe):
    """Check and validate a recipe.

    Parameters
    ----------
    recipe : dict
        Map the images according to their field of view, their round,
        their channel and their spatial dimensions.

    Returns
    -------
    expected_dimension : int
        The number of dimensions expected in the tensors used with this
        recipe. A 0 value means the recipe is not valid.

    """
    expected_dimension = 0
    # check recipe is a dictionary with the "fov" key
    if not isinstance(recipe, dict) or "fov" not in recipe:
        return expected_dimension

    # determine the minimum number of dimensions expected for the tensors
    if ("round" in recipe and isinstance(recipe["round"], list)
            and len(recipe["round"]) > 0):
        expected_dimension = 4
    if ("channel" in recipe and isinstance(recipe["channel"], list)
            and len(recipe["channel"]) > 0):
        expected_dimension = 3
    if ("z" in recipe and isinstance(recipe["z"], list)
            and len(recipe["z"]) > 0):
        expected_dimension = 2

    return expected_dimension


def _extract_recipe(recipe):
    """

    Parameters
    ----------
    recipe

    Returns
    -------

    """
    # check recipe
    expected_dimension = check_recipe(recipe)
    if expected_dimension == 0:
        raise Exception("The recipe is not valid")

    # we collect the different morphemes we use to identify the images
    if ("round" in recipe
            and isinstance(recipe["round"], list)
            and len(recipe["round"]) > 0):
        l_round = recipe["round"]
    else:
        l_round = [""]

    if ("channel" in recipe
            and isinstance(recipe["channel"], list)
            and len(recipe["channel"]) > 0):
        l_channel = recipe["channel"]
    else:
        l_channel = [""]

    if ("z" in recipe
            and isinstance(recipe["z"], list)
            and len(recipe["z"]) > 0):
        l_z = recipe["z"]
    else:
        l_z = [""]

    return expected_dimension, l_round, l_channel, l_z


def _build_stack_from_2d(recipe, input_folder):
    """

    Parameters
    ----------
    recipe
    input_folder

    Returns
    -------

    """
    # check we can find the tensors to stack from the recipe
    expected_dimension, l_round, l_channel, l_z = _extract_recipe(recipe)

    # stack the images
    fov_str = recipe["fov"]
    ext_str = "." + recipe["ext"]

    tensors_4d = []
    for round_str in l_round:
        if round_str != "":
            round_str = "_" + round_str

        tensors_3d = []
        for channel_str in l_channel:
            if channel_str != "":
                channel_str = "_" + channel_str

            tensors_2d = []
            for z_str in l_z:
                if z_str != "":
                    z_str = "_" + z_str

                filename = fov_str + z_str + channel_str + round_str + ext_str

                path = os.path.join(input_folder, filename)
                tensor_2d = read_tif(path)
                tensors_2d.append(tensor_2d)

            tensor_3d = np.stack(tensors_2d, axis=0)
            tensors_3d.append(tensor_3d)

        tensor_4d = np.stack(tensors_3d, axis=0)
        tensors_4d.append(tensor_4d)

    tensor_5d = np.stack(tensors_4d, axis=0)

    return tensor_5d


def _build_stack_from_3d(recipe, input_folder):
    """

    Parameters
    ----------
    recipe
    input_folder

    Returns
    -------

    """
    # check we can find the tensors to stack from the recipe
    expected_dimension, l_round, l_channel, l_z = _extract_recipe(recipe)

    # stack the images
    fov_str = recipe["fov"]
    ext_str = "." + recipe["ext"]

    tensors_4d = []
    for round_str in l_round:
        if round_str != "":
            round_str = "_" + round_str

        tensors_3d = []
        for channel_str in l_channel:
            if channel_str != "":
                channel_str = "_" + channel_str

            filename = fov_str + channel_str + round_str + ext_str

            path = os.path.join(input_folder, filename)
            tensor_3d = read_tif(path)
            tensors_3d.append(tensor_3d)

        tensor_4d = np.stack(tensors_3d, axis=0)
        tensors_4d.append(tensor_4d)

    tensor_5d = np.stack(tensors_4d, axis=0)

    return tensor_5d


def _build_stack_from_4d(recipe, input_folder):
    """

    Parameters
    ----------
    recipe
    input_folder

    Returns
    -------

    """
    # check we can find the tensors to stack from the recipe
    expected_dimension, l_round, l_channel, l_z = _extract_recipe(recipe)

    # stack the images
    fov_str = recipe["fov"]
    ext_str = "." + recipe["ext"]

    tensors_4d = []
    for round_str in l_round:
        if round_str != "":
            round_str = "_" + round_str

        filename = fov_str + round_str + ext_str

        path = os.path.join(input_folder, filename)
        tensor_4d = read_tif(path)
        tensors_4d.append(tensor_4d)

    tensor_5d = np.stack(tensors_4d, axis=0)

    return tensor_5d


def maximum_projection(tensor):
    """Project the z-dimension of a tensor, keeping the maximum intensity of
    each yx pixel.

    Parameters
    ----------
    tensor : np.ndarray, np.float32
        A 5-d tensor with shape (round, channel, z, y, x).

    Returns
    -------
    projected_tensor : np.ndarray, np.float32
        A 5-d tensor with shape (round, channel, 1, y, x).

    """
    # check tensor dimensions
    if tensor.ndim != 5:
        raise ValueError("Tensor should have 5 dimensions instead of {0}"
                         .format(tensor.ndim))

    # project tensor along the z axis
    projected_tensor = tensor.max(axis=2, keepdims=True)

    return projected_tensor


def focus_measurement_2d(image, neighborhood_size):
    """Helmli and Scherer’s mean method used as a focus metric.

    For each pixel xy in an image, we compute the ratio:

        R(x, y) = mu(x, y) / I(x, y), if mu(x, y) >= I(x, y)

    or

        R(x, y) = I(x, y) / mu(x, y), otherwise

    with I(x, y) the intensity of the pixel xy and mu(x, y) the mean intensity
    of the pixels of its neighborhood.

    Parameters
    ----------
    image : np.ndarray, np.float32
        A 2-d tensor with shape (y, x).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.

    Returns
    -------
    global_focus : np.float32
        Mean value of the ratio computed for every pixels of the image. Can be
        used as a metric to quantify the focus level of an 2-d image.
    ratio : np.ndarray, np.float32
        A 2-d tensor with the R(x, y) computed for each pixel of the original
        image.
    image_filtered_mean : np.ndarray, np.float32
        A 2-d tensor with shape (y, x).

    """

    # scikit-image filter use np.uint dtype (so we cast to np.uint8)
    image_2d = img_as_ubyte(image)

    # filter the image with a mean filter
    selem = square(neighborhood_size)
    image_filtered_mean = rank.mean(image_2d, selem)

    # cast again in np.float32
    image_2d = img_as_float32(image_2d)
    image_filtered_mean = img_as_float32(image_filtered_mean)

    # case where mu(x, y) >= I(x, y)
    mask_1 = image_2d != 0
    out_1 = np.zeros_like(image_filtered_mean, dtype=np.float32)
    ratio_1 = np.divide(image_filtered_mean, image_2d, out=out_1, where=mask_1)
    ratio_1 = np.where(image_filtered_mean >= image_2d, ratio_1, 0)

    # case where I(x, y) > mu(x, y)
    mask_2 = image_filtered_mean != 0
    out_2 = np.zeros_like(image_2d, dtype=np.float32)
    ratio_2 = np.divide(image_2d, image_filtered_mean, out=out_2, where=mask_2)
    ratio_2 = np.where(image_2d > image_filtered_mean, ratio_2, 0)

    # compute ratio and global focus for the entire image
    ratio = ratio_1 + ratio_2
    global_focus = ratio.mean()

    return global_focus, ratio, image_filtered_mean


def focus_measurement_3d(image, neighborhood_size):
    """Helmli and Scherer’s mean method used as a focus metric.

    Parameters
    ----------
    image : np.ndarray, np.float32
        A 3-d tensor with shape (z, y, x).
    neighborhood_size : int
        The size of the square used to define the neighborhood of each pixel.

    Returns
    -------
    ratio : np.ndarray, np.float32
        A 3-d tensor with the R(x, y) computed for each pixel of the original
        3-d image, for each z-slice.
    l_focus : list
        List of the global focus computed for each z-slice.

    """
    # apply focus_measurement_2d for each z-slice
    l_ratio = []
    l_focus = []
    for z in range(image.shape[0]):
        focus, ratio_2d, _ = focus_measurement_2d(image[z], neighborhood_size)
        l_ratio.append(ratio_2d)
        l_focus.append(focus)

    # get 3-d Helmli and Scherer’s ratio
    ratio = np.stack(l_ratio)

    return ratio, l_focus


def get_in_focus(l_focus, proportion):
    """ Select the best in-focus z-slices.

    Parameters
    ----------
    l_focus : array_like
        List of the global focus computed for each z-slice.
    proportion : float or int
        Proportion of z-slices to keep (float between 0 and 1) or number of
        z-slices to keep (integer above 1).

    Returns
    -------
    indices_to_keep : np.array
    """
    # get the number of z-slices to keep
    if proportion < 1 and isinstance(proportion, float):
        n = int(len(l_focus) * proportion)
    else:
        n = int(proportion)

    # select the best z-slices
    indices_to_keep = np.argsort(l_focus)[-n:]

    return indices_to_keep


def one_hot_3d(tensor_2d, depth):
    """Build a 3-d one-hot matrix from a 2-d indices matrix.

    Parameters
    ----------
    tensor_2d : np.ndarray, int
        A 2-d tensor with integer indices and shape (y, x).
    depth : int
        Depth of the 3-d one-hot matrix.

    Returns
    -------
    one_hot : np.ndarray, np.uint8
        A 3-d binary tensor with shape (depth, y, x)

    """
    # initialize the 3-d one-hot matrix
    one_hot = np.zeros((tensor_2d.size, depth), dtype=np.uint8)

    # flatten the matrix to easily one-hot encode it, then reshape it
    one_hot[np.arange(tensor_2d.size), tensor_2d.ravel()] = 1
    one_hot.shape = tensor_2d.shape + (depth,)

    # rearrange the axis
    one_hot = np.moveaxis(one_hot, source=2, destination=0)

    return one_hot


def focus_projection(tensor, channel=0, p=0.75, global_neighborhood_size=30, method="best"):
    """

    Parameters
    ----------
    tensor
    channel
    p
    global_neighborhood_size

    Returns
    -------

    """

    # get 3-d image
    image = tensor[0, channel, :, :, :]

    # measure global focus level for each z-slices
    ratio, l_focus = focus_measurement_3d(image, global_neighborhood_size)

    # remove out-of-focus slices
    indices_to_keep = get_in_focus(l_focus, p)
    in_focus_image = image[indices_to_keep]

    projected_image = None
    if method == "bast":
        # for each pixel, we project the z-slice value with the highest focus
        ratio_2d = np.argmax(ratio[indices_to_keep], axis=0)
        one_hot = one_hot_3d(ratio_2d, depth=len(indices_to_keep))
        projected_image = np.multiply(in_focus_image, one_hot).max(axis=0)
    elif method == "median":
        # for each pixel, we compute the median value of the in-focus z-slices
        projected_image = np.median(in_focus_image, axis=0)
    elif method == "mean":
        # for each pixel, we compute the mean value of the in-focus z-slices
        projected_image = np.median(in_focus_image, axis=0)

    return projected_image, ratio, l_focus




