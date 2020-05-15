# -*- coding: utf-8 -*-

"""
Function used to read data from various sources and store them in a
multidimensional tensor (np.ndarray) or a dataframe (pandas.DataFrame).
"""

import pickle
import warnings

import numpy as np
import pandas as pd

from skimage import io
from .utils import check_array, check_df


# ### Read ###

def read_image(path):
    """Read an image with the .png, .tif or .tiff extension.

    The input image should be in 2-d or 3-d, with unsigned integer 8 or 16
    bits, integer

    Parameters
    ----------
    path : str
        Path of the image to read.

    Returns
    -------
    tensor : ndarray, np.uint or np.int
        A 2-d or 3-d tensor with spatial dimensions.

    """
    # TODO allow more input dtype
    # read image
    tensor = io.imread(path)

    # check the image is in unsigned integer 16 bits with 2 or 3 dimensions
    check_array(tensor,
                dtype=[np.uint8, np.uint16, np.int64],
                ndim=[2, 3],
                allow_nan=False)

    return tensor


def read_cell_json(path):
    """Read the json file 'cellLibrary.json' used by FishQuant.

    Parameters
    ----------
    path : str
        Path of the json file to read.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the 2D coordinates of the nucleus and the cytoplasm of
        actual cells used to simulate data.

    """
    # read json file and open it in a dataframe
    df = pd.read_json(path)

    # check the output has the right features
    check_df(df,
             features=["name_img_BGD", "pos_cell", "pos_nuc"],
             features_nan=["name_img_BGD", "pos_cell", "pos_nuc"])

    return df


def read_rna_json(path):
    """Read json files simulated by FishQuant with RNA 3D coordinates.

    Parameters
    ----------
    path : str
        Path of the json file to read.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with 3D coordinates of the simulated RNA, localization
        pattern used to simulate them and its strength.

    """
    # read json file and open it in a dataframe
    df = pd.read_json(path)

    # check the output has the right number of features
    if df.shape[1] != 9:
        raise ValueError("The file does not seem to have the right number of "
                         "features. It returns {0} dimensions instead of 9."
                         .format(df.ndim))

    # check the output has the right features
    expected_features = ['RNA_pos', 'cell_ID', 'mRNA_level_avg',
                         'mRNA_level_label', 'n_RNA', 'name_img_BGD',
                         'pattern_level', 'pattern_name', 'pattern_prop']
    check_df(df,
             features=expected_features,
             features_nan=expected_features)

    return df


def read_pickle(path):
    """Read serialized pickle file.

    Parameters
    ----------
    path : str
        Path of the file to read.

    Returns
    -------
    data = pandas.DataFrame or np.ndarray
        Data store in the pickle file (an image or coordinates with labels and
        metadata).

    """
    # open the file and read it
    with open(path, mode='rb') as f:
        data = pickle.load(f)

    return data


# ### Write ###

def save_image(image, path):
    """Save a 2-d or 3-d image.

    Parameters
    ----------
    image : np.ndarray
        Tensor to save with shape (z, y, x) or (y, x).
    path : str
        Path of the saved image.

    Returns
    -------

    """
    # check image
    check_array(image,
                dtype=[np.uint8, np.uint16, np.int64,
                       np.float32, np.float64,
                       bool],
                ndim=[2, 3],
                allow_nan=False)

    # save image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(path, image)

    # import warnings
    # warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    # warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

    return
