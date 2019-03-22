# -*- coding: utf-8 -*-

"""
Function used to read data from various sources and store them in a
multidimensional tensor (np.ndarray) or a dataframe (pandas.DataFrame).
"""

import pickle

import numpy as np
import pandas as pd

from skimage import io
from .utils import check_array, check_features_df


def read_tif(path):
    """Read an image with the .tif or .tiff extension.

    The input image should be in 2-d or 3-d, with unsigned integer 16 bits.

    Parameters
    ----------
    path : str
        Path of the image to read.

    Returns
    -------
    tensor : ndarray, np.uint16
        A 2-d or 3-d tensor with spatial dimensions.

    """
    # read image
    tensor = io.imread(path)

    # check the image is in unsigned integer 16 bits with 2 or 3 dimensions
    check_array(tensor, dtype=np.uint16, ndim=[2, 3])

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
    check_features_df(df, features=["name_img_BGD", "pos_cell", "pos_nuc"])

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
    check_features_df(df, features=expected_features)

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
