# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.plot subpackage.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap


def save_plot(path_output, ext):
    """Save the plot.

    Parameters
    ----------
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # add extension at the end of the filename
    extension = "." + ext
    if extension not in path_output:
        path_output += extension

    # save the plot
    if isinstance(ext, str):
        # add extension at the end of the filename
        extension = "." + ext
        if extension not in path_output:
            path_output += extension
        plt.savefig(path_output, format=ext)
    elif isinstance(ext, list):
        for ext_ in ext:
            # add extension at the end of the filename
            extension = "." + ext_
            if extension not in path_output:
                path_output += extension
            plt.savefig(path_output, format=ext_)
    else:
        Warning("Plot is not saved because the extension is not valid: "
                "{0}.".format(ext))

    return


def get_minmax_values(tensor):
    """Get the minimum and maximum value of the image according to its dtype.

    Parameters
    ----------
    tensor : np.ndarray
        A 2-d, 3-d or 5-d tensor with shape (y, x), (z, y, x) or
        (r, c, z, y, x) respectively.

    Returns
    -------
    vmin : int
        Minimum value display in the plot.
    vmax : int
        Maximum value display in the plot.

    """
    vmin, vmax = None, None
    if tensor.dtype == np.uint8:
        vmin, vmax = 0, 255
    elif tensor.dtype == np.uint16:
        vmin, vmax = 0, 65535
    elif tensor.dtype == np.float32:
        vmin, vmax = 0, 1
    elif tensor.dtype == np.float64:
        vmin, vmax = 0, 1
    elif tensor.dtype == bool:
        vmin, vmax = 0, 1

    return vmin, vmax


def create_colormap():
    """Create a customized colormap to display segmentation masks.

    Returns
    -------
    colormap : ListedColormap object
        Colormap for matplotlib.

    """
    colors = ['#525252', '#b8e186', '#8c510a', '#9970ab', '#de77ae', '#bdbdbd',
              '#d6604d', '#c51b7d', '#d1e5f0', '#969696', '#f4a582', '#fddbc7',
              '#fde0ef', '#d9d9d9', '#35978f', '#dfc27d', '#f6e8c3', '#b35806',
              '#5aae61', '#7f3b08', '#4d9221', '#543005', '#d9f0d3', '#4575b4',
              '#8e0152', '#f46d43', '#c2a5cf', '#a50026', '#7fbc41', '#40004b',
              '#fee0b6', '#bf812d', '#313695', '#4393c3', '#737373', '#f1b6da',
              '#67001f', '#e7d4e8', '#e0f3f8', '#e6f5d0', '#74add1', '#053061',
              '#00441b', '#fee090', '#e08214', '#f5f5f5', '#01665e', '#f7f7f7',
              '#1b7837', '#b2abd2', '#542788', '#2d004b', '#b2182b', '#d8daeb',
              '#92c5de', '#ffffbf', '#762a83', '#8073ac', '#fdae61', '#a6dba0',
              '#80cdc1', '#003c30', '#d73027', '#fdb863', '#2166ac', '#abd9e9',
              '#276419', '#252525', '#c7eae5'] * 50
    colors.insert(0, "#000000")

    colormap = ListedColormap(colors, name='color_mask')

    return colormap
