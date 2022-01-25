# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for bigfish.plot subpackage.
"""

import matplotlib.pyplot as plt
import numpy as np


def save_plot(path_output, ext):
    """Save the plot.

    Parameters
    ----------
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

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
    """Create a shuffled colormap to display segmentation masks.

    Returns
    -------
    colormap : ListedColormap object
        Colormap for matplotlib.

    """
    values = np.linspace(0, 1, 256)
    np.random.shuffle(values)
    colormap = plt.cm.colors.ListedColormap(plt.cm.YlGnBu(values))
    colormap.set_bad(color='black')

    return colormap
