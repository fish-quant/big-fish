# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Function to plot quality control indicators.
"""

import bigfish.stack as stack

import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot


def plot_sharpness(focus_measures, labels=None, title=None, framesize=(5, 5),
                   size_title=20, size_axes=15, size_legend=15,
                   path_output=None, ext="png", show=True):
    """

    Parameters
    ----------
    focus_measures : np.ndarray or List[np.ndarray]
        A list of 1-d array with the sharpness measure for each z-slices.
    labels : List[str]
        List of labels for the different measures to compare.
    title : str
        Title of the plot.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    size_title : int
        Size of the title.
    size_axes : int
        Size of the axes label.
    size_legend : int
        Size of the legend.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    Returns
    -------

    """
    # enlist image if necessary
    if isinstance(focus_measures, np.ndarray):
        focus_measures = [focus_measures]

    # check parameters
    stack.check_parameter(focus_measures=list,
                          labels=(list, type(None)),
                          title=(str, list, type(None)),
                          framesize=tuple,
                          size_title=int,
                          size_axes=int,
                          size_legend=int,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)
    length = 0
    for focus_measure in focus_measures:
        stack.check_array(focus_measure,
                          ndim=1,
                          dtype=[np.float32, np.float64])
        length = max(length, focus_measure.size)

    # plot
    plt.figure(figsize=framesize)
    y = np.array([i for i in range(length)])
    for i, focus_measure in enumerate(focus_measures):
        if labels is not None:
            plt.plot(focus_measure, y, label=labels[i])
        else:
            plt.plot(focus_measure, y)

    # axes
    if title is not None :
        plt.title(title, fontweight="bold", fontsize=size_title)
    plt.xlabel("sharpness measure", fontweight="bold", fontsize=size_axes)
    plt.ylabel("z-slices", fontweight="bold", fontsize=size_axes)
    if labels is not None:
        plt.legend(prop={'size': size_legend})

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_illumination_surface(illumination_surface, r=0, framesize=(15, 15),
                              titles=None, path_output=None, ext="png"):
    """Subplot the yx plan of the dimensions of an illumination surface for
    all channels.

    Parameters
    ----------
    illumination_surface : np.ndarray, np.float
        A 4-d tensor with shape (r, c, y, x) approximating the average
        differential of illumination in our stack of images, for each channel
        and each round.
    r : int
        Index of the round to keep.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    titles : List[str]
        Titles of the subplots (one per channel).
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # TODO add title in the plot and remove axes
    # TODO add parameter for vmin and vmax
    # check tensor
    stack.check_array(illumination_surface,
                      ndim=4,
                      dtype=[np.float32, np.float64])

    # get the number of channels
    nb_channels = illumination_surface.shape[1]

    # plot
    fig, ax = plt.subplots(1, nb_channels, sharex='col', figsize=framesize)
    for i in range(nb_channels):
        ax[i].imshow(illumination_surface[r, i, :, :])
        if titles is not None:
            ax[i].set_title(titles[i], fontweight="bold", fontsize=15)
    plt.tight_layout()
    save_plot(path_output, ext)
    plt.show()

    return

