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
    """Plot focus measures.

    Parameters
    ----------
    focus_measures : np.ndarray or List[np.ndarray]
        A list of 1-d arrays with the sharpness measure for each z-slices.
    labels : List[str] or None
        List of labels for the different measures to compare.
    title : str or None
        Title of the plot.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    size_title : int
        Size of the title.
    size_axes : int
        Size of the axes label.
    size_legend : int
        Size of the legend.
    path_output : str or None
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    Returns
    -------

    """
    # enlist values if necessary
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
    if title is not None:
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


def plot_snr_spots(snr_spots, labels=None, colors=None, x_lim=None, y_lim=None,
                   title=None, framesize=(10, 5), size_title=20, size_axes=15,
                   size_legend=15, path_output=None, ext="png", show=True):
    """Plot Signal-to-Noise Ratio computed for all detected spots.

    Parameters
    ----------
    snr_spots : List[np.ndarray] or np.ndarray
        A list of 1-d arrays with the SNR computed for every spot of an image.
        One array per image.
    labels : List[str], str or None
        Labels or the curves.
    colors : List[str], str or None
        Colors or the curves.
    x_lim : tuple or None
        Limits of the x-axis.
    y_lim : tuple or None
        Limits of the y-axis.
    title : str or None
        Title of the plot.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    size_title : int
        Size of the title.
    size_axes : int
        Size of the axes label.
    size_legend : int
        Size of the legend.
    path_output : str or None
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    Returns
    -------

    """
    # check parameters
    stack.check_parameter(snr_spots=(list, np.ndarray),
                          labels=(list, str, type(None)),
                          colors=(list, str, type(None)),
                          x_lim=(tuple, type(None)),
                          y_lim=(tuple, type(None)),
                          title=(str, type(None)),
                          framesize=tuple,
                          size_title=int,
                          size_axes=int,
                          size_legend=int,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)

    # enlist values if necessary
    if isinstance(snr_spots, np.ndarray):
        snr_spots = [snr_spots]
    if labels is not None and isinstance(labels, str):
        labels = [labels]
    if colors is not None and isinstance(colors, str):
        colors = [colors]

    # check arrays
    for snr_spots_ in snr_spots:
        stack.check_array(snr_spots_,
                          ndim=1,
                          dtype=[np.float32, np.float64])

    # check number of parameters
    if labels is not None and len(snr_spots) != len(labels):
        raise ValueError("The number of labels provided ({0}) differs from "
                         "the number of arrays to plot ({1})."
                         .format(len(labels), len(snr_spots)))
    if colors is not None and len(snr_spots) != len(colors):
        raise ValueError("The number of colors provided ({0}) differs from "
                         "the number of arrays to plot ({1})."
                         .format(len(colors), len(snr_spots)))

    # frame
    plt.figure(figsize=framesize)

    # plot
    for i, snr_spots_ in enumerate(snr_spots):
        values = sorted(snr_spots_, reverse=True)
        if labels is None and colors is None:
            plt.plot(values, lw=2)
        elif labels is None and colors is not None:
            color = colors[i]
            plt.plot(values, lw=2, c=color)
        elif labels is not None and colors is None:
            label = labels[i]
            plt.plot(values, lw=2, label=label)
        else:
            label = labels[i]
            color = colors[i]
            plt.plot(values, lw=2, c=color, label=label)

    # axes
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=size_title)
    plt.xlabel("Detected Spots", fontweight="bold", fontsize=size_axes)
    plt.ylabel("Signal-to-Noise Ratio", fontweight="bold", fontsize=size_axes)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
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


