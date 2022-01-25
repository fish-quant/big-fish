# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Function to plot quality control indicators.
"""

import bigfish.stack as stack
import bigfish.detection as detection
import bigfish.multistack as multistack

import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot


# ### Focus - sharpness ###

def plot_sharpness(focus_measures, labels=None, title=None, framesize=(5, 5),
                   size_title=20, size_axes=15, size_legend=15,
                   path_output=None, ext="png", show=True):
    """Plot focus measures of a 3-d image, at the z-slice level.

    A measure of focus for each z-slice can be computed by averaging the
    pixel-wise focus measure returned from :func:`bigfish.stack.compute_focus`.

    Parameters
    ----------
    focus_measures : np.ndarray or List[np.ndarray]
        A list of 1-d arrays with the sharpness measure for each z-slices.
    labels : List[str] or None
        List of labels for the different measures to compare.
    title : str or None
        Title of the plot.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
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

    """
    # enlist values if necessary
    if isinstance(focus_measures, np.ndarray):
        focus_measures = [focus_measures]

    # check parameters
    stack.check_parameter(
        focus_measures=list,
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
        stack.check_array(
            focus_measure,
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


# ### Elbow plots ###

def plot_elbow(images, voxel_size=None, spot_radius=None, log_kernel_size=None,
               minimum_distance=None, title=None, framesize=(5, 5),
               size_title=20, size_axes=15, size_legend=15, path_output=None,
               ext="png", show=True):
    """Plot the elbow curve that allows an automated spot detection.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images with shape (z, y, x) or (y, x). The same threshold is
        applied to every images.
    voxel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions. Not used if 'log_kernel_size' and 'minimum_distance' are
        provided.
    spot_radius : int, float, Tuple(int, float), List(int, float) or None
        Radius of the spot, in nanometer. One value per spatial dimension (zyx
        or yx dimensions). If it's a scalar, the same radius is applied to
        every dimensions. Not used if 'log_kernel_size' and 'minimum_distance'
        are provided.
    log_kernel_size : int, float, Tuple(int, float), List(int, float) or None
        Size of the LoG kernel. It equals the standard deviation (in pixels)
        used for the gaussian kernel (one for each dimension). One value per
        spatial dimension (zyx or yx dimensions). If it's a scalar, the same
        standard deviation is applied to every dimensions. If None, we estimate
        it with the voxel size and spot radius.
    minimum_distance : int, float, Tuple(int, float), List(int, float) or None
        Minimum distance (in pixels) between two spots we want to be able to
        detect separately. One value per spatial dimension (zyx or yx
        dimensions). If it's a scalar, the same distance is applied to every
        dimensions. If None, we estimate it with the voxel size and spot
        radius.
    title : str or None
        Title of the plot.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
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

    """
    # check parameters
    stack.check_parameter(
        title=(str, list, type(None)),
        framesize=tuple,
        size_title=int,
        size_axes=int,
        size_legend=int,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool)

    # get candidate thresholds and spots count to plot the elbow curve
    thresholds, count_spots, threshold = detection.get_elbow_values(
        images,
        voxel_size=voxel_size,
        spot_radius=spot_radius,
        log_kernel_size=log_kernel_size,
        minimum_distance=minimum_distance)

    # plot
    plt.figure(figsize=framesize)
    plt.plot(thresholds, count_spots, c="#2c7bb6", lw=2)
    if threshold is not None:
        i_threshold = np.argmax(thresholds == threshold)
        plt.scatter(
            threshold,
            count_spots[i_threshold],
            marker="D",
            c="#d7191c",
            s=60,
            label="Selected threshold")

    # axes
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=size_title)
    plt.xlabel("Thresholds", fontweight="bold", fontsize=size_axes)
    plt.ylabel(
        "Number of mRNAs detected (log scale)",
        fontweight="bold",
        fontsize=size_axes)
    if threshold is not None:
        plt.legend(prop={'size': size_legend})
    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()


def plot_elbow_colocalized(spots_1, spots_2, voxel_size, threshold_max=None,
                           title=None, framesize=(5, 5), size_title=20,
                           size_axes=15, size_legend=15, path_output=None,
                           ext="png", show=True):
    """Plot the elbow curve that allows an automated colocalized spot
    detection.

    Parameters
    ----------
    spots_1 : np.ndarray
        Coordinates of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    spots_2 : np.ndarray
        Coordinates of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    voxel_size : int, float, Tuple(int, float), or List(int, float)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    threshold_max : int, float or None
        Maximum threshold value to consider.
    title : str or None
        Title of the plot.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
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

    """
    # check parameters
    stack.check_parameter(
        threshold_max=(int, float, type(None)),
        title=(str, list, type(None)),
        framesize=tuple,
        size_title=int,
        size_axes=int,
        size_legend=int,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool)

    # get thresholds and colocalized spots count to plot the elbow curve
    (thresholds, count_colocalized,
     threshold) = multistack.get_elbow_value_colocalized(
        spots_1=spots_1,
        spots_2=spots_2,
        voxel_size=voxel_size)

    # plot
    plt.figure(figsize=framesize)
    plt.hlines(
        len(spots_1),
        thresholds[0],
        thresholds[-1],
        colors="forestgreen",
        linestyles="--",
        label="Spots 1")
    plt.hlines(
        len(spots_2),
        thresholds[0],
        thresholds[-1],
        colors="steelblue",
        linestyles="--",
        label="Spots 2")
    plt.plot(
        thresholds,
        count_colocalized,
        color="firebrick",
        label="Colocalized spots")
    if threshold is not None:
        i_threshold = np.argmax(thresholds == threshold)
        plt.scatter(
            threshold,
            count_colocalized[i_threshold],
            marker="D",
            c="#d7191c",
            s=60,
            label="Selected threshold")

    #  define xlim
    if threshold_max is not None:
        plt.xlim((0, threshold_max))

    # axes
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=size_title)
    plt.xlabel("Thresholds", fontweight="bold", fontsize=size_axes)
    plt.ylabel(
        "Number of mRNAs detected",
        fontweight="bold",
        fontsize=size_axes)
    if threshold is not None:
        plt.legend(prop={'size': size_legend})
    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()
