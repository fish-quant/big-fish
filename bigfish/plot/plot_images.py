# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to plot 2-d pixel and coordinates images.
"""

import warnings

import bigfish.stack as stack
import bigfish.multistack as multistack

from .utils import save_plot, get_minmax_values, create_colormap

import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import find_boundaries
from matplotlib.colors import ListedColormap
from matplotlib.patches import RegularPolygon


# ### General plot ###

def plot_yx(image, r=0, c=0, z=0, rescale=False, contrast=False,
            title=None, framesize=(8, 8), remove_frame=True, path_output=None,
            ext="png", show=True):
    """Plot the selected yx plan of the selected dimensions of an image.

    Parameters
    ----------
    image : np.ndarray
        A 2-d, 3-d, 4-d or 5-d image with shape (y, x), (z, y, x),
        (c, z, y, x) or (r, c, z, y, x) respectively.
    r : int
        Index of the round to keep.
    c : int
        Index of the channel to keep.
    z : int
        Index of the z slice to keep.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3, 4, 5],
        dtype=[np.uint8, np.uint16, np.int64, np.float32, np.float64, bool])
    stack.check_parameter(
        r=int, c=int, z=int,
        rescale=bool,
        contrast=bool,
        title=(str, type(None)),
        framesize=tuple,
        remove_frame=bool,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool)

    # get the 2-d image
    if image.ndim == 2:
        xy_image = image
    elif image.ndim == 3:
        xy_image = image[z, :, :]
    elif image.ndim == 4:
        xy_image = image[c, z, :, :]
    else:
        xy_image = image[r, c, z, :, :]

    # plot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        plt.imshow(xy_image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        plt.imshow(xy_image)
    else:
        if xy_image.dtype not in [np.int64, bool]:
            xy_image = stack.rescale(xy_image, channel_to_stretch=0)
        plt.imshow(xy_image)
    if title is not None and not remove_frame:
        plt.title(title, fontweight="bold", fontsize=25)
    if not remove_frame:
        plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()


def plot_images(images, rescale=False, contrast=False, titles=None,
                framesize=(15, 10), remove_frame=True, path_output=None,
                ext="png", show=True):
    """Plot or subplot of 2-d images.

    Parameters
    ----------
    images : np.ndarray or List[np.ndarray]
        Images with shape (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    titles : List[str]
        Titles of the subplots.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.


    """
    # enlist image if necessary
    if isinstance(images, np.ndarray):
        images = [images]

    # check parameters
    stack.check_parameter(
        images=list,
        rescale=bool,
        contrast=bool,
        titles=(str, list, type(None)),
        framesize=tuple,
        remove_frame=bool,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool)
    for image in images:
        stack.check_array(
            image,
            ndim=2,
            dtype=[np.uint8, np.uint16, np.int64,
                   np.float32, np.float64,
                   bool])

    # we plot 3 images by row maximum
    nrow = int(np.ceil(len(images)/3))
    ncol = min(len(images), 3)

    # plot one image
    if len(images) == 1:
        if titles is not None:
            title = titles[0]
        else:
            title = None
        plot_yx(
            images[0],
            rescale=rescale,
            contrast=contrast,
            title=title,
            framesize=framesize,
            remove_frame=remove_frame,
            path_output=path_output,
            ext=ext,
            show=show)

        return

    # plot multiple images
    fig, ax = plt.subplots(nrow, ncol, figsize=framesize)

    # one row
    if len(images) in [2, 3]:
        for i, image in enumerate(images):
            if remove_frame:
                ax[i].axis("off")
            if not rescale and not contrast:
                vmin, vmax = get_minmax_values(image)
                ax[i].imshow(image, vmin=vmin, vmax=vmax)
            elif rescale and not contrast:
                ax[i].imshow(image)
            else:
                if image.dtype not in [np.int64, bool]:
                    image = stack.rescale(image, channel_to_stretch=0)
                ax[i].imshow(image)
            if titles is not None:
                ax[i].set_title(titles[i], fontweight="bold", fontsize=10)

    # several rows
    else:
        # we complete the row with empty frames
        r = nrow * 3 - len(images)
        images_completed = [image for image in images] + [None] * r

        for i, image in enumerate(images_completed):
            row = i // 3
            col = i % 3
            if image is None:
                ax[row, col].set_visible(False)
                continue
            if remove_frame:
                ax[row, col].axis("off")
            if not rescale and not contrast:
                vmin, vmax = get_minmax_values(image)
                ax[row, col].imshow(image, vmin=vmin, vmax=vmax)
            elif rescale and not contrast:
                ax[row, col].imshow(image)
            else:
                if image.dtype not in [np.int64, bool]:
                    image = stack.rescale(image, channel_to_stretch=0)
                ax[row, col].imshow(image)
            if titles is not None:
                ax[row, col].set_title(
                    titles[i],
                    fontweight="bold",
                    fontsize=10)

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()


# ### Segmentation plot ###

def plot_segmentation(image, mask, rescale=False, contrast=False, title=None,
                      framesize=(15, 10), remove_frame=True,
                      path_output=None, ext="png", show=True):
    """Plot result of a 2-d segmentation, with labelled instances if available.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with shape (y, x).
    mask : np.ndarray
        A 2-d image with shape (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, np.float32, np.float64, bool])
    stack.check_array(
        mask,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(
        rescale=bool,
        contrast=bool,
        title=(str, type(None)),
        framesize=tuple,
        remove_frame=bool,
        path_output=(str, type(None)),
        ext=(str, list))

    # plot
    fig, ax = plt.subplots(1, 3, sharex='col', figsize=framesize)

    # image
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        ax[0].imshow(image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        ax[0].imshow(image)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        ax[0].imshow(image)
    if title is not None:
        ax[0].set_title(title, fontweight="bold", fontsize=10)
    if remove_frame:
        ax[0].axis("off")

    # label
    ax[1].imshow(mask)
    if title is not None:
        ax[1].set_title("Segmentation", fontweight="bold", fontsize=10)
    if remove_frame:
        ax[1].axis("off")

    # superposition
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        ax[2].imshow(image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        ax[2].imshow(image)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        ax[2].imshow(image)
    masked = np.ma.masked_where(mask == 0, mask)
    ax[2].imshow(masked, cmap=ListedColormap(['red']), alpha=0.5)
    if title is not None:
        ax[2].set_title("Surface", fontweight="bold", fontsize=10)
    if remove_frame:
        ax[2].axis("off")

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()


def plot_segmentation_boundary(image, cell_label=None, nuc_label=None,
                               rescale=False, contrast=False, title=None,
                               framesize=(10, 10), remove_frame=True,
                               path_output=None, ext="png", show=True):
    """Plot the boundary of the segmented objects.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with shape (y, x).
    cell_label : np.ndarray
        A 2-d image with shape (y, x).
    nuc_label : np.ndarray
        A 2-d image with shape (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, np.float32, np.float64, bool])
    if cell_label is not None:
        stack.check_array(
            cell_label,
            ndim=2,
            dtype=[np.uint8, np.uint16, np.int64, bool])
    if nuc_label is not None:
        stack.check_array(
            nuc_label,
            ndim=2,
            dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(
        rescale=bool,
        contrast=bool,
        title=(str, type(None)),
        framesize=tuple,
        remove_frame=bool,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool)

    # get boundaries
    cell_boundaries = None
    nuc_boundaries = None
    if cell_label is not None:
        cell_boundaries = find_boundaries(cell_label, mode='thick')
        cell_boundaries = np.ma.masked_where(
            cell_boundaries == 0,
            cell_boundaries)
    if nuc_label is not None:
        nuc_boundaries = find_boundaries(nuc_label, mode='thick')
        nuc_boundaries = np.ma.masked_where(
            nuc_boundaries == 0,
            nuc_boundaries)

    # plot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        plt.imshow(image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        plt.imshow(image)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        plt.imshow(image)
    if cell_label is not None:
        plt.imshow(cell_boundaries, cmap=ListedColormap(['red']))
    if nuc_label is not None:
        plt.imshow(nuc_boundaries, cmap=ListedColormap(['blue']))
    if title is not None and not remove_frame:
        plt.title(title, fontweight="bold", fontsize=25)
    if not remove_frame:
        plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()


def plot_segmentation_diff(image, mask_pred, mask_gt, rescale=False,
                           contrast=False, title=None, framesize=(15, 10),
                           remove_frame=True, path_output=None, ext="png",
                           show=True):
    """Plot segmentation results along with ground truth to compare.

    Parameters
    ----------
    image : np.ndarray, np.uint, np.int, np.float or bool
        Image with shape (y, x).
    mask_pred : np.ndarray, np.uint, np.int or np.float
        Image with shape (y, x).
    mask_gt : np.ndarray, np.uint, np.int or np.float
        Image with shape (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    title : str or None
        Title of the plot.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
    remove_frame : bool
        Remove axes and frame.
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
        rescale=bool,
        contrast=bool,
        title=(str, type(None)),
        framesize=tuple,
        remove_frame=bool,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool)
    stack.check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, np.float32, np.float64, bool])
    stack.check_array(
        mask_pred,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int32, np.int64,
               np.float32, np.float64,
               bool])
    stack.check_array(
        mask_gt,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int32, np.int64,
               np.float32, np.float64,
               bool])

    # plot multiple images
    fig, ax = plt.subplots(1, 3, figsize=framesize)

    # image
    if remove_frame:
        ax[0].axis("off")
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        ax[0].imshow(image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        ax[0].imshow(image)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        ax[0].imshow(image)
    if title is None:
        ax[0].set_title("", fontweight="bold", fontsize=10)
    else:
        ax[0].set_title(title, fontweight="bold", fontsize=10)

    # build colormap
    cmap = create_colormap()

    # prediction
    im_mask_pred = np.ma.masked_where(mask_pred == 0, mask_pred)
    if remove_frame:
        ax[1].axis("off")
    ax[1].imshow(im_mask_pred, cmap=cmap)
    ax[1].set_title("Prediction", fontweight="bold", fontsize=10)

    # ground truth
    im_mask_gt = np.ma.masked_where(mask_gt == 0, mask_gt)
    if remove_frame:
        ax[2].axis("off")
    ax[2].imshow(im_mask_gt, cmap=cmap)
    ax[2].set_title("Ground truth", fontweight="bold", fontsize=10)

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()


# ### Detection plot ###

def plot_detection(image, spots, shape="circle", radius=3, color="red",
                   linewidth=1, fill=False, rescale=False, contrast=False,
                   title=None, framesize=(15, 10), remove_frame=True,
                   path_output=None, ext="png", show=True):
    """Plot detected spots and foci on a 2-d image.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with shape (y, x).
    spots : List[np.ndarray] or np.ndarray
        Array with coordinates and shape (nb_spots, 3) or (nb_spots, 2). To
        plot different kind of detected spots with different symbols, use a
        list of arrays.
    shape : List[str] or str
        List of symbols used to localized the detected spots in the image,
        among `circle`, `square` or `polygon`. One symbol per array in `spots`.
        If `shape` is a string, the same symbol is used for every elements of
        'spots'.
    radius : List[int or float], int or float
        List of yx radii of the detected spots, in pixel. One radius per array
        in `spots`. If `radius` is a scalar, the same value is applied for
        every elements of `spots`.
    color : List[str] or str
        List of colors of the detected spots. One color per array in `spots`.
        If `color` is a string, the same color is applied for every elements
        of `spots`.
    linewidth : List[int] or int
        List of widths or width of the border symbol. One integer per array
        in `spots`. If `linewidth` is an integer, the same width is applied
        for every elements of `spots`.
    fill : List[bool] or bool
        List of boolean to fill the symbol the detected spots.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize``.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=2,
        dtype=[np.uint8, np.uint16, np.int64, np.float32, np.float64])
    stack.check_parameter(
        spots=(list, np.ndarray),
        shape=(list, str),
        radius=(list, int, float),
        color=(list, str),
        linewidth=(list, int),
        fill=(list, bool),
        rescale=bool,
        contrast=bool,
        title=(str, type(None)),
        framesize=tuple,
        remove_frame=bool,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool)
    if isinstance(spots, list):
        for spots_ in spots:
            stack.check_array(spots_, ndim=2, dtype=[np.int64, np.float64])
    else:
        stack.check_array(spots, ndim=2, dtype=[np.int64, np.float64])

    # enlist and format parameters
    if not isinstance(spots, list):
        spots = [spots]
    n = len(spots)
    if not isinstance(shape, list):
        shape = [shape] * n
    elif isinstance(shape, list) and len(shape) != n:
        raise ValueError("If 'shape' is a list, it should have the same "
                         "number of items than spots ({0}).".format(n))
    if not isinstance(radius, list):
        radius = [radius] * n
    elif isinstance(radius, list) and len(radius) != n:
        raise ValueError("If 'radius' is a list, it should have the same "
                         "number of items than spots ({0}).".format(n))
    if not isinstance(color, list):
        color = [color] * n
    elif isinstance(color, list) and len(color) != n:
        raise ValueError("If 'color' is a list, it should have the same "
                         "number of items than spots ({0}).".format(n))
    if not isinstance(linewidth, list):
        linewidth = [linewidth] * n
    elif isinstance(linewidth, list) and len(linewidth) != n:
        raise ValueError("If 'linewidth' is a list, it should have the same "
                         "number of items than spots ({0}).".format(n))
    if not isinstance(fill, list):
        fill = [fill] * n
    elif isinstance(fill, list) and len(fill) != n:
        raise ValueError("If 'fill' is a list, it should have the same "
                         "number of items than spots ({0}).".format(n))

    # plot
    fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)

    # image
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        ax[0].imshow(image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        ax[0].imshow(image)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        ax[0].imshow(image)

    # spots
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        ax[1].imshow(image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        ax[1].imshow(image)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        ax[1].imshow(image)

    for i, coordinates in enumerate(spots):

        # get 2-d coordinates
        if coordinates.shape[1] == 3:
            coordinates_2d = coordinates[:, 1:]
        else:
            coordinates_2d = coordinates

        # plot symbols
        for y, x in coordinates_2d:
            x = _define_patch(x, y, shape[i], radius[i], color[i],
                              linewidth[i], fill[i])
            ax[1].add_patch(x)

    # titles and frames
    if title is not None:
        ax[0].set_title(title, fontweight="bold", fontsize=10)
        ax[1].set_title("Detection results", fontweight="bold", fontsize=10)
    if remove_frame:
        ax[0].axis("off")
        ax[1].axis("off")
    plt.tight_layout()

    # output
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()


def _define_patch(x, y, shape, radius, color, linewidth, fill):
    """Define a matplotlib.patches to plot.

    Parameters
    ----------
    x : int or float
        Coordinate x for the patch center.
    y : int or float
        Coordinate y for the patch center.
    shape : str
        Shape of the patch to define (among `circle`, `square` or `polygon`)
    radius : int or float
        Radius of the patch.
    color : str
        Color of the patch.
    linewidth : int
        Width of the patch border.
    fill : bool
        Make the patch shape empty or not.

    Returns
    -------
    x : matplotlib.patches object
        Geometric form to add to a plot.

    """
    # circle
    if shape == "circle":
        x = plt.Circle(
            (x, y),
            radius,
            color=color,
            linewidth=linewidth,
            fill=fill)
    # square
    elif shape == "square":
        x = plt.Rectangle(
            (x, y),
            radius,
            radius,
            color=color,
            linewidth=linewidth,
            fill=fill)
    # polygon
    elif shape == "polygon":
        x = RegularPolygon(
            (x, y),
            5,
            radius,
            color=color,
            linewidth=linewidth,
            fill=fill)
    else:
        warnings.warn("shape should take a value among 'circle', 'square' or "
                      "'polygon', but not {0}".format(shape), UserWarning)

    return x


def plot_reference_spot(reference_spot, rescale=False, contrast=False,
                        title=None, framesize=(8, 8), remove_frame=True,
                        path_output=None, ext="png", show=True):
    """Plot the selected yx plan of the selected dimensions of an image.

    Parameters
    ----------
    reference_spot : np.ndarray
        Spot image with shape (z, y, x) or (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    """
    # check parameters
    stack.check_array(
        reference_spot,
        ndim=[2,  3],
        dtype=[np.uint8, np.uint16, np.int64, np.float32, np.float64])
    stack.check_parameter(
        rescale=bool,
        contrast=bool,
        title=(str, type(None)),
        framesize=tuple,
        remove_frame=bool,
        path_output=(str, type(None)),
        ext=(str, list),
        show=bool)

    # project spot in 2-d if necessary
    if reference_spot.ndim == 3:
        reference_spot = stack.maximum_projection(reference_spot)

    # plot reference spot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(reference_spot)
        plt.imshow(reference_spot, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        plt.imshow(reference_spot)
    else:
        if reference_spot.dtype not in [np.int64, bool]:
            reference_spot = stack.rescale(
                reference_spot,
                channel_to_stretch=0)
        plt.imshow(reference_spot)
    if title is not None and not remove_frame:
        plt.title(title, fontweight="bold", fontsize=25)
    if not remove_frame:
        plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()


# ### Individual cell plot ###

def plot_cell(ndim, cell_coord=None, nuc_coord=None, rna_coord=None,
              foci_coord=None, other_coord=None, image=None, cell_mask=None,
              nuc_mask=None, title=None, remove_frame=True, rescale=False,
              contrast=False, framesize=(15, 10), path_output=None, ext="png",
              show=True):
    """
    Plot image and coordinates extracted for a specific cell.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions to consider in the coordinates (2 or 3).
    cell_coord : np.ndarray, np.int64
        Coordinates of the cell border with shape (nb_points, 2). If None,
        coordinate representation of the cell is not shown.
    nuc_coord : np.ndarray, np.int64
        Coordinates of the nucleus border with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Coordinates of the detected spots with shape (nb_spots, 4) or
        (nb_spots, 3). One coordinate per dimension (zyx or yx dimensions)
        plus the index of the cluster assigned to the spot. If no cluster was
        assigned, value is -1. If only coordinates of spatial dimensions are
        available, only centroid of foci can be shown.
    foci_coord : np.ndarray, np.int64
        Array with shape (nb_foci, 5) or (nb_foci, 4). One coordinate per
        dimension for the foci centroid (zyx or yx dimensions), the number of
        spots detected in the foci and its index.
    other_coord : np.ndarray, np.int64
        Coordinates of the detected elements with shape (nb_elements, 3) or
        (nb_elements, 2). One coordinate per dimension (zyx or yx dimensions).
    image : np.ndarray, np.uint
        Original image of the cell with shape (y, x). If None, original image
        of the cell is not shown.
    cell_mask : np.ndarray, np.uint
        Mask of the cell.
    nuc_mask : np.ndarray, np.uint
        Mask of the nucleus.
    title : str or None
        Title of the image.
    remove_frame : bool
        Remove axes and frame.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    framesize : tuple
        Size of the frame used to plot with ``plt.figure(figsize=framesize)``.
    path_output : str or None
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    """
    if cell_coord is None and image is None:
        return

    # check parameters
    if cell_coord is not None:
        stack.check_array(cell_coord, ndim=2, dtype=np.int64)
    if nuc_coord is not None:
        stack.check_array(nuc_coord, ndim=2, dtype=np.int64)
    if rna_coord is not None:
        stack.check_array(rna_coord, ndim=2, dtype=np.int64)
    if foci_coord is not None:
        stack.check_array(foci_coord, ndim=2, dtype=np.int64)
    if other_coord is not None:
        stack.check_array(other_coord, ndim=2, dtype=np.int64)
    if image is not None:
        stack.check_array(
            image,
            ndim=2,
            dtype=[np.uint8, np.uint16, np.int64, np.float32, np.float64])
    if cell_mask is not None:
        stack.check_array(
            cell_mask,
            ndim=2,
            dtype=[np.uint8, np.uint16, np.int64, bool])
    if nuc_mask is not None:
        stack.check_array(
            nuc_mask,
            ndim=2,
            dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(
        ndim=int,
        title=(str, type(None)),
        remove_frame=bool,
        rescale=bool,
        contrast=bool,
        framesize=tuple,
        path_output=(str, type(None)),
        ext=(str, list))

    # plot original image and coordinate representation
    if cell_coord is not None and image is not None:
        fig, ax = plt.subplots(1, 2, figsize=framesize)

        # original image
        if not rescale and not contrast:
            vmin, vmax = get_minmax_values(image)
            ax[0].imshow(image, vmin=vmin, vmax=vmax)
        elif rescale and not contrast:
            ax[0].imshow(image)
        else:
            if image.dtype not in [np.int64, bool]:
                image = stack.rescale(image, channel_to_stretch=0)
            ax[0].imshow(image)
        if cell_mask is not None:
            cell_boundaries = multistack.from_surface_to_boundaries(
                cell_mask)
            cell_boundaries = np.ma.masked_where(
                cell_boundaries == 0,
                cell_boundaries)
            ax[0].imshow(cell_boundaries, cmap=ListedColormap(['red']))
        if nuc_mask is not None:
            nuc_boundaries = multistack.from_surface_to_boundaries(nuc_mask)
            nuc_boundaries = np.ma.masked_where(
                nuc_boundaries == 0,
                nuc_boundaries)
            ax[0].imshow(nuc_boundaries, cmap=ListedColormap(['blue']))

        # coordinate image
        ax[1].plot(cell_coord[:, 1], cell_coord[:, 0], c="black", linewidth=2)
        if nuc_coord is not None:
            ax[1].plot(
                nuc_coord[:, 1],
                nuc_coord[:, 0],
                c="steelblue",
                linewidth=2)
        if rna_coord is not None:
            ax[1].scatter(
                rna_coord[:, ndim - 1],
                rna_coord[:, ndim - 2],
                s=25,
                c="firebrick",
                marker=".")
        if foci_coord is not None:
            for foci in foci_coord:
                ax[1].text(
                    foci[ndim-1] + 5,
                    foci[ndim-2] - 5,
                    str(foci[ndim]),
                    color="darkorange",
                    size=20)
            # case where we know which rna belong to a foci
            if rna_coord.shape[1] == ndim + 1:
                foci_indices = foci_coord[:, ndim + 1]
                mask_rna_in_foci = np.isin(rna_coord[:, ndim], foci_indices)
                rna_in_foci_coord = rna_coord[mask_rna_in_foci, :].copy()
                ax[1].scatter(
                    rna_in_foci_coord[:, ndim - 1],
                    rna_in_foci_coord[:, ndim - 2],
                    s=25,
                    c="darkorange",
                    marker=".")
            # case where we only know the foci centroid
            else:
                ax[1].scatter(
                    foci_coord[:, ndim - 1],
                    foci_coord[:, ndim - 2],
                    s=40,
                    c="darkorange",
                    marker="o")
        if other_coord is not None:
            ax[1].scatter(
                other_coord[:, ndim - 1],
                other_coord[:, ndim - 2],
                s=25,
                c="forestgreen",
                marker="D")

        # titles and frames
        _, _, min_y, max_y = ax[1].axis()
        ax[1].set_ylim(max_y, min_y)
        ax[1].use_sticky_edges = True
        ax[1].margins(0.01, 0.01)
        ax[1].axis('scaled')
        if remove_frame:
            ax[0].axis("off")
            ax[1].axis("off")
        if title is not None:
            ax[0].set_title(
                "Original image ({0})".format(title),
                fontweight="bold",
                fontsize=10)
            ax[1].set_title(
                "Coordinate representation ({0})".format(title),
                fontweight="bold",
                fontsize=10)
        plt.tight_layout()

        # output
        if path_output is not None:
            save_plot(path_output, ext)
        if show:
            plt.show()
        else:
            plt.close()

    # plot coordinate representation only
    elif cell_coord is not None and image is None:
        if remove_frame:
            fig = plt.figure(figsize=framesize, frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
        else:
            plt.figure(figsize=framesize)

        # coordinate image
        plt.plot(cell_coord[:, 1], cell_coord[:, 0], c="black", linewidth=2)
        if nuc_coord is not None:
            plt.plot(
                nuc_coord[:, 1],
                nuc_coord[:, 0],
                c="steelblue",
                linewidth=2)
        if rna_coord is not None:
            plt.scatter(
                rna_coord[:, ndim - 1],
                rna_coord[:, ndim - 2],
                s=25,
                c="firebrick",
                marker=".")
        if foci_coord is not None:
            for foci in foci_coord:
                plt.text(
                    foci[ndim-1] + 5,
                    foci[ndim-2] - 5,
                    str(foci[ndim]),
                    color="darkorange",
                    size=20)
            # case where we know which rna belong to a foci
            if rna_coord.shape[1] == ndim + 1:
                foci_indices = foci_coord[:, ndim + 1]
                mask_rna_in_foci = np.isin(rna_coord[:, ndim], foci_indices)
                rna_in_foci_coord = rna_coord[mask_rna_in_foci, :].copy()
                plt.scatter(
                    rna_in_foci_coord[:, ndim - 1],
                    rna_in_foci_coord[:, ndim - 2],
                    s=25,
                    c="darkorange",
                    marker=".")
            # case where we only know the foci centroid
            else:
                plt.scatter(
                    foci_coord[:, ndim - 1],
                    foci_coord[:, ndim - 2],
                    s=40,
                    c="darkorange",
                    marker="o")
        if other_coord is not None:
            plt.scatter(
                other_coord[:, ndim - 1],
                other_coord[:, ndim - 2],
                s=25,
                c="forestgreen",
                marker="D")

        # titles and frames
        _, _, min_y, max_y = plt.axis()
        plt.ylim(max_y, min_y)
        plt.use_sticky_edges = True
        plt.margins(0.01, 0.01)
        plt.axis('scaled')
        if title is not None:
            plt.title(
                "Coordinate representation ({0})".format(title),
                fontweight="bold",
                fontsize=10)
        if not remove_frame:
            plt.tight_layout()

        # output
        if path_output is not None:
            save_plot(path_output, ext)
        if show:
            plt.show()
        else:
            plt.close()

    # plot original image only
    elif cell_coord is None and image is not None:
        plot_segmentation_boundary(
            image=image,
            cell_label=cell_mask,
            nuc_label=nuc_mask,
            rescale=rescale,
            contrast=contrast,
            title=title,
            framesize=framesize,
            remove_frame=remove_frame,
            path_output=path_output,
            ext=ext,
            show=show)
