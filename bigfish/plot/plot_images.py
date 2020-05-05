# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to plot 2-d pixel and coordinates images.
"""

import warnings

import bigfish.stack as stack

import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot, get_minmax_values

from matplotlib.colors import ListedColormap
from matplotlib.patches import RegularPolygon


# ### General plot ###

def plot_yx(tensor, r=0, c=0, z=0, rescale=False, title=None,
            framesize=(8, 8), remove_frame=False, path_output=None,
            ext="png", show=True):
    """Plot the selected yx plan of the selected dimensions of an image.

    Parameters
    ----------
    tensor : np.ndarray
        A 2-d, 3-d, 4-d or 5-d tensor with shape (y, x), (z, y, x),
        (c, z, y, x) or (r, c, z, y, x) respectively.
    r : int
        Index of the round to keep.
    c : int
        Index of the channel to keep.
    z : int
        Index of the z slice to keep.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
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
    # check parameters
    stack.check_array(tensor,
                      ndim=[2, 3, 4, 5],
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64,
                             bool])
    stack.check_parameter(r=int, c=int, z=int,
                          rescale=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list))

    # get the 2-d tensor
    if tensor.ndim == 2:
        xy_tensor = tensor
    elif tensor.ndim == 3:
        xy_tensor = tensor[z, :, :]
    elif tensor.ndim == 4:
        xy_tensor = tensor[c, z, :, :]
    else:
        xy_tensor = tensor[r, c, z, :, :]

    # get minimum and maximum value of the image
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(tensor)

    # plot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
    if not rescale:
        plt.imshow(xy_tensor, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(xy_tensor)
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

    return


def plot_images(images, rescale=False, titles=None, framesize=(15, 5),
                remove_frame=False, path_output=None, ext="png", show=True):
    """Plot or subplot of 2-d images.

    Parameters
    ----------
    images : np.ndarray or List[np.ndarray]
        Images with shape (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    titles : List[str]
        Titles of the subplots.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
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
    if isinstance(images, np.ndarray):
        images = [images]

    # check parameters
    stack.check_parameter(images=list,
                          rescale=bool,
                          titles=(str, list, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)
    for image in images:
        stack.check_array(image,
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
        plot_yx(images[0],
                rescale=rescale,
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
        for i, tensor in enumerate(images):
            if remove_frame:
                ax[i].axis("off")
            if not rescale:
                vmin, vmax = get_minmax_values(tensor)
                ax[i].imshow(tensor, vmin=vmin, vmax=vmax)
            else:
                ax[i].imshow(tensor)
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
            if not rescale:
                vmin, vmax = get_minmax_values(image)
                ax[row, col].imshow(image, vmin=vmin, vmax=vmax)
            else:
                ax[row, col].imshow(image)
            if titles is not None:
                ax[row, col].set_title(titles[i],
                                       fontweight="bold", fontsize=10)

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


# ### Segmentation plot ###

def plot_segmentation(image, mask, rescale=False, title=None,
                      framesize=(15, 5), remove_frame=False,
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
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
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
    # check parameters
    stack.check_array(image,
                      ndim=2,
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64,
                             bool])
    stack.check_array(mask,
                      ndim=2,
                      dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(rescale=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list))

    # get minimum and maximum value of the image
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(image)

    # plot
    fig, ax = plt.subplots(1, 3, sharex='col', figsize=framesize)

    # image
    if not rescale:
        ax[0].imshow(image, vmin=vmin, vmax=vmax)
    else:
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
    if not rescale:
        ax[2].imshow(image, vmin=vmin, vmax=vmax)
    else:
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

    return


def plot_segmentation_boundary(image, cell_mask=None, nuc_mask=None,
                               rescale=False, title=None, framesize=(10, 10),
                               remove_frame=False, path_output=None,
                               ext="png", show=True):
    """Plot the boundary of the segmented objects.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with shape (y, x).
    cell_mask : np.ndarray
        A 2-d image with shape (y, x).
    nuc_mask : np.ndarray
        A 2-d image with shape (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
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
    # check parameters
    stack.check_array(image,
                      ndim=2,
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64,
                             bool])
    if cell_mask is not None:
        stack.check_array(cell_mask,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    if nuc_mask is not None:
        stack.check_array(nuc_mask,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(rescale=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)

    # get minimum and maximum value of the image
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(image)

    # get boundaries
    cell_boundaries = None
    nuc_boundaries = None
    if cell_mask is not None:
        cell_boundaries = stack.from_surface_to_boundaries(cell_mask)
        cell_boundaries = np.ma.masked_where(cell_boundaries == 0,
                                             cell_boundaries)
    if nuc_mask is not None:
        nuc_boundaries = stack.from_surface_to_boundaries(nuc_mask)
        nuc_boundaries = np.ma.masked_where(nuc_boundaries == 0,
                                            nuc_boundaries)

    # plot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
    if not rescale:
        plt.imshow(image, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(image)
    if cell_mask is not None:
        plt.imshow(cell_boundaries, cmap=ListedColormap(['red']))
    if nuc_mask is not None:
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

    return


# ### Detection plot ###

def plot_spot_detection(image, spots, radius_yx, rescale=False,
                        title=None, framesize=(15, 5), remove_frame=False,
                        path_output=None, ext="png", show=True):
    """Plot detected spot on a 2-d image.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with shape (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    radius_yx : float or int
        Radius yx of the detected spots.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot (plt.figure(figsize=framesize).
    remove_frame : bool
        Remove axes and frame.
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
    # check parameters
    stack.check_array(image,
                      ndim=2,
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64])
    stack.check_array(spots,
                      ndim=2,
                      dtype=np.int64)
    stack.check_parameter(radius_yx=(float, int),
                          rescale=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)

    # get minimum and maximum value of the image
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(image)

    # plot
    fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)

    # image
    if not rescale:
        ax[0].imshow(image, vmin=vmin, vmax=vmax)
    else:
        ax[0].imshow(image)
    if title is not None:
        ax[0].set_title(title, fontweight="bold", fontsize=10)
    if remove_frame:
        ax[0].axis("off")

    # spots
    if not rescale:
        ax[1].imshow(image, vmin=vmin, vmax=vmax)
    else:
        ax[1].imshow(image)
    if spots.shape[1] == 3:
        spots_2d = spots[:, 1:]
    else:
        spots_2d = spots
    for y, x in spots_2d:
        c = plt.Circle((x, y), radius_yx,
                       color="red",
                       linewidth=1,
                       fill=False)
        ax[1].add_patch(c)
    if title is not None:
        ax[1].set_title("All detected spots", fontweight="bold", fontsize=10)
    if remove_frame:
        ax[1].axis("off")

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_reference_spot(reference_spot, rescale=False, title=None,
                        framesize=(8, 8), remove_frame=False,
                        path_output=None, ext="png", show=True):
    """Plot the selected yx plan of the selected dimensions of an image.

    Parameters
    ----------
    reference_spot : np.ndarray
        Spot image with shape (z, y, x) or (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
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
    # check parameters
    stack.check_array(reference_spot,
                      ndim=[2,  3],
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64])
    stack.check_parameter(rescale=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)

    # project spot in 2-d if necessary
    if reference_spot.ndim == 3:
        reference_spot = stack.maximum_projection(reference_spot)

    # get minimum and maximum value of the image
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(reference_spot)

    # plot reference spot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
    if not rescale:
        plt.imshow(reference_spot, vmin=vmin, vmax=vmax)
    else:
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

    return


def plot_foci_detection(image, spots, foci, radius_spots_yx,
                        rescale=False, title=None, framesize=(15, 10),
                        remove_frame=False, path_output=None, ext="png",
                        show=True):
    """Plot detected spots and foci on a 2-d image.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with shape (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or (nb_spots, 2).
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5) or (nb_foci, 4). One coordinate per
        dimension (zyx or  yx coordinates), number of RNAs in the foci and
        index of the foci.
    radius_spots_yx : float or int
        Radius yx of the detected spots.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot (plt.figure(figsize=framesize).
    remove_frame : bool
        Remove axes and frame.
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
    # check parameters
    stack.check_array(image,
                      ndim=2,
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64])
    stack.check_array(foci,
                      ndim=2,
                      dtype=np.int64)
    stack.check_parameter(spots=(np.ndarray, type(None)),
                          radius_spots_yx=(float, int),
                          rescale=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)
    if spots is not None:
        stack.check_array(spots,
                          ndim=2,
                          dtype=np.int64)

    # get minimum and maximum value of the image
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(image)

    # plot
    fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)

    # image
    if not rescale:
        ax[0].imshow(image, vmin=vmin, vmax=vmax)
    else:
        ax[0].imshow(image)
    if title is not None:
        ax[0].set_title(title, fontweight="bold", fontsize=10)
    if remove_frame:
        ax[0].axis("off")

    # spots and foci
    if not rescale:
        ax[1].imshow(image, vmin=vmin, vmax=vmax)
    else:
        ax[1].imshow(image)
    if spots is not None:
        if spots.shape[1] == 3:
            spots_2d = spots[:, 1:]
        else:
            spots_2d = spots
        for y, x in spots_2d:
            c = plt.Circle((x, y), radius_spots_yx,
                           color="red",
                           linewidth=1,
                           fill=False)
            ax[1].add_patch(c)
        title_ = "Detected spots and foci"
    else:
        title_ = "Detected foci"
    if foci.shape[1] == 5:
        foci_2d = foci[:, 1:3]
    else:
        foci_2d = foci[:, :2]
    for y, x in foci_2d:
        c = plt.Circle((x, y), radius_spots_yx * 2,
                       color="blue",
                       linewidth=2,
                       fill=False)
        ax[1].add_patch(c)
    if title is not None:
        ax[1].set_title(title_,
                        fontweight="bold",
                        fontsize=10)
    if remove_frame:
        ax[1].axis("off")

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_detection(image, spots, shape="circle", radius=3, color="red",
                   linewidth=1, fill=False, rescale=False, title=None,
                   framesize=(15, 10), remove_frame=False, path_output=None,
                   ext="png", show=True):
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
        among 'circle', 'square' or 'polygon'. One symbol per array in 'spots'.
        If 'shape' is a string, the same symbol is used for every elements of
        'spots'.
    radius : List[int or float] or int or float
        List of yx radii of the detected spots. One radius per array in
        'spots'. If 'radius' is a scalar, the same value is applied for every
        elements of 'spots'.
    color : List[str] or str
        List of colors of the detected spots. One color per array in 'spots'.
        If 'color' is a string, the same color is applied for every elements
        of 'spots'.
    linewidth : List[int] or int
        List of widths or width of the border symbol. One integer per array
        in 'spots'. If 'linewidth' is an integer, the same width is applied
        for every elements of 'spots'.
    fill : List[bool] or bool
        List of boolean to fill the symbol the detected spots.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot (plt.figure(figsize=framesize).
    remove_frame : bool
        Remove axes and frame.
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
    # check parameters
    stack.check_array(image, ndim=2, dtype=[np.uint8, np.uint16])
    stack.check_parameter(spots=(list, np.ndarray),
                          shape=(list, str),
                          radius=(list, int, float),
                          color=(list, str),
                          linewidth=(list, int),
                          fill=(list, bool),
                          rescale=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)
    if isinstance(spots, list):
        for spots_ in spots:
            stack.check_array(spots_, ndim=2, dtype=np.int64)
    else:
        stack.check_array(spots, ndim=2, dtype=np.int64)

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

    # get minimum and maximum value of the image
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(image)

    # plot
    fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)

    # image
    if not rescale:
        ax[0].imshow(image, vmin=vmin, vmax=vmax)
    else:
        ax[0].imshow(image)

    # spots
    if not rescale:
        ax[1].imshow(image, vmin=vmin, vmax=vmax)
    else:
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

    return


def _define_patch(x, y, shape, radius, color, linewidth, fill):
    """Define a matplotlib.patches to plot.

    Parameters
    ----------
    x : int or float
        Coordinate x for the patch center.
    y : int or float
        Coordinate y for the patch center.
    shape : str
        Shape of the patch to define (among 'circle', 'square' or 'polygon')
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
        x = plt.Circle((x, y), radius,
                       color=color,
                       linewidth=linewidth,
                       fill=fill)
    # square
    elif shape == "square":
        x = plt.Rectangle((x, y), radius, radius,
                          color=color,
                          linewidth=linewidth,
                          fill=fill)
    # polygon
    elif shape == "polygon":
        x = RegularPolygon((x, y), 5, radius,
                           color=color,
                           linewidth=linewidth,
                           fill=fill)
    else:
        warnings.warn("shape should take a value among 'circle', 'square' or "
                      "'polygon', but not {0}".format(shape), UserWarning)

    return x


# ### Individual cell plot ###

def plot_cell(ndim, cell_coord=None, nuc_coord=None, rna_coord=None,
              foci_coord=None, other_coord=None, image=None, cell_mask=None,
              nuc_mask=None, title=None, remove_frame=False, rescale=False,
              framesize=(15, 10), path_output=None, ext="png", show=True):
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
    title : str
        Title of the image.
    remove_frame : bool
        Remove axes and frame.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
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
        stack.check_array(image, ndim=2, dtype=[np.uint8, np.uint16])
    if cell_mask is not None:
        stack.check_array(cell_mask,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    if nuc_mask is not None:
        stack.check_array(nuc_mask,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(ndim=int,
                          title=(str, type(None)),
                          remove_frame=bool,
                          rescale=bool,
                          framesize=tuple,
                          path_output=(str, type(None)),
                          ext=(str, list))

    # plot original image and coordinate representation
    if cell_coord is not None and image is not None:
        fig, ax = plt.subplots(1, 2, figsize=framesize)

        # original image
        if not rescale:
            vmin, vmax = get_minmax_values(image)
            ax[0].imshow(image, vmin=vmin, vmax=vmax)
        else:
            ax[0].imshow(image)
        if cell_mask is not None:
            cell_boundaries = stack.from_surface_to_boundaries(cell_mask)
            cell_boundaries = np.ma.masked_where(cell_boundaries == 0,
                                                 cell_boundaries)
            ax[0].imshow(cell_boundaries, cmap=ListedColormap(['red']))
        if nuc_mask is not None:
            nuc_boundaries = stack.from_surface_to_boundaries(nuc_mask)
            nuc_boundaries = np.ma.masked_where(nuc_boundaries == 0,
                                                nuc_boundaries)
            ax[0].imshow(nuc_boundaries, cmap=ListedColormap(['blue']))

        # coordinate image
        ax[1].plot(cell_coord[:, 1], cell_coord[:, 0],
                   c="black", linewidth=2)
        if nuc_coord is not None:
            ax[1].plot(nuc_coord[:, 1], nuc_coord[:, 0],
                       c="steelblue", linewidth=2)
        if rna_coord is not None:
            ax[1].scatter(rna_coord[:, ndim - 1], rna_coord[:, ndim - 2],
                          s=25, c="firebrick", marker=".")
        if foci_coord is not None:
            for (_, y, x, nb_rna, _) in foci_coord:
                ax[1].text(x + 5, y - 5, str(nb_rna),
                           color="darkorange", size=20)
            # case where we know which rna belong to a foci
            if rna_coord.shape[1] == ndim + 1:
                foci_indices = foci_coord[:, ndim + 1]
                mask_rna_in_foci = np.isin(rna_coord[:, ndim], foci_indices)
                rna_in_foci_coord = rna_coord[mask_rna_in_foci, :].copy()
                ax[1].scatter(rna_in_foci_coord[:, ndim - 1],
                              rna_in_foci_coord[:, ndim - 2],
                              s=25, c="darkorange", marker=".")
            # case where we only know the foci centroid
            else:
                ax[1].scatter(foci_coord[:, ndim - 1], foci_coord[:, ndim - 2],
                              s=40, c="darkorange", marker="o")
        if other_coord is not None:
            ax[1].scatter(other_coord[:, ndim - 1], other_coord[:, ndim - 2],
                          s=25, c="forestgreen", marker="D")

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
            ax[0].set_title("Original image ({0})".format(title),
                            fontweight="bold", fontsize=10)
            ax[1].set_title("Coordinate representation ({0})".format(title),
                            fontweight="bold", fontsize=10)
        plt.tight_layout()

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
            plt.plot(nuc_coord[:, 1], nuc_coord[:, 0],
                     c="steelblue", linewidth=2)
        if rna_coord is not None:
            plt.scatter(rna_coord[:, ndim - 1], rna_coord[:, ndim - 2],
                        s=25, c="firebrick", marker=".")
        if foci_coord is not None:
            for (_, y, x, nb_rna, _) in foci_coord:
                plt.text(x + 5, y - 5, str(nb_rna),
                         color="darkorange", size=20)
            # case where we know which rna belong to a foci
            if rna_coord.shape[1] == ndim + 1:
                foci_indices = foci_coord[:, ndim + 1]
                mask_rna_in_foci = np.isin(rna_coord[:, ndim], foci_indices)
                rna_in_foci_coord = rna_coord[mask_rna_in_foci, :].copy()
                plt.scatter(rna_in_foci_coord[:, ndim - 1],
                            rna_in_foci_coord[:, ndim - 2],
                            s=25, c="darkorange", marker=".")
            # case where we only know the foci centroid
            else:
                plt.scatter(foci_coord[:, ndim - 1], foci_coord[:, ndim - 2],
                            s=40, c="darkorange", marker="o")
        if other_coord is not None:
            plt.scatter(other_coord[:, ndim - 1], other_coord[:, ndim - 2],
                        s=25, c="forestgreen", marker="D")

        # titles and frames
        _, _, min_y, max_y = plt.axis()
        plt.ylim(max_y, min_y)
        plt.use_sticky_edges = True
        plt.margins(0.01, 0.01)
        plt.axis('scaled')
        if title is not None:
            plt.title("Coordinate representation ({0})".format(title),
                      fontweight="bold", fontsize=10)
        if not remove_frame:
            plt.tight_layout()

    # plot original image only
    elif cell_coord is None and image is not None:
        plot_segmentation_boundary(
            image=image, cell_mask=cell_mask, nuc_mask=nuc_mask,
            rescale=rescale, title=title, framesize=framesize,
            remove_frame=remove_frame, path_output=path_output,
            ext=ext, show=show)

    # output
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_cell_image(ndim, cell_coord=None, nuc_coord=None,
                    rna_coord=None, rna_size=3, foci_coord=None, foci_size=3,
                    other_coord=None, other_size=3,
                    image=None, cell_mask=None, nuc_mask=None, title=None,
                    remove_frame=False, rescale=False, framesize=(15, 10),
                    path_output=None, ext="png", show=True):
    """
    Plot image and coordinates extracted for a specific cell.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions to consider (2 or 3).
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
    rna_size : int
        Size in pixels of the rna is calculated from the formula
        2 * rna_size - 1.
    foci_coord : np.ndarray, np.int64
        Array with shape (nb_foci, 5) or (nb_foci, 4). One coordinate per
        dimension for the foci centroid (zyx or yx dimensions), the number of
        spots detected in the foci and its index.
    foci_size : int
        Size in pixels of the foci is calculated from the formula
        2 * foci_size - 1.
    other_coord : np.ndarray, np.int64
        Coordinates of the detected elements with shape (nb_elements, 3) or
        (nb_elements, 2). One coordinate per dimension (zyx or yx dimensions).
    other_size : int
        Size in pixels of the element is calculated from the formula
        2 * other_size - 1.
    image : np.ndarray, np.uint
        Original image of the cell with shape (y, x). If None, original image
        of the cell is not shown.
    cell_mask : np.ndarray, np.uint
        Mask of the cell.
    nuc_mask : np.ndarray, np.uint
        Mask of the nucleus.
    title : str
        Title of the image.
    remove_frame : bool
        Remove axes and frame.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
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
        stack.check_array(image, ndim=2, dtype=[np.uint8, np.uint16])
    if cell_mask is not None:
        stack.check_array(cell_mask,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    if nuc_mask is not None:
        stack.check_array(nuc_mask,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(ndim=int,
                          rna_size=int,
                          foci_size=int,
                          other_size = int,
                          title=(str, type(None)),
                          remove_frame=bool,
                          rescale=bool,
                          framesize=tuple,
                          path_output=(str, type(None)),
                          ext=(str, list))

    # get shape of image built from coordinates
    image_shape, min_y, min_x, marge = stack.from_coord_to_frame(cell_coord)

    # get cell layer
    cell = np.zeros(image_shape, dtype=bool)
    cell_coord[:, 0] = cell_coord[:, 0] - min_y + marge
    cell_coord[:, 1] = cell_coord[:, 1] - min_x + marge
    cell[cell_coord[:, 0], cell_coord[:, 1]] = True

    # get nucleus layer
    nuc = np.zeros(image_shape, dtype=bool)
    if nuc_coord is not None:
        nuc_coord[:, 0] = nuc_coord[:, 0] - min_y + marge
        nuc_coord[:, 1] = nuc_coord[:, 1] - min_x + marge
        nuc[nuc_coord[:, 0], nuc_coord[:, 1]] = True

    # get rna layer
    rna = np.zeros(image_shape, dtype=bool)
    if rna_coord is not None:
        rna_coord[:, ndim - 2] = rna_coord[:, ndim - 2] - min_y + marge
        rna_coord[:, ndim - 1] = rna_coord[:, ndim - 1] - min_x + marge
        rna[rna_coord[:, ndim - 2], rna_coord[:, ndim - 1]] = True
        rna = stack.dilation_filter(rna, "square", rna_size)

    # get foci layer
    foci = np.zeros(image_shape, dtype=bool)
    if foci_coord is not None:
        foci_coord[:, ndim - 2] = foci_coord[:, ndim - 2] - min_y + marge
        foci_coord[:, ndim - 1] = foci_coord[:, ndim - 1] - min_x + marge

        # case where we know which rna belong to a foci
        if rna_coord.shape[1] == ndim + 1:
            foci_indices = foci_coord[:, ndim + 1]
            mask_rna_in_foci = np.isin(rna_coord[:, ndim], foci_indices)
            rna_in_foci = rna_coord[mask_rna_in_foci, :].copy()
            foci[rna_in_foci[:, ndim - 2], rna_in_foci[:, ndim - 1]] = True
            foci = stack.dilation_filter(foci, "square", rna_size)

        # case when only the foci centroid is known
        else:
            foci[foci_coord[:, ndim - 2], foci_coord[:, ndim - 1]] = True
            foci = stack.dilation_filter(foci, "square", foci_size)

    # get other layer
    other = np.zeros(image_shape, dtype=bool)
    if other_coord is not None:
        other_coord[:, ndim - 2] = other_coord[:, ndim - 2] - min_y + marge
        other_coord[:, ndim - 1] = other_coord[:, ndim - 1] - min_x + marge
        other[other_coord[:, ndim - 2], other_coord[:, ndim - 1]] = True
        other = stack.dilation_filter(other, "square", other_size)

    # build image coordinate
    image_coord = np.ones(shape=(image_shape[0], image_shape[1], 3),
                          dtype=np.float32)
    image_coord[cell, :] = [0, 0, 0]  # black
    image_coord[nuc, :] = [43 / 255, 131 / 255, 186 / 255]  # blue
    image_coord[rna, :] = [165 / 255, 0 / 255, 38 / 255]  # red
    image_coord[foci, :] = [244 / 255, 109 / 255, 67 / 255]  # orange
    image_coord[other, :] = [102 / 255, 204 / 255, 0 / 255]  # green

    # plot original image and coordinate representation
    if cell_coord is not None and image is not None:
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=framesize)

        # original image
        if remove_frame:
            ax[0].axis("off")
        if not rescale:
            vmin, vmax = get_minmax_values(image)
            ax[0].imshow(image, vmin=vmin, vmax=vmax)
        else:
            ax[0].imshow(image)
        if cell_mask is not None:
            cell_boundaries = stack.from_surface_to_boundaries(cell_mask)
            cell_boundaries = np.ma.masked_where(cell_boundaries == 0,
                                                 cell_boundaries)
            ax[0].imshow(cell_boundaries, cmap=ListedColormap(['red']))
        if nuc_mask is not None:
            nuc_boundaries = stack.from_surface_to_boundaries(nuc_mask)
            nuc_boundaries = np.ma.masked_where(nuc_boundaries == 0,
                                                nuc_boundaries)
            ax[0].imshow(nuc_boundaries, cmap=ListedColormap(['blue']))

        # coordinate image
        ax[1].imshow(image_coord)
        if foci_coord is not None:
            if ndim == 3:
                foci_coord_2d = foci_coord[:, 1:].copy()
            else:
                foci_coord_2d = foci_coord.copy()
            for (y, x, nb_rna, _) in foci_coord_2d:
                ax[1].text(x+5, y-5, str(nb_rna), color="#fdae61", size=20)

        # titles and frames
        if remove_frame:
            ax[0].axis("off")
            ax[1].axis("off")
        if title is not None:
            ax[0].set_title("Original image ({0})".format(title),
                            fontweight="bold", fontsize=10)
            ax[1].set_title("Coordinate representation ({0})".format(title),
                            fontweight="bold", fontsize=10)
        plt.tight_layout()

    # plot coordinate representation only
    elif cell_coord is not None and image is None:

        # title and frame
        if remove_frame:
            fig = plt.figure(figsize=framesize, frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
        else:
            plt.figure(figsize=framesize)
            if title is not None:
                plt.title("Coordinate representation ({0})".format(title),
                          fontweight="bold", fontsize=10)

        # coordinate image
        plt.imshow(image_coord)
        if foci_coord is not None:
            if ndim == 3:
                foci_coord_2d = foci_coord[:, 1:].copy()
            else:
                foci_coord_2d = foci_coord.copy()
            for (y, x, nb_rna, _) in foci_coord_2d:
                plt.text(x+5, y-5, str(nb_rna), color="#fdae61", size=20)

        if not remove_frame:
            plt.tight_layout()

    # plot original image only
    elif cell_coord is None and image is not None:
        plot_segmentation_boundary(
            image=image, cell_mask=cell_mask, nuc_mask=nuc_mask,
            rescale=rescale, title=title, framesize=framesize,
            remove_frame=remove_frame, path_output=path_output,
            ext=ext, show=show)

    # output
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_cell_coordinates(cell_coord, nuc_coord, rna_coord, foci_coord,
                          other_coord, title=None, remove_frame=False,
                          framesize=(15, 10), path_output=None, ext="png",
                          show=True):
    """

    Parameters
    ----------
    cell_coord : np.ndarray, np.int64
        Coordinates of the cell border with shape (nb_points, 2).
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
    title : str
        Title of the image.
    remove_frame : bool
        Remove axes and frame.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
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
    stack.check_parameter(title=(str, type(None)),
                          remove_frame=bool,
                          framesize=tuple,
                          path_output=(str, type(None)),
                          ext=(str, list))

    #cell_coord, others = stack.center_mask_coord(
    #    main=cell_coord,
    #    others=[nuc_coord, rna_coord, foci_coord, other_coord])
    #shape, min_y, min_x, marge = stack.from_coord_to_frame(cell_coord)

    # initialize plot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
        if title is not None:
            plt.title(title, fontweight="bold", fontsize=10)

    # plot
    plt.plot(cell_coord[:, 1], cell_coord[:, 0], c="black", linewidth=2)
    plt.plot(nuc_coord[:, 1], nuc_coord[:, 0], c="steelblue", linewidth=2)
    plt.scatter(rna_coord[:, 2], rna_coord[:, 1], s=25, c="firebrick",
                marker=".")

    if foci_coord is not None:
        plt.scatter(foci_coord[:, 2], foci_coord[:, 1], s=30, c="forestgreen",
                    marker="x")
        for (_, y, x, nb_rna, _) in foci_coord:
            plt.text(x + 5, y - 5, str(nb_rna), color="#66cc00", size=20)

    # format plot
    _, _, min_y, max_y = plt.axis()
    plt.ylim(max_y, min_y)
    plt.axis('scaled')
    if not remove_frame:
        plt.tight_layout()

    # output
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


