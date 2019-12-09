# -*- coding: utf-8 -*-

"""
Function to plot 2-d images.
"""

import bigfish.stack as stack

import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot, get_minmax_values

from skimage.segmentation import find_boundaries
from matplotlib.colors import ListedColormap


# TODO clean this script (remove useless functions)
# TODO add parameter to show the figure

def plot_yx(tensor, r=0, c=0, z=0, rescale=False, title=None,
            framesize=(8, 8), remove_frame=False, path_output=None,
            ext="png", show=True):
    """Plot the selected yx plan of the selected dimensions of an image.

    Parameters
    ----------
    tensor : np.ndarray
        A 2-d, 3-d or 5-d tensor with shape (y, x), (z, y, x) or
        (r, c, z, y, x) respectively.
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
                      ndim=[2, 3, 5],
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
    xy_tensor = None
    if tensor.ndim == 2:
        xy_tensor = tensor
    elif tensor.ndim == 3:
        xy_tensor = tensor[z, :, :]
    elif tensor.ndim == 5:
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


def plot_images(tensors, rescale=False, titles=None, framesize=(15, 5),
                remove_frame=False, path_output=None, ext="png", show=True):
    """Plot or subplot of 2-d images.

    Parameters
    ----------
    tensors : np.ndarray or List[np.ndarray]
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
    if isinstance(tensors, np.ndarray):
        tensors = [tensors]

    # check parameters
    stack.check_parameter(tensors=list,
                          rescale=bool,
                          titles=(str, list, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)
    for tensor in tensors:
        stack.check_array(tensor,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64,
                                 np.float32, np.float64,
                                 bool])

    # we plot 3 images by row maximum
    nrow = int(np.ceil(len(tensors)/3))
    ncol = min(len(tensors), 3)

    # plot one image
    if len(tensors) == 1:
        if titles is not None:
            title = titles[0]
        else:
            title = None
        plot_yx(tensors[0],
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
    if len(tensors) in [2, 3]:
        for i, tensor in enumerate(tensors):
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
        r = nrow * 3 - len(tensors)
        tensors_completed = [tensor for tensor in tensors] + [None] * r

        for i, tensor in enumerate(tensors_completed):
            row = i // 3
            col = i % 3
            if tensor is None:
                ax[row, col].set_visible(False)
                continue
            if remove_frame:
                ax[row, col].axis("off")
            if not rescale:
                vmin, vmax = get_minmax_values(tensor)
                ax[row, col].imshow(tensor, vmin=vmin, vmax=vmax)
            else:
                ax[row, col].imshow(tensor)
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


def plot_channels_2d(tensor, r=0, z=0, rescale=False, titles=None,
                     framesize=(15, 5), remove_frame=False, path_output=None,
                     ext="png"):
    """Subplot the yx plan of the selected dimensions of an image for all
    channels.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 5-d tensor with shape (r, c, z, y, x).
    r : int
        Index of the round to keep.
    z : int
        Index of the z slice to keep.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    titles : List[str]
        Titles of the subplots (one per channel).
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # check parameters
    stack.check_array(tensor,
                      ndim=5,
                      dtype=[np.uint8, np.uint16])
    stack.check_parameter(r=int,
                          z=int,
                          rescale=bool,
                          titles=(list, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list))

    # get the number of channels
    nb_channels = tensor.shape[1]

    # get the minimum and maximal values of the tensor dtype
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(tensor)

    # plot
    fig, ax = plt.subplots(1, nb_channels, sharex='col', figsize=framesize)
    for i in range(nb_channels):
        if not rescale:
            ax[i].imshow(tensor[r, i, z, :, :], vmin=vmin, vmax=vmax)
        else:
            ax[i].imshow(tensor[r, i, z, :, :])
        if titles is not None:
            ax[i].set_title(titles[i], fontweight="bold", fontsize=10)
        if remove_frame:
            ax[i].axis("off")

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    plt.show()

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


def plot_segmentation(tensor, mask, rescale=False, title=None,
                      framesize=(15, 5), remove_frame=False,
                      path_output=None, ext="png", show=True):
    """Plot result of a 2-d segmentation, with labelled instances if available.

    Parameters
    ----------
    tensor : np.ndarray
        A 2-d tensor with shape (y, x).
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
    stack.check_array(tensor,
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
        vmin, vmax = get_minmax_values(tensor)

    # plot
    fig, ax = plt.subplots(1, 3, sharex='col', figsize=framesize)

    # image
    if not rescale:
        ax[0].imshow(tensor, vmin=vmin, vmax=vmax)
    else:
        ax[0].imshow(tensor)
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
        ax[2].imshow(tensor, vmin=vmin, vmax=vmax)
    else:
        ax[2].imshow(tensor)
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


def plot_segmentation_boundary(tensor, mask_nuc=None, mask_cyt=None,
                               rescale=False, title=None, framesize=(10, 10),
                               remove_frame=False, path_output=None,
                               ext="png", show=True):
    """Plot the boundary of the segmented objects.

    Parameters
    ----------
    tensor : np.ndarray
        A 2-d tensor with shape (y, x).
    mask_nuc : np.ndarray
        A 2-d image with shape (y, x).
    mask_cyt : np.ndarray
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
    stack.check_array(tensor,
                      ndim=2,
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64,
                             bool])
    if mask_nuc is not None:
        stack.check_array(mask_nuc,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    if mask_cyt is not None:
        stack.check_array(mask_cyt,
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
        vmin, vmax = get_minmax_values(tensor)

    # get boundaries
    boundaries_nuc = None
    boundaries_cyt = None
    if mask_nuc is not None:
        boundaries_nuc = find_boundaries(mask_nuc, mode='thick')
        boundaries_nuc = np.ma.masked_where(boundaries_nuc == 0,
                                            boundaries_nuc)
    if mask_cyt is not None:
        boundaries_cyt = find_boundaries(mask_cyt, mode='thick')
        boundaries_cyt = np.ma.masked_where(boundaries_cyt == 0,
                                            boundaries_cyt)

    # plot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
    if not rescale:
        plt.imshow(tensor, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(tensor)
    if mask_nuc is not None:
        plt.imshow(boundaries_nuc, cmap=ListedColormap(['blue']))
    if mask_cyt is not None:
        plt.imshow(boundaries_cyt, cmap=ListedColormap(['red']))
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


def plot_spot_detection(tensor, spots, radius_yx, rescale=False,
                        title=None, framesize=(15, 5), remove_frame=False,
                        path_output=None, ext="png", show=True):
    """Plot detected spot on a 2-d image.

    Parameters
    ----------
    tensor : np.ndarray
        A 2-d tensor with shape (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or
        (nb_spots, 2) for 3-d or 2-d images respectively.
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
    # TODO check coordinates shape
    # check parameters
    stack.check_array(tensor,
                      ndim=2,
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64])
    stack.check_array(spots,
                      ndim=2,
                      dtype=[np.int64])
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
        vmin, vmax = get_minmax_values(tensor)

    # plot
    fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)

    # image
    if not rescale:
        ax[0].imshow(tensor, vmin=vmin, vmax=vmax)
    else:
        ax[0].imshow(tensor)
    if title is not None:
        ax[0].set_title(title, fontweight="bold", fontsize=10)
    if remove_frame:
        ax[0].axis("off")

    # spots
    if not rescale:
        ax[1].imshow(tensor, vmin=vmin, vmax=vmax)
    else:
        ax[1].imshow(tensor)
    for spot_coordinate in spots:
        _, y, x = spot_coordinate
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


def plot_foci_detection(tensor, spots, foci, radius_spots_yx,
                        rescale=False, title=None, framesize=(15, 10),
                        remove_frame=False, path_output=None, ext="png",
                        show=True):
    """Plot detected spots and foci on a 2-d image.

    Parameters
    ----------
    tensor : np.ndarray
        A 2-d tensor with shape (y, x).
    spots : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3).
    foci : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension (zyx
        coordinates), number of RNAs in the foci and index of the foci.
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
    # TODO check coordinates shape
    # TODO allow a plot for a specific z-slice
    # check parameters
    stack.check_array(tensor,
                      ndim=2,
                      dtype=[np.uint8, np.uint16,
                             np.float32, np.float64])
    stack.check_array(foci,
                      ndim=2,
                      dtype=[np.int64])
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
                          dtype=[np.int64])

    # get minimum and maximum value of the image
    vmin, vmax = None, None
    if not rescale:
        vmin, vmax = get_minmax_values(tensor)

    # plot
    fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)

    # image
    if not rescale:
        ax[0].imshow(tensor, vmin=vmin, vmax=vmax)
    else:
        ax[0].imshow(tensor)
    if title is not None:
        ax[0].set_title(title, fontweight="bold", fontsize=10)
    if remove_frame:
        ax[0].axis("off")

    # spots and foci
    if not rescale:
        ax[1].imshow(tensor, vmin=vmin, vmax=vmax)
    else:
        ax[1].imshow(tensor)
    if spots is not None:
        for (_, y, x) in spots:
            c = plt.Circle((x, y), radius_spots_yx,
                           color="red",
                           linewidth=1,
                           fill=False)
            ax[1].add_patch(c)
        title_ = "Detected spots and foci"
    else:
        title_ = "Detected foci"
    for (_, y, x, _, _) in foci:
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
