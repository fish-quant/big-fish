# -*- coding: utf-8 -*-

"""
Function to plot 2-d images.
"""

import bigfish.stack as stack

import matplotlib.pyplot as plt
import numpy as np


# TODO add title in the plot and remove axes

def plot_yx(tensor, r=0, c=0, z=0, title=None, framesize=(15, 15),
            path_output=None, ext="png"):
    """Plot the selected x and y dimensions of an image.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 2-d, 3-d or 5-d tensor with shape (y, x), (z, y, x) or
        (r, c, z, y, x) respectively.
    r : int
        Index of the round to keep.
    c : int
        Index of the channel to keep.
    z : int
        Index of the z slice to keep.
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # check tensor
    stack.check_array(tensor, ndim=[2, 3, 5],
                      dtype=[np.uint8, np.uint16, np.float32, bool])

    # get the 2-d tensor
    xy_tensor = None
    if tensor.ndim == 2:
        xy_tensor = tensor
    elif tensor.ndim == 3:
        xy_tensor = tensor[z, :, :]
    elif tensor.ndim == 5:
        xy_tensor = tensor[r, c, z, :, :]

    # plot
    plt.figure(figsize=framesize)
    plt.imshow(xy_tensor)
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=25)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # save the plot
    _save_plot(path_output, ext)

    return


def plot_images(images, framesize=(15, 15), path_output=None, ext="png"):
    """Plot or subplot of 2-d images.

    Parameters
    ----------
    images : np.ndarray or List[np.ndarray]
        Images with shape (y, x).
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # enlist image if necessary
    if isinstance(images, np.ndarray):
        images = [images]

    # check images
    for image in images:
        stack.check_array(image, ndim=2,
                          dtype=[np.uint8, np.uint16, np.float32, np.float64,
                                 bool])

    # we plot 3 images by row maximum
    nrow = int(np.ceil(len(images)/3))
    ncol = min(len(images), 3)

    # plot one image
    if len(images) == 1:
        plot_yx(images[0], framesize=framesize,
                path_output=path_output, ext=ext)
        return

    # plot multiple images
    fig, ax = plt.subplots(nrow, ncol, figsize=framesize)
    if len(images) in [2, 3]:
        for i, image in enumerate(images):
            ax[i].imshow(image)
    else:
        for i, image in enumerate(images):
            row = i // 3
            col = i % 3
            ax[row, col].imshow(image)
    plt.tight_layout()
    plt.show()

    # save the plot
    _save_plot(path_output, ext)


def plot_channels_2d(tensor, r=0, z=0, framesize=(15, 15), path_output=None,
                     ext="png"):
    """Subplot the selected x and y dimensions of an image for all channels.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 5-d tensor with shape (r, c, z, y, x).
    r : int
        Index of the round to keep.
    z : int
        Index of the z slice to keep.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # check tensor
    stack.check_array(tensor, ndim=5, dtype=[np.uint8, np.uint16])

    # get the number of channels
    nb_channels = tensor.shape[1]

    # plot
    fig, ax = plt.subplots(1, nb_channels, sharex='col', figsize=framesize)
    for i in range(nb_channels):
        ax[i].imshow(tensor[r, i, z, :, :])
    plt.tight_layout()
    plt.show()

    # save the plot
    _save_plot(path_output, ext)

    return


def plot_projection(tensor, projection, r=0, c=0, z=0, framesize=(15, 15),
                    path_output=None, ext="png"):
    """Plot result of a 2-d projection.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 5-d tensor with shape (r, c, z, y, x).
    projection : np.ndarray, np.uint8
        A 2-d image with shape (y, x).
    r : int
        Index of the round to keep.
    c : int
        Index of the channel to keep.
    z : int
        Index of the z-slice to keep.
    framesize : tuple
        Size of the frame used to plot (plt.figure(figsize=framesize).
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # check tensor
    stack.check_array(tensor, ndim=5, dtype=[np.uint8, np.uint16])
    stack.check_array(projection, ndim=2, dtype=[np.uint8, np.uint16])

    # plot
    fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)
    ax[0].imshow(tensor[r, c, z, :, :])
    ax[1].imshow(projection)
    plt.tight_layout()
    plt.show()

    # save the plot
    _save_plot(path_output, ext)


def plot_segmentation(tensor, segmentation, r=0, c=0, z=0, label=None,
                      framesize=(15, 15), path_output=None, ext="png"):
    """Plot result of a 2-d segmentation, with labelled instances is available.

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 5-d tensor with shape (r, c, z, y, x).
    segmentation : np.ndarray, bool
        A 2-d image with shape (y, x).
    r : int
        Index of the round to keep.
    c : int
        Index of the channel to keep.
    z : int
        Index of the z-slice to keep.
    label : np.ndarray, np.int64
        A 2-d image with shape (y, x).
    framesize : tuple
        Size of the frame used to plot (plt.figure(figsize=framesize).
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # check tensor
    stack.check_array(tensor, ndim=5, dtype=[np.uint8, np.uint16])
    stack.check_array(segmentation, ndim=2, dtype=bool)
    if label is not None:
        stack.check_array(label, ndim=2, dtype=np.int64)

    # plot
    if label is not None:
        fig, ax = plt.subplots(1, 3, sharex='col', figsize=framesize)
        ax[0].imshow(tensor[r, c, z, :, :])
        ax[1].imshow(segmentation)
        ax[2].imshow(label)
    else:
        fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)
        ax[0].imshow(tensor[r, c, z, :, :])
        ax[1].imshow(segmentation)
    plt.tight_layout()
    plt.show()

    # save the plot
    _save_plot(path_output, ext)


def plot_spot_detection(tensor, coordinates, radius, r=0, c=0, z=0,
                        framesize=(15, 15), projection_2d=False,
                        path_output=None, ext="png"):
    """

    Parameters
    ----------
    tensor : np.ndarray, np.uint
        A 5-d tensor with shape (r, c, z, y, x).
    coordinates : np.ndarray, np.int64
        Coordinate of the spots with shape (nb_spots, 3) or
        (nb_spots, 2) for 3-d or 2-d images respectively.
    radius : float
        Radius of the detected spots.
    r : int
        Index of the round to keep.
    c : int
        Index of the channel to keep.
    z : int
        Index of the z-slice to keep.
    framesize : tuple
        Size of the frame used to plot (plt.figure(figsize=framesize).
    projection_2d : bool
        Project the image in 2-d and plot the spot detected on the projection.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # TODO check coordinates shape
    # check tensor
    stack.check_array(tensor, ndim=5, dtype=[np.uint8, np.uint16])
    stack.check_array(coordinates, ndim=2, dtype=np.int64)

    # projection 2d
    if projection_2d:
        image_2d = stack.projection(tensor,
                                    method="mip",
                                    r=r,
                                    c=c)

        # plot
        fig, ax = plt.subplots(1, 2, figsize=framesize)
        ax[0].imshow(image_2d)
        ax[1].imshow(image_2d)
        for spot_coordinate in coordinates:
            _, y, x = spot_coordinate
            c = plt.Circle((x, y), radius,
                           color="red",
                           linewidth=1,
                           fill=False)
            ax[1].add_patch(c)
        plt.tight_layout()
        plt.show()

    # a specific z-slice
    else:
        # keep spot detected for a specific height
        if coordinates.shape[1] == 3:
            coordinates = coordinates[coordinates[:, 0] == z]
            coordinates = coordinates[:, 1:]

        image_2d = tensor[r, c, z, :, :]

        # plot
        fig, ax = plt.subplots(1, 2, figsize=framesize)
        ax[0].imshow(image_2d)
        ax[1].imshow(image_2d)
        for spot_coordinate in coordinates:
            y, x = spot_coordinate
            c = plt.Circle((x, y), radius,
                           color="red",
                           linewidth=1,
                           fill=False)
            ax[1].add_patch(c)
        plt.tight_layout()
        plt.show()

    # save the plot
    _save_plot(path_output, ext)


def _save_plot(path_output, ext):
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
    # save the plot
    if path_output is not None:
        if isinstance(ext, str):
            plt.savefig(path_output, format=ext)
        elif isinstance(ext, list):
            for ext_ in ext:
                plt.savefig(path_output, format=ext_)
        else:
            Warning("Plot is not saved because the extension is not valid: "
                    "{0}.".format(ext))
