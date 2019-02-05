# -*- coding: utf-8 -*-

"""
Function to plot 2-d images.
"""

import matplotlib.pyplot as plt


def plot_yx(tensor, round=0, channel=0, z=0, title=None, framesize=(15, 15),
            path_output=None, ext="png"):
    """Plot the selected x and Y dimensions of an image.

    Parameters
    ----------
    tensor : np.ndarray, np.float32
        A 2-d, 3-d or 5-d tensor with shape (y, x), (z, y, x) or
        (round, channel, z, y, x) respectively.
    round : int
        Indice of the round to keep.
    channel : int
        Indice of the channel to keep.
    z : int
        Indice of the z slice to keep.
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot (plt.figure(figsize=framesize).
    path_output : str
        Path to save the image (without extension).
    ext : str or list
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.

    Returns
    -------

    """
    # get the 2-d tensor
    if tensor.ndim == 2:
        xy_tensor = tensor
    elif tensor.ndim == 3:
        xy_tensor = tensor[z, :, :]
    elif tensor.ndim == 5:
        xy_tensor = tensor[round, channel, z, :, :]
    else:
        raise ValueError("{0} is not a valid shape for the tensor."
                         .format(tensor.shape))

    # plot
    plt.figure(figsize=framesize)
    plt.imshow(xy_tensor)
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=25)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

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

    return

