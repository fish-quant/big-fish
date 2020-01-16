# -*- coding: utf-8 -*-

"""
Functions to plot nucleus, cytoplasm and RNA coordinates.
"""
import bigfish.stack as stack

import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot, get_minmax_values
from matplotlib.colors import ListedColormap


def plot_volume(data_cell, id_cell, framesize=(7, 7), path_output=None,
                ext="png"):
    """Plot Cytoplasm and nucleus borders.

    Parameters
    ----------
    data_cell : pandas.DataFrame
        Dataframe with the coordinates of the cell.
    id_cell : int
        Id of the cell volume to plot.
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
    # TODO Sanity check of the dataframe

    # get cloud points
    cyto = data_cell.loc[id_cell, "pos_cell"]
    cyto = np.array(cyto)
    nuc = data_cell.loc[id_cell, "pos_nuc"]
    nuc = np.array(nuc)

    # plot
    plt.figure(figsize=framesize)
    plt.plot(cyto[:, 1], cyto[:, 0], c="black", linewidth=2)
    plt.plot(nuc[:, 1], nuc[:, 0], c="steelblue", linewidth=2)
    plt.title("Cell id: {}".format(id_cell), fontweight="bold", fontsize=15)
    plt.tight_layout()
    save_plot(path_output, ext)
    plt.show()

    return


def plot_rna(data_merged, id_cell, framesize=(7, 7), path_output=None,
             ext="png"):
    """Plot cytoplasm border and RNA spots.

    Parameters
    ----------
    data_merged : pandas.DataFrame
        Dataframe with the coordinate of the cell and those of the RNA.
    id_cell : int
        ID of the cell to plot.
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
    # TODO Sanity check of the dataframe

    # get cloud points
    cyto = data_merged.loc[id_cell, "pos_cell"]
    cyto = np.array(cyto)
    rna = data_merged.loc[id_cell, "RNA_pos"]
    rna = np.array(rna)

    # plot
    plt.figure(figsize=framesize)
    plt.plot(cyto[:, 1], cyto[:, 0], c="black", linewidth=2)
    plt.scatter(rna[:, 1], rna[:, 0], c="firebrick", s=50, marker="x")
    plt.title("Cell id: {}".format(id_cell), fontweight="bold", fontsize=15)
    plt.tight_layout()
    save_plot(path_output, ext)
    plt.show()

    return


def plot_distribution_rna(data, data_validation=None, data_test=None,
                          framesize=(10, 5), path_output=None, ext="png"):
    """Plot RNA distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with all the data (or the train data in case of split data).
    data_validation : pandas.DataFrame
        Dataframe with the validation data
    data_test : pandas.DataFrame
        Dataframe with the test data.
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
    # plot one histogram
    if data_validation is None and data_test is None:
        plt.figure(figsize=framesize)
        plt.title("RNA distribution", fontweight="bold")
        plt.hist(data["nb_rna"], bins=100, color="steelblue",
                 edgecolor='black', linewidth=1.2)
        plt.xlabel("Number of RNA")
        plt.ylabel("Frequency")
        plt.tight_layout()
        save_plot(path_output, ext)
        plt.show()

    # plot several histograms
    elif data_validation is not None and data_test is not None:
        fig, ax = plt.subplots(3, 1, sharex="col", figsize=framesize)
        ax[0].hist(data["nb_rna"], bins=100, color="steelblue",
                   edgecolor='black', linewidth=1.2)
        ax[0].set_title("RNA distribution (train)", fontweight="bold",
                        fontsize=15)
        ax[0].set_ylabel("Frequency")
        ax[1].hist(data_validation["nb_rna"], bins=100, color="steelblue",
                   edgecolor='black', linewidth=1.2)
        ax[1].set_title("RNA distribution (validation)", fontweight="bold",
                        fontsize=15)
        ax[1].set_ylabel("Frequency")
        ax[2].hist(data_test["nb_rna"], bins=100, color="steelblue",
                   edgecolor='black', linewidth=1.2)
        ax[2].set_title("RNA distribution (test)", fontweight="bold",
                        fontsize=15)
        ax[2].set_ylabel("Frequency")
        ax[2].set_xlabel("Number of RNA")
        plt.tight_layout()
        save_plot(path_output, ext)
        plt.show()

    return


def plot_cell_coordinates(cyt_coord, nuc_coord, rna_coord, foci_coord,
                          count_rna=False, title=None, remove_frame=False,
                          framesize=(15, 10), path_output=None, ext="png",
                          show=True):
    """

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Coordinates of the cytoplasm border with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Coordinates of the nuclei border with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Coordinates of the RNA spots with shape (nb_spots, 4). One
        coordinate per dimension (zyx dimension), plus the index of a
        potential foci.
    foci_coord : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of RNAs detected in the
        foci and its index.
    count_rna : bool
        Display the number of RNAs in a foci.
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
    stack.check_array(cyt_coord,
                      ndim=2,
                      dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord,
                          ndim=2,
                          dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord,
                          ndim=2,
                          dtype=[np.int64])
    if foci_coord is not None:
        stack.check_array(foci_coord,
                          ndim=2,
                          dtype=[np.int64])
    stack.check_parameter(count_rna=bool,
                          title=(str, type(None)),
                          remove_frame=bool,
                          framesize=tuple,
                          path_output=(str, type(None)),
                          ext=(str, list))

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
    plt.plot(cyt_coord[:, 1], cyt_coord[:, 0], c="black", linewidth=2)
    plt.plot(nuc_coord[:, 1], nuc_coord[:, 0], c="steelblue", linewidth=2)
    plt.scatter(rna_coord[:, 2], rna_coord[:, 1], s=25, c="firebrick",
                marker=".")
    if count_rna and foci_coord is not None:
        plt.scatter(foci_coord[:, 2], foci_coord[:, 1], s=30, c="forestgreen",
                    marker="x")
        for (_, y, x, nb_rna, _) in foci_coord:
            plt.text(x + 5, y - 5, str(nb_rna), color="#66CC00", size=20)

    # format plot
    _, _, min_y, max_y = plt.axis()
    plt.ylim(max_y, min_y)
    if not remove_frame:
        plt.tight_layout()

    # save and show
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_layers_coordinates(layers, titles=None, framesize=(5, 10),
                            remove_frame=False, path_output=None, ext="png",
                            show=True):
    """Plot input layers of the classification model.

    Parameters
    ----------
    layers : List[np.ndarray]
        List of the input images feed into the model.
    titles : List[str]
        List of the subtitles.
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
    # TODO to improve

    # plot
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=framesize)
    if remove_frame:
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
    ax[0].imshow(layers[0], cmap="binary", origin='lower')
    ax[1].imshow(layers[1], cmap="binary", origin='lower')
    ax[2].imshow(layers[2], cmap="binary", origin='lower')
    ax[0].tick_params(top='False', bottom='False',
                      left='False', right='False',
                      labelleft='False', labelbottom='False')
    ax[1].tick_params(top='False', bottom='False',
                      left='False', right='False',
                      labelleft='False', labelbottom='False')
    ax[2].tick_params(top='False', bottom='False',
                      left='False', right='False',
                      labelleft='False', labelbottom='False')

    if titles is not None and not remove_frame:
        ax[0].set_title(titles[0], fontweight="bold", fontsize=15)
        ax[1].set_title(titles[1], fontweight="bold", fontsize=15)
        ax[2].set_title(titles[2], fontweight="bold", fontsize=15)
    if not remove_frame:
        plt.tight_layout()

    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_cell(cyt_coord, nuc_coord=None, rna_coord=None, foci_coord=None,
              image_cyt=None, mask_cyt=None, mask_nuc=None, count_rna=False,
              dilation=False, title=None, remove_frame=False, rescale=False,
              framesize=(15, 10), path_output=None, ext="png", show=True):
    """
    Plot image and coordinates extracted for a specific cell.

    Parameters
    ----------
    cyt_coord : np.ndarray, np.int64
        Coordinates of the cytoplasm border with shape (nb_points, 2).
    nuc_coord : np.ndarray, np.int64
        Coordinates of the nuclei border with shape (nb_points, 2).
    rna_coord : np.ndarray, np.int64
        Coordinates of the RNA spots with shape (nb_spots, 4). One
        coordinate per dimension (zyx dimension), plus the index of a
        potential foci.
    foci_coord : np.ndarray, np.int64
        Array with shape (nb_foci, 5). One coordinate per dimension for the
        foci centroid (zyx coordinates), the number of RNAs detected in the
        foci and its index.
    image_cyt : np.ndarray, np.uint
        Original image of the cytoplasm.
    mask_cyt : np.ndarray, np.uint
        Mask of the cytoplasm.
    mask_nuc : np.ndarray, np.uint
        Mask of the nucleus.
    count_rna : bool
        Display the number of RNAs in a foci.
    dilation : bool
        Dilate the spot and foci representations in the plot.
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
    # check parameters
    stack.check_array(cyt_coord,
                      ndim=2,
                      dtype=[np.int64])
    if nuc_coord is not None:
        stack.check_array(nuc_coord,
                          ndim=2,
                          dtype=[np.int64])
    if rna_coord is not None:
        stack.check_array(rna_coord,
                          ndim=2,
                          dtype=[np.int64])
    if foci_coord is not None:
        stack.check_array(foci_coord,
                          ndim=2,
                          dtype=[np.int64])
    if image_cyt is not None:
        stack.check_array(image_cyt,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64])
    if mask_cyt is not None:
        stack.check_array(mask_cyt,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    if mask_nuc is not None:
        stack.check_array(mask_nuc,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64, bool])
    stack.check_parameter(count_rna=bool,
                          dilation=bool,
                          title=(str, type(None)),
                          remove_frame=bool,
                          rescale=bool,
                          framesize=tuple,
                          path_output=(str, type(None)),
                          ext=(str, list))
    if title is None:
        title = ""

    # get shape of image built from coordinates
    image_shape, min_y, min_x, marge = stack.from_coord_to_frame(cyt_coord)

    # get cytoplasm layer
    cyt = np.zeros(image_shape, dtype=bool)
    cyt_coord[:, 0] = cyt_coord[:, 0] - min_y + marge
    cyt_coord[:, 1] = cyt_coord[:, 1] - min_x + marge
    cyt[cyt_coord[:, 0], cyt_coord[:, 1]] = True

    # get nucleus layer
    nuc = np.zeros(image_shape, dtype=bool)
    if nuc_coord is not None:
        nuc_coord[:, 0] = nuc_coord[:, 0] - min_y + marge
        nuc_coord[:, 1] = nuc_coord[:, 1] - min_x + marge
        nuc[nuc_coord[:, 0], nuc_coord[:, 1]] = True
        print(nuc_coord)

    # get rna layer
    rna = np.zeros(image_shape, dtype=bool)
    if rna_coord is not None:
        rna_coord[:, 1] = rna_coord[:, 1] - min_y + marge
        rna_coord[:, 2] = rna_coord[:, 2] - min_x + marge
        rna[rna_coord[:, 1], rna_coord[:, 2]] = True
        print(rna_coord)
        if dilation:
            rna = stack.dilation_filter(rna,
                                        kernel_shape="square",
                                        kernel_size=3)

    # get foci layer
    foci = np.zeros(image_shape, dtype=bool)
    if foci_coord is not None:
        rna_in_foci_coord = rna_coord[rna_coord[:, 3] != -1, :].copy()
        foci[rna_in_foci_coord[:, 1], rna_in_foci_coord[:, 2]] = True
        if dilation:
            foci = stack.dilation_filter(foci,
                                         kernel_shape="square",
                                         kernel_size=3)

    # build image coordinate
    image_coord = np.ones(shape=(image_shape[0], image_shape[1], 3),
                          dtype=np.float32)
    image_coord[cyt, :] = [0, 0, 0]  # black
    image_coord[nuc, :] = [0, 102 / 255, 204 / 255]  # blue
    image_coord[rna, :] = [204 / 255, 0, 0]  # red
    image_coord[foci, :] = [102 / 255, 204 / 255, 0]  # green

    # plot original and coordinate image
    if image_cyt is not None:
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=framesize)

        # original image
        if remove_frame:
            ax[0].axis("off")
        if not rescale:
            vmin, vmax = get_minmax_values(image_cyt)
            ax[0].imshow(image_cyt, vmin=vmin, vmax=vmax)
        else:
            ax[0].imshow(image_cyt)
        if mask_cyt is not None:
            boundaries_cyt = stack.from_surface_to_boundaries(mask_cyt)
            boundaries_cyt = np.ma.masked_where(boundaries_cyt == 0,
                                                boundaries_cyt)
            ax[0].imshow(boundaries_cyt, cmap=ListedColormap(['red']))
        if mask_nuc is not None:
            boundaries_nuc = stack.from_surface_to_boundaries(mask_nuc)
            boundaries_nuc = np.ma.masked_where(boundaries_nuc == 0,
                                                boundaries_nuc)
            ax[0].imshow(boundaries_nuc, cmap=ListedColormap(['blue']))
        ax[0].set_title("Original image ({0})".format(title),
                        fontweight="bold", fontsize=10)

        # coordinate image
        if remove_frame:
            ax[1].axis("off")
        ax[1].imshow(image_coord)
        if count_rna and foci_coord is not None:
            for (_, y, x, nb_rna, _) in foci_coord:
                ax[1].text(x+5, y-5, str(nb_rna), color="#66CC00", size=20)
        ax[1].set_title("Coordinate image ({0})".format(title),
                        fontweight="bold", fontsize=10)

        plt.tight_layout()

    # plot coordinate image only
    else:
        if remove_frame:
            fig = plt.figure(figsize=framesize, frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
        else:
            plt.figure(figsize=framesize)
            plt.title(title, fontweight="bold", fontsize=10)
        plt.imshow(image_coord)
        if count_rna and foci_coord is not None:
            for (_, y, x, nb_rna, _) in foci_coord:
                plt.text(x+5, y-5, str(nb_rna), color="#66CC00", size=20)

        if not remove_frame:
            plt.tight_layout()

    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return
