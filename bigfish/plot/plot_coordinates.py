# -*- coding: utf-8 -*-

"""
Functions to plot nucleus, cytoplasm and RNA coordinates.
"""
import bigfish.stack as stack

import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot, get_minmax_values

from skimage.segmentation import find_boundaries
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


def plot_cell_coordinates(data, id_cell, title=None, framesize=(5, 10),
                          path_output=None, ext="png"):
    """

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with all the data.
    id_cell : int
        Index of the cell to plot
    title : str
        Title of the plot
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
    # get the cytoplasm, the nuclei and the rna spots
    rna_coord, cyt_coord, nuc_coord = stack.get_coordinates(data, id_cell)

    # plot
    plt.figure(figsize=framesize)
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=25)
    plt.plot(cyt_coord[:, 1], cyt_coord[:, 0], c="black", linewidth=2)
    plt.plot(nuc_coord[:, 1], nuc_coord[:, 0], c="steelblue", linewidth=2)
    plt.scatter(rna_coord[:, 1], rna_coord[:, 0], s=25, c="firebrick",
                marker=".")
    plt.tight_layout()
    save_plot(path_output, ext)
    plt.show()

    return


def plot_layers_coordinates(layers, titles=None, framesize=(5, 10),
                            path_output=None, ext="png"):
    """Plot input layers of the classification model.

    Parameters
    ----------
    layers : List[np.ndarray]
        List of the input images feed into the model.
    titles : List[str]
        List of the subtitles.
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
    # plot
    fig, ax = plt.subplots(1, 3, figsize=framesize)
    ax[0].imshow(layers[0], cmap="binary", origin='lower')
    ax[1].imshow(layers[1], cmap="binary", origin='lower')
    ax[2].imshow(layers[2], cmap="binary", origin='lower')
    if titles is not None:
        ax[0].set_title(titles[0], fontweight="bold", fontsize=15)
        ax[1].set_title(titles[1], fontweight="bold", fontsize=15)
        ax[2].set_title(titles[2], fontweight="bold", fontsize=15)
    plt.tight_layout()
    save_plot(path_output, ext)
    plt.show()

    return


def plot_extraction_image(results, remove_frame=False, title=None,
                          framesize=None, path_output=None, ext="png",
                          show=True):
    """Plot or subplot of 2-d coordinates extracted from an image.

    Parameters
    ----------
    results : List[(cyt_coord, nuc_coord, rna_coord, cell_foci, cell)]
        - cyt_coord : np.ndarray, np.int64
            Coordinates of the cytoplasm border with shape (nb_points, 2).
        - nuc_coord : np.ndarray, np.int64
            Coordinates of the nuclei border with shape (nb_points, 2).
        - rna_coord : np.ndarray, np.int64
            Coordinates of the RNA spots with shape (nb_spots, 3). One
            coordinate per dimension (yx dimension), plus the index of a
            potential foci.
        - cell_foci : np.ndarray, np.int64
            Array with shape (nb_foci, 7). One coordinate per dimension for the
            foci centroid (zyx coordinates), the number of RNAs detected in the
            foci, its index, the area of the foci region and its maximum
            intensity value.
        - cell : Tuple[int]
            Box coordinate of the cell in the original image (min_y, min_x,
            max_y and max_x).
    remove_frame : bool
        Remove axes and frame.
    title : str
        Title of the image.
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
    stack.check_parameter(results=list,
                          remove_frame=bool,
                          title=(str, type(None)),
                          framesize=(tuple, type(None)),
                          path_output=(str, type(None)),
                          ext=(str, list))

    # we plot 3 images by row maximum
    nrow = int(np.ceil(len(results)/3))
    ncol = min(len(results), 3)
    if framesize is None:
        framesize = (5 * ncol, 5 * nrow)

    # plot one image
    marge = stack.get_offset_value()
    if len(results) == 1:
        cyt, nuc, rna, foci, _ = results[0]
        if remove_frame:
            fig = plt.figure(figsize=(8, 8), frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
        else:
            plt.figure(figsize=(8, 8))
        plt.xlim(-marge, max(cyt[:, 1]) + marge)
        plt.ylim(max(cyt[:, 0]) + marge, -marge)
        plt.scatter(cyt[:, 1], cyt[:, 0], c="black", s=5, marker=".")
        plt.scatter(nuc[:, 1], nuc[:, 0], c="steelblue", s=5, marker=".")
        plt.scatter(rna[:, 1], rna[:, 0], c="firebrick", s=50, marker="x")
        if len(foci) > 0:
            plt.scatter(foci[:, 2], foci[:, 1], c="chartreuse", s=60,
                        marker="D")
        if title is not None and not remove_frame:
            title_plot = title + "_cell_0"
            plt.title(title_plot, fontweight="bold", fontsize=25)
        if not remove_frame:
            plt.tight_layout()
        if path_output is not None:
            save_plot(path_output, ext)
        if show:
            plt.show()
        else:
            plt.close()

        return

    # plot multiple images
    fig, ax = plt.subplots(nrow, ncol, figsize=framesize)

    # one row
    if len(results) in [2, 3]:
        for i, (cyt, nuc, rna, foci, _) in enumerate(results):
            if remove_frame:
                ax[i].axis("off")
            ax[i].set_xlim(-marge, max(cyt[:, 1]) + marge)
            ax[i].set_ylim(max(cyt[:, 0]) + marge, -marge)
            ax[i].scatter(cyt[:, 1], cyt[:, 0], c="black", s=5, marker=".")
            ax[i].scatter(nuc[:, 1], nuc[:, 0], c="steelblue", s=5, marker=".")
            ax[i].scatter(rna[:, 1], rna[:, 0], c="firebrick", s=50,
                          marker="x")
            if len(foci) > 0:
                ax[i].scatter(foci[:, 2], foci[:, 1], c="chartreuse", s=60,
                              marker="D")
            if title is not None:
                title_plot = title + "_cell_{0}".format(i)
                ax[i].set_title(title_plot, fontweight="bold", fontsize=10)

    # several rows
    else:
        # we complete the row with empty frames
        r = nrow * 3 - len(results)
        results_completed = [(cyt, nuc, rna, foci, _)
                             for (cyt, nuc, rna, foci, _) in results]
        results_completed += [None] * r
        for i, result in enumerate(results_completed):
            row = i // 3
            col = i % 3
            if result is None:
                ax[row, col].set_visible(False)
                continue
            else:
                cyt, nuc, rna, foci, cell = result
            if remove_frame:
                ax[row, col].axis("off")
            ax[row, col].set_xlim(-marge, max(cyt[:, 1]) + marge)
            ax[row, col].set_ylim(max(cyt[:, 0]) + marge, -marge)
            ax[row, col].scatter(cyt[:, 1], cyt[:, 0], c="black", s=5,
                                 marker=".")
            ax[row, col].scatter(nuc[:, 1], nuc[:, 0], c="steelblue", s=5,
                                 marker=".")
            ax[row, col].scatter(rna[:, 1], rna[:, 0], c="firebrick", s=50,
                                 marker="x")
            if len(foci) > 0:
                ax[row, col].scatter(foci[:, 2], foci[:, 1], c="chartreuse",
                                     s=60, marker="D")
            if title is not None:
                title_plot = title + "_cell_{0}".format(i)
                ax[row, col].set_title(title_plot,
                                       fontweight="bold", fontsize=10)

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
              title=None, remove_frame=False, rescale=False,
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
    # TODO recode it
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
                          title=(str, type(None)),
                          remove_frame=bool,
                          rescale=bool,
                          framesize=tuple,
                          path_output=(str, type(None)),
                          ext=(str, list))
    if title is None:
        title = ""
    else:
        title = " ({0})".format(title)

    # get shape of image built from coordinates
    marge = stack.get_offset_value()
    max_y = cyt_coord[:, 0].max() + 2 * marge + 1
    max_x = cyt_coord[:, 1].max() + 2 * marge + 1
    image_shape = (max_y, max_x)

    # get cytoplasm layer
    cyt = np.zeros(image_shape, dtype=bool)
    cyt[cyt_coord[:, 0] + marge, cyt_coord[:, 1] + marge] = True

    # get nucleus layer
    nuc = np.zeros(image_shape, dtype=bool)
    if nuc_coord is not None:
        nuc[nuc_coord[:, 0] + marge, nuc_coord[:, 1] + marge] = True

    # get rna layer
    rna = np.zeros(image_shape, dtype=bool)
    if rna_coord is not None:
        rna[rna_coord[:, 1] + marge, rna_coord[:, 2] + marge] = True
        rna = stack.dilation_filter(rna,
                                    kernel_shape="square",
                                    kernel_size=3)

    # get foci layer
    foci = np.zeros(image_shape, dtype=bool)
    if foci_coord is not None:
        rna_in_foci_coord = rna_coord[rna_coord[:, 3] != -1, :].copy()
        foci[rna_in_foci_coord[:, 1] + marge, rna_in_foci_coord[:, 2] + marge] = True
        foci = stack.dilation_filter(foci,
                                     kernel_shape="square",
                                     kernel_size=3)

    # build image coordinate
    image_coord = np.ones((max_y, max_x, 3), dtype=np.float32)
    image_coord[cyt, :] = [0, 0, 0]  # black
    image_coord[nuc, :] = [0, 102 / 255, 204 / 255]  # blue
    image_coord[rna, :] = [204 / 255, 0, 0]  # red
    image_coord[foci, :] = [102 / 255, 204 / 255, 0]  # green

    # plot original and coordinate image
    if image_cyt is not None:
        fig, ax = plt.subplots(1, 2, sharex='col', figsize=framesize)

        # original image
        if remove_frame:
            ax[0].axis("off")
        if not rescale:
            vmin, vmax = get_minmax_values(image_cyt)
            ax[0].imshow(image_cyt, vmin=vmin, vmax=vmax)
        else:
            ax[0].imshow(image_cyt)
        if mask_cyt is not None:
            boundaries_cyt = find_boundaries(mask_cyt, mode='inner')
            boundaries_cyt = np.ma.masked_where(boundaries_cyt == 0,
                                                boundaries_cyt)
            ax[0].imshow(boundaries_cyt, cmap=ListedColormap(['red']))
        if mask_nuc is not None:
            boundaries_nuc = find_boundaries(mask_nuc, mode='inner')
            boundaries_nuc = np.ma.masked_where(boundaries_nuc == 0,
                                                boundaries_nuc)
            ax[0].imshow(boundaries_nuc, cmap=ListedColormap(['blue']))
        ax[0].set_title("Original image" + title,
                        fontweight="bold", fontsize=10)

        # coordinate image
        if remove_frame:
            ax[1].axis("off")
        ax[1].imshow(image_coord)
        if count_rna and foci_coord is not None:
            for (_, y, x, nb_rna, _) in foci_coord:
                ax[1].text(x+5, y-5, str(nb_rna), color="#66CC00", size=20)
        ax[1].set_title("Coordinate image" + title,
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
            plt.title("Coordinate image" + title,
                      fontweight="bold", fontsize=25)
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
