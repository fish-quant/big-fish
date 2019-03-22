# -*- coding: utf-8 -*-

"""
Functions to plot nucleus, cytoplasm and RNA coordinates.
"""
import bigfish.stack as stack

import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot


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
    cyt, nuc, rna = stack.get_coordinates(data, id_cell)

    # plot
    plt.figure(figsize=framesize)
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=25)
    plt.plot(cyt[:, 1], cyt[:, 0], c="black", linewidth=2)
    plt.plot(nuc[:, 1], nuc[:, 0], c="steelblue", linewidth=2)
    plt.scatter(rna[:, 1], rna[:, 0], s=25, c="firebrick", marker=".")
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
