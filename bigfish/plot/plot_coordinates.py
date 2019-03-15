# -*- coding: utf-8 -*-

"""
Functions to plot nucleus, cytoplasm and RNA coordinates.
"""

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
    """

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
