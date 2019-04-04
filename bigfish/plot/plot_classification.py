# -*- coding: utf-8 -*-

"""
Functions to plot results from classification model.
"""
import matplotlib.pyplot as plt
import numpy as np

from .utils import save_plot

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, normalize=False, classes_num=None,
                          classes_str=None, title=None, framesize=(8, 8),
                          path_output=None, ext="png"):
    """

    Parameters
    ----------
    y_true
    y_pred
    normalize
    classes_num
    classes_str
    title
    framesize
    path_output
    ext

    Returns
    -------

    """
    # TODO add documentation
    # compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes_num)

    # normalize confusion matrix
    if normalize:
        cm = cm.astype(np.float32)
        mask = (cm != 0)
        cm = np.divide(cm, cm.sum(axis=1)[:, np.newaxis],
                       out=np.zeros_like(cm),
                       where=mask)

    # plot confusion matrix and colorbar
    fig, ax = plt.subplots(figsize=framesize)
    frame = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap("Blues"))
    colorbar = ax.figure.colorbar(frame, ax=ax, fraction=0.0453, pad=0.05)
    if normalize:
        colorbar.ax.set_ylabel("Density", rotation=-90, va="bottom",
                               fontweight="bold", fontsize=10)
    else:
        colorbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom",
                               fontweight="bold", fontsize=10)
    # cax = divider.append_axes("right", size=width, pad=pad)

    # set ticks
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    if classes_str is not None:
        ax.set_xticklabels(classes_str, rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=10)
        ax.set_yticklabels(classes_str, fontsize=10)
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=20)
    ax.set_xlabel("Predicted label", fontweight="bold", fontsize=15)
    ax.set_ylabel("True label", fontweight="bold", fontsize=15)

    # text annotations in the matrix
    fmt = '.2f' if normalize else 'd'
    threshold = np.nanmax(cm) / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), fontsize=8,
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    save_plot(path_output, ext)
    fig.show()

    return


def plot_2d_projection(x, y, labels_num, labels_str, colors, markers=None,
                       title=None, framesize=(8, 8), path_output=None,
                       ext="png"):
    """

    Parameters
    ----------
    x
    y
    labels_num
    labels_str
    colors
    markers
    title
    framesize
    path_output
    ext

    Returns
    -------

    """
    # TODO add documentation
    # define markers
    if markers is None:
        markers = ["."] * len(labels_str)

    # plot
    plt.figure(figsize=framesize)
    for i, label_num in enumerate(labels_num):
        plt.scatter(x[y == label_num, 0], x[y == label_num, 1],
                    s=30, c=colors[i], label=labels_str[i], marker=markers[i])
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=20)
    plt.xlabel("First component", fontweight="bold", fontsize=15)
    plt.ylabel("Second component", fontweight="bold", fontsize=15)
    plt.legend(prop={'size': 10})
    plt.tight_layout()
    save_plot(path_output, ext)
    plt.show()

    return
