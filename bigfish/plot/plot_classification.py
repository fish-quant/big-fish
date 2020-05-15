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
                          size_title=20, size_axes=15, path_output=None,
                          ext="png"):
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

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=framesize)
    frame = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap("Blues"))

    # colorbar
    colorbar = ax.figure.colorbar(frame, ax=ax, fraction=0.0453, pad=0.05)
    if normalize:
        colorbar.ax.set_ylabel("Density", rotation=-90, va="bottom",
                               fontweight="bold", fontsize=size_axes-5)
    else:
        colorbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom",
                               fontweight="bold", fontsize=size_axes-5)
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
                           rotation_mode="anchor", fontsize=size_axes-5)
        ax.set_yticklabels(classes_str, fontsize=size_axes-5)

    # title and axes labels
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=size_title)
    ax.set_xlabel("Predicted label", fontweight="bold", fontsize=size_axes)
    ax.set_ylabel("True label", fontweight="bold", fontsize=size_axes)

    # text annotations in the matrix
    fmt = '.2f' if normalize else 'd'
    threshold = np.nanmax(cm) / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), fontsize=size_axes-7,
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    # show frame
    fig.tight_layout()
    save_plot(path_output, ext)
    fig.show()

    return


def plot_2d_projection(x, y, labels_num, labels_str, colors, markers=None,
                       title=None, framesize=(10, 10), size_data=50, alpha=0.8,
                       size_title=20, size_axes=15, size_legend=15,
                       path_output=None, ext="png"):
    # TODO add documentation
    # define markers
    if markers is None:
        markers = ["."] * len(labels_str)

    # plot
    plt.figure(figsize=framesize)
    for i, label_num in enumerate(labels_num):
        plt.scatter(x[y == label_num, 0], x[y == label_num, 1],
                    s=size_data, c=colors[i], label=labels_str[i],
                    marker=markers[i], alpha=alpha)

    # text annotations
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=size_title)
    plt.xlabel("First component", fontweight="bold", fontsize=size_axes)
    plt.ylabel("Second component", fontweight="bold", fontsize=size_axes)
    plt.legend(prop={'size': size_legend})

    # show frame
    plt.tight_layout()
    save_plot(path_output, ext)
    plt.show()

    return
