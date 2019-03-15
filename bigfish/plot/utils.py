# -*- coding: utf-8 -*-

"""
Utility functions for bigfish.plot submodule.
"""

import matplotlib.pyplot as plt


def save_plot(path_output, ext):
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

    return
