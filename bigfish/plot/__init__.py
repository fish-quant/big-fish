# -*- coding: utf-8 -*-

"""
The bigfish.plot module includes function to plot images and simulated data.
"""

from .plot_images import (plot_yx, plot_channels_2d, plot_segmentation,
                          plot_projection)


__all__ = ["plot_yx",
           "plot_channels_2d",
           "plot_projection",
           "plot_segmentation"]
