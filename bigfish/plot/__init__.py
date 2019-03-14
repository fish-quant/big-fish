# -*- coding: utf-8 -*-

"""
The bigfish.plot module includes function to plot images and simulated data.
"""

from .plot_images import (plot_yx, plot_channels_2d, plot_segmentation,
                          plot_projection, plot_images, plot_spot_detection,
                          plot_illumination_surface)


__all__ = ["plot_yx",
           "plot_images",
           "plot_channels_2d",
           "plot_projection",
           "plot_segmentation",
           "plot_spot_detection",
           "plot_illumination_surface"]
