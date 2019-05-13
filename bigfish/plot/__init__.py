# -*- coding: utf-8 -*-

"""
The bigfish.plot module includes function to plot images and simulated data.
"""

from .plot_images import (plot_yx, plot_channels_2d, plot_segmentation,
                          plot_images, plot_spot_detection, plot_boundaries,
                          plot_illumination_surface)
from .plot_coordinates import (plot_volume, plot_rna, plot_distribution_rna,
                               plot_cell_coordinates, plot_layers_coordinates)
from .plot_classification import plot_confusion_matrix, plot_2d_projection


_images = ["plot_yx", "plot_images", "plot_channels_2d", "plot_boundaries",
           "plot_illumination_surface", "plot_segmentation",
           "plot_spot_detection"]

_coordinates = ["plot_volume", "plot_rna", "plot_distribution_rna",
                "plot_cell_coordinates", "plot_layers_coordinates"]

_classification = ["plot_confusion_matrix", "plot_2d_projection"]

__all__ = _images + _coordinates + _classification
