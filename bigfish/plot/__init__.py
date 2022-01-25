# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.plot subpackage includes functions to plot images and results.
"""

from .plot_images import plot_yx
from .plot_images import plot_images
from .plot_images import plot_segmentation
from .plot_images import plot_segmentation_boundary
from .plot_images import plot_segmentation_diff

from .plot_images import plot_detection
from .plot_images import plot_reference_spot
from .plot_images import plot_cell

from .plot_quality import plot_sharpness
from .plot_quality import plot_elbow
from .plot_quality import plot_elbow_colocalized

from .utils import save_plot
from .utils import get_minmax_values
from .utils import create_colormap


_images = [
    "plot_yx",
    "plot_images",
    "plot_segmentation",
    "plot_segmentation_boundary",
    "plot_segmentation_diff",
    "plot_detection",
    "plot_reference_spot",
    "plot_cell"]

_quality = [
    "plot_sharpness",
    "plot_elbow",
    "plot_elbow_colocalized"]

_utils = [
    "save_plot",
    "get_minmax_values",
    "create_colormap"]

__all__ = _images + _quality + _utils
