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
from .plot_images import plot_detection
from .plot_images import plot_reference_spot
from .plot_images import plot_cell

from .plot_quality import plot_sharpness

from .plot_classification import plot_confusion_matrix
from .plot_classification import plot_2d_projection


_images = [
    "plot_yx",
    "plot_images",
    "plot_segmentation",
    "plot_segmentation_boundary",
    "plot_detection",
    "plot_reference_spot",
    "plot_cell"]

_classification = [
    "plot_confusion_matrix",
    "plot_2d_projection"]

_quality = [
    "plot_sharpness"]

__all__ = _images + _classification + _quality
