# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.segmentation subpackage includes functions to segment or label
nuclei and cells.
"""

from .utils import label_instances
from .utils import merge_labels
from .utils import thresholding
from .utils import clean_segmentation
from .utils import remove_disjoint
from .utils import compute_mean_diameter
from .utils import compute_mean_convexity_ratio
from .utils import compute_surface_ratio
from .utils import count_instances
from .utils import match_nuc_cell

from .nuc_segmentation import unet_3_classes_nuc
from .nuc_segmentation import remove_segmented_nuc

from .cell_segmentation import unet_3_classes_cell
from .cell_segmentation import unet_distance_edge_cell
from .cell_segmentation import cell_watershed
from .cell_segmentation import get_watershed_relief
from .cell_segmentation import apply_watershed


_utils = [
    "label_instances",
    "merge_labels",
    "thresholding",
    "clean_segmentation",
    "remove_disjoint",
    "compute_mean_diameter",
    "compute_mean_convexity_ratio",
    "compute_surface_ratio",
    "count_instances",
    "match_nuc_cell"]

_nuc = [
    "unet_3_classes_nuc",
    "remove_segmented_nuc"]

_cell = [
    "unet_3_classes_cell",
    "unet_distance_edge_cell",
    "cell_watershed",
    "get_watershed_relief",
    "apply_watershed"]


__all__ = _utils + _nuc + _cell
