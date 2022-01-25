# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.segmentation subpackage includes functions to segment or label
nuclei and cells.
"""

from .cell_segmentation import unet_distance_edge_double
from .cell_segmentation import apply_unet_distance_double
from .cell_segmentation import from_distance_to_instances
from .cell_segmentation import cell_watershed
from .cell_segmentation import get_watershed_relief
from .cell_segmentation import apply_watershed

from .nuc_segmentation import unet_3_classes_nuc
from .nuc_segmentation import apply_unet_3_classes
from .nuc_segmentation import from_3_classes_to_instances
from .nuc_segmentation import remove_segmented_nuc

from .postprocess import label_instances
from .postprocess import merge_labels
from .postprocess import clean_segmentation
from .postprocess import remove_disjoint

from .utils import thresholding
from .utils import compute_mean_diameter
from .utils import compute_mean_convexity_ratio
from .utils import compute_surface_ratio
from .utils import count_instances


_cell = [
    "unet_distance_edge_double",
    "apply_unet_distance_double",
    "from_distance_to_instances",
    "cell_watershed",
    "get_watershed_relief",
    "apply_watershed"]

_nuc = [
    "unet_3_classes_nuc",
    "apply_unet_3_classes",
    "from_3_classes_to_instances",
    "remove_segmented_nuc"]

_postprocess = [
    "label_instances",
    "merge_labels",
    "clean_segmentation",
    "remove_disjoint"]

_utils = [
    "thresholding",
    "compute_mean_diameter",
    "compute_mean_convexity_ratio",
    "compute_surface_ratio",
    "count_instances"]

__all__ = _cell + _nuc + _postprocess + _utils
