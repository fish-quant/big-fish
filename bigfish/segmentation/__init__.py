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
from .utils import compute_instances_mean_diameter
from .utils import match_nuc_cell

from .nuc_segmentation import remove_segmented_nuc

from .cell_segmentation import cell_watershed
from .cell_segmentation import get_watershed_relief
from .cell_segmentation import apply_watershed


_utils = [
    "label_instances",
    "merge_labels",
    "thresholding",
    "clean_segmentation",
    "compute_instances_mean_diameter",
    "match_nuc_cell"]

_nuc = [
    "remove_segmented_nuc"]

_cyt = [
    "cell_watershed",
    "get_watershed_relief",
    "apply_watershed"]


__all__ = _utils + _nuc + _cyt
