# -*- coding: utf-8 -*-

"""
The bigfish.segmentation module includes function to segment nucleus,
cytoplasm and label them, in 2-d and 3-d.
"""

from .utils import label_instances, compute_mean_size_object, merge_labels
from .nuc_segmentation import (nuc_segmentation_2d, filtered_threshold,
                               remove_segmented_nuc)
from .cyt_segmentation import cyt_segmentation_2d, watershed_2d


_nuc = ["nuc_segmentation_2d", "filtered_threshold", "remove_segmented_nuc"]

_cyt = ["cyt_segmentation_2d", "watershed_2d"]

_utils = ["label_instances", "compute_mean_size_object", "merge_labels"]

__all__ = _utils + _nuc + _cyt
