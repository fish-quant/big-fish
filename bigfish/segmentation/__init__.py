# -*- coding: utf-8 -*-

"""
The bigfish.segmentation module includes function to segment nucleus,
cytoplasm and label them, in 2-d and 3-d.
"""

from .utils import label_instances
from .nuc_segmentation import nuc_segmentation_2d, filtered_threshold
from .cyt_segmentation import cyt_segmentation_2d, watershed_2d


_nuc = ["nuc_segmentation_2d", "filtered_threshold"]

_cyt = ["cyt_segmentation_2d", "watershed_2d"]

_utils = ["label_instances"]

__all__ = _utils + _nuc + _cyt
