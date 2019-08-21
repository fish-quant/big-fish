# -*- coding: utf-8 -*-

"""
The bigfish.segmentation module includes function to segment nucleus,
cytoplasm and label them, in 2-d and 3-d.
"""

from .utils import (label_instances, compute_mean_size_object, merge_labels,
                    get_boundaries, dilate_erode_labels)
from .nuc_segmentation import (filtered_threshold, remove_segmented_nuc)
from .cyt_segmentation import (build_cyt_relief, build_cyt_binary_mask,
                               cyt_watershed)
# from .unet import get_input_size_unet

_nuc = ["filtered_threshold", "remove_segmented_nuc"]

_cyt = ["build_cyt_relief", "build_cyt_binary_mask", cyt_watershed]

# _unet = ["get_input_size_unet"]

_utils = ["label_instances", "compute_mean_size_object", "merge_labels",
          "get_boundaries", "dilate_erode_labels"]

__all__ = _utils + _nuc + _cyt
