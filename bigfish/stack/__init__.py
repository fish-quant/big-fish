# -*- coding: utf-8 -*-

"""
The bigfish.stack module includes function to read data, preprocess them and
build stack of images.
"""

from .loader import read_tif, read_pickle
from .preprocess import (build_stack, check_recipe, build_simulated_dataset,
                         projection, rescale, cast_img_uint8, cast_img_uint16,
                         log_filter, mean_filter, median_filter,
                         maximum_filter, minimum_filter, load_stack,
                         gaussian_filter, build_stacks, cast_img_float32,
                         cast_img_float64, compute_illumination_surface,
                         correct_illumination_surface)
from .utils import check_array, check_features_df


__all__ = ["read_tif",
           "read_pickle",
           "build_simulated_dataset",
           "load_stack",
           "build_stack",
           "build_stacks",
           "check_recipe",
           "projection",
           "rescale",
           "cast_img_uint8",
           "cast_img_uint16",
           "cast_img_float32",
           "cast_img_float64",
           "log_filter",
           "gaussian_filter",
           "mean_filter",
           "median_filter",
           "maximum_filter",
           "minimum_filter",
           "check_array",
           "check_features_df",
           "compute_illumination_surface",
           "correct_illumination_surface"]
