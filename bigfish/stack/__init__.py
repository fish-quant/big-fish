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
                         correct_illumination_surface, clean_simulated_data)
from .preparation import (split_from_background, build_cell_2d,
                          get_coordinates, from_coord_to_image,
                          get_distance_layers, get_surface_layers,
                          build_input_image, resize_image, build_batch,
                          get_label, one_hot_label, Generator,
                          encode_labels, get_map_label)
from .augmentation import augment
from .utils import check_array, check_features_df, check_range_value


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
           "correct_illumination_surface",
           "clean_simulated_data",
           "split_from_background",
           "build_cell_2d",
           "get_coordinates",
           "from_coord_to_image",
           "get_distance_layers",
           "get_surface_layers",
           "build_input_image",
           "check_range_value",
           "resize_image",
           "augment",
           "build_batch",
           "get_label",
           "one_hot_label",
           "Generator",
           "encode_labels",
           "get_map_label"]
