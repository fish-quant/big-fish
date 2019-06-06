# -*- coding: utf-8 -*-

"""
The bigfish.stack module includes function to read data, preprocess them and
build stack of images.
"""

from .utils import (check_array, check_df, check_recipe, check_parameter,
                    check_range_value, complete_coordinates_2d,
                    from_coord_to_image, get_offset_value)
from .io import (read_image, read_pickle, read_cell_json, read_rna_json,
                 save_image)
from .preprocess import (build_simulated_dataset, build_stacks, build_stack,
                         build_stack_no_recipe, rescale,
                         cast_img_uint8, cast_img_uint16, cast_img_float32,
                         cast_img_float64, clean_simulated_data,
                         deconstruct_image, reconstruct_image)
from .filter import (log_filter, mean_filter, median_filter, maximum_filter,
                     minimum_filter, gaussian_filter, remove_background_mean,
                     remove_background_gaussian)
from .projection import (maximum_projection, mean_projection,
                         median_projection, in_focus_selection,
                         focus_measurement, get_in_focus_indices,
                         focus_projection, focus_projection_fast)
from .illumination import (compute_illumination_surface,
                           correct_illumination_surface)
from .preparation import (split_from_background, build_image, get_coordinates,
                          get_distance_layers, get_surface_layers, build_batch,
                          get_label, Generator, encode_labels, get_map_label,
                          format_experimental_data, get_label_encoder,
                          remove_transcription_site, filter_data, balance_data,
                          get_gene_encoder)
from .augmentation import augment


_utils = ["check_array", "check_df", "check_recipe", "check_parameter",
          "check_range_value", "complete_coordinates_2d",
          "from_coord_to_image", "get_offset_value"]

_io = ["read_image", "read_pickle", "read_cell_json", "read_rna_json",
       "save_image"]

_preprocess = ["build_simulated_dataset", "build_stacks", "build_stack",
               "build_stack_no_recipe", "rescale",
               "cast_img_uint8", "cast_img_uint16", "cast_img_float32",
               "cast_img_float64", "clean_simulated_data", "deconstruct_image",
               "reconstruct_image"]

_filter = ["log_filter", "mean_filter", "median_filter", "maximum_filter",
           "minimum_filter", "gaussian_filter", "remove_background_mean",
           "remove_background_gaussian"]

_projection = ["maximum_projection", "mean_projection", "median_projection",
               "in_focus_selection", "focus_measurement",
               "get_in_focus_indices", "focus_projection",
               "focus_projection_fast"]

_illumination = ["compute_illumination_surface",
                 "correct_illumination_surface"]

_augmentation = ["augment"]

_preparation = ["split_from_background", "build_image", "get_coordinates",
                "get_distance_layers", "get_surface_layers", "build_batch",
                "get_label", "Generator", "encode_labels", "get_map_label",
                "format_experimental_data", "get_label_encoder",
                "remove_transcription_site", "filter_data", "balance_data",
                "get_gene_encoder"]

__all__ = (_utils + _io + _preprocess +
           _filter + _projection + _illumination +
           _augmentation + _preparation)
