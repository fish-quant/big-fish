# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.stack subpackage includes functions to read data, preprocess them
and build stack of images.
"""

from .utils import check_array
from .utils import check_df
from .utils import check_parameter
from .utils import check_range_value
from .utils import check_recipe
from .utils import check_datamap
from .utils import get_margin_value
from .utils import get_eps_float32
from .utils import load_and_save_url
from .utils import check_hash
from .utils import compute_hash

from .io import read_image
from .io import read_dv
from .io import read_array
from .io import read_uncompressed
from .io import read_cell_extracted
from .io import read_array_from_csv
from .io import save_image
from .io import save_array
from .io import save_cell_extracted
from .io import save_array_to_csv

from .preprocess import build_stacks
from .preprocess import build_stack
from .preprocess import build_stack_no_recipe
from .preprocess import rescale
from .preprocess import cast_img_uint8
from .preprocess import cast_img_uint16
from .preprocess import cast_img_float32
from .preprocess import cast_img_float64

from .filter import mean_filter
from .filter import median_filter
from .filter import maximum_filter
from .filter import minimum_filter
from .filter import log_filter
from .filter import gaussian_filter
from .filter import remove_background_mean
from .filter import remove_background_gaussian
from .filter import dilation_filter
from .filter import erosion_filter

from .projection import maximum_projection
from .projection import mean_projection
from .projection import median_projection
from .projection import focus_projection
from .projection import focus_projection_fast
from .projection import focus_measurement
from .projection import in_focus_selection
from .projection import get_in_focus_indices

from .illumination import compute_illumination_surface
from .illumination import correct_illumination_surface

from .postprocess import identify_transcription_site
from .postprocess import remove_transcription_site
from .postprocess import extract_cell
from .postprocess import extract_spots_from_frame
from .postprocess import center_mask_coord
from .postprocess import from_boundaries_to_surface
from .postprocess import from_surface_to_boundaries
from .postprocess import from_binary_to_coord
from .postprocess import complete_coord_boundaries
from .postprocess import from_coord_to_frame
from .postprocess import from_coord_to_surface

from .augmentation import augment_2d


_utils = [
    "check_array",
    "check_df",
    "check_recipe",
    "check_datamap",
    "check_parameter",
    "check_range_value",
    "get_margin_value",
    "get_eps_float32",
    "load_and_save_url",
    "check_hash",
    "compute_hash"]

_io = [
    "read_image",
    "read_dv",
    "read_array",
    "read_uncompressed",
    "read_cell_extracted",
    "read_array_from_csv",
    "save_image",
    "save_array",
    "save_cell_extracted",
    "save_array_to_csv"]

_preprocess = [
    "build_stacks",
    "build_stack",
    "build_stack_no_recipe",
    "rescale",
    "cast_img_uint8",
    "cast_img_uint16",
    "cast_img_float32",
    "cast_img_float64"]

_filter = [
    "log_filter",
    "mean_filter",
    "median_filter",
    "maximum_filter",
    "minimum_filter",
    "gaussian_filter",
    "remove_background_mean",
    "remove_background_gaussian",
    "dilation_filter",
    "erosion_filter"]

_projection = [
    "maximum_projection",
    "mean_projection",
    "median_projection",
    "in_focus_selection",
    "focus_measurement",
    "get_in_focus_indices",
    "focus_projection",
    "focus_projection_fast"]

_illumination = [
    "compute_illumination_surface",
    "correct_illumination_surface"]

_postprocess = [
    "identify_transcription_site",
    "remove_transcription_site",
    "extract_cell",
    "extract_spots_from_frame",
    "center_mask_coord",
    "from_boundaries_to_surface",
    "from_surface_to_boundaries",
    "from_binary_to_coord",
    "complete_coord_boundaries",
    "from_coord_to_frame",
    "from_coord_to_surface"]

_augmentation = [
    "augment_2d"]


__all__ = (_utils + _io + _preprocess + _postprocess + _filter + _projection +
           _illumination + _augmentation)
