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
from .utils import get_margin_value
from .utils import get_eps_float32
from .utils import load_and_save_url
from .utils import check_hash
from .utils import compute_hash
from .utils import check_input_data
from .utils import moving_average
from .utils import centered_moving_average

from .io import read_image
from .io import read_dv
from .io import read_array
from .io import read_uncompressed
from .io import read_cell_extracted
from .io import read_array_from_csv
from .io import read_dataframe_from_csv
from .io import save_image
from .io import save_array
from .io import save_cell_extracted
from .io import save_data_to_csv

from .preprocess import rescale
from .preprocess import cast_img_uint8
from .preprocess import cast_img_uint16
from .preprocess import cast_img_float32
from .preprocess import cast_img_float64
from .preprocess import resize_image
from .preprocess import get_marge_padding
from .preprocess import compute_image_standardization

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
from .projection import in_focus_selection
from .projection import get_in_focus_indices

from .augmentation import augment_2d
from .augmentation import augment_2d_function
from .augmentation import augment_8_times
from .augmentation import augment_8_times_reversed

from .quality import compute_focus


_utils = [
    "check_array",
    "check_df",
    "check_parameter",
    "check_range_value",
    "get_margin_value",
    "get_eps_float32",
    "load_and_save_url",
    "check_hash",
    "compute_hash",
    "check_input_data",
    "moving_average",
    "centered_moving_average"]

_io = [
    "read_image",
    "read_dv",
    "read_array",
    "read_uncompressed",
    "read_cell_extracted",
    "read_array_from_csv",
    "read_dataframe_from_csv",
    "save_image",
    "save_array",
    "save_cell_extracted",
    "save_data_to_csv"]

_preprocess = [
    "rescale",
    "cast_img_uint8",
    "cast_img_uint16",
    "cast_img_float32",
    "cast_img_float64",
    "resize_image",
    "get_marge_padding",
    "compute_image_standardization"]

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
    "get_in_focus_indices",
    "focus_projection"]

_augmentation = [
    "augment_2d",
    "augment_2d_function",
    "augment_8_times",
    "augment_8_times_reversed"]

_quality = [
    "compute_focus"]


__all__ = (_utils + _io + _preprocess + _filter + _projection + _augmentation
           + _quality)
