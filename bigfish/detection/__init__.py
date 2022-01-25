# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.detection subpackage includes function to detect RNA spot in 2-d
and 3-d.
"""

from .spot_detection import detect_spots
from .spot_detection import local_maximum_detection
from .spot_detection import spots_thresholding
from .spot_detection import automated_threshold_setting
from .spot_detection import get_elbow_values

from .dense_decomposition import decompose_dense
from .dense_decomposition import get_dense_region
from .dense_decomposition import simulate_gaussian_mixture

from .spot_modeling import modelize_spot
from .spot_modeling import initialize_grid
from .spot_modeling import gaussian_2d
from .spot_modeling import gaussian_3d
from .spot_modeling import precompute_erf
from .spot_modeling import fit_subpixel

from .cluster_detection import detect_clusters

from .utils import convert_spot_coordinates
from .utils import get_object_radius_pixel
from .utils import get_object_radius_nm
from .utils import build_reference_spot
from .utils import compute_snr_spots
from .utils import get_breaking_point


_spots = [
    "detect_spots",
    "local_maximum_detection",
    "spots_thresholding",
    "automated_threshold_setting",
    "get_elbow_values"]

_dense = [
    "decompose_dense",
    "get_dense_region",
    "simulate_gaussian_mixture"]

_model = [
    "modelize_spot",
    "initialize_grid",
    "gaussian_2d",
    "gaussian_3d",
    "precompute_erf",
    "fit_subpixel"]

_clusters = [
    "detect_clusters"]

_utils = [
    "convert_spot_coordinates",
    "get_object_radius_pixel",
    "get_object_radius_nm",
    "build_reference_spot",
    "compute_snr_spots",
    "get_breaking_point"]

__all__ = _spots + _dense + _model + _clusters + _utils
