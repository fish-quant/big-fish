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

from .spot_modeling import fit_subpixel
from .spot_modeling import build_reference_spot
from .spot_modeling import modelize_spot
from .spot_modeling import precompute_erf
from .spot_modeling import initialize_grid
from .spot_modeling import gaussian_2d
from .spot_modeling import gaussian_3d

from .cluster_detection import detect_clusters

from .snr import compute_snr_spots


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
    "fit_subpixel",
    "build_reference_spot",
    "modelize_spot",
    "precompute_erf",
    "initialize_grid",
    "gaussian_2d",
    "gaussian_3d"]

_clusters = [
    "detect_clusters"]

_snr = [
    "compute_snr_spots"]

__all__ = _spots + _dense + _model + _clusters + _snr
