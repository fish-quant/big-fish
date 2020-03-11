# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.detection module includes function to detect RNA spot in 2-d and
3-d.
"""

from .spot_detection import log_lm
from .spot_detection import local_maximum_detection
from .spot_detection import spots_thresholding
from .spot_detection import compute_snr
from .spot_detection import from_threshold_to_snr
from .spot_detection import get_sigma
from .spot_detection import log_cc
from .spot_detection import get_cc
from .spot_detection import log_lm
from .spot_detection import log_lm
from .spot_detection import log_lm

from .cluster_decomposition import gaussian_3d
from .cluster_decomposition import precompute_erf
from .cluster_decomposition import build_reference_spot_3d
from .cluster_decomposition import initialize_spot_parameter_3d
from .cluster_decomposition import objective_function
from .cluster_decomposition import fit_gaussian_3d
from .cluster_decomposition import simulate_fitted_gaussian_3d
from .cluster_decomposition import fit_gaussian_mixture
from .cluster_decomposition import filter_clusters
from .cluster_decomposition import decompose_clusters
from .cluster_decomposition import run_decomposition

from .foci_detection import convert_spot_coordinates
from .foci_detection import cluster_spots
from .foci_detection import extract_foci


_spots = [
    "log_lm",
    "local_maximum_detection",
    "spots_thresholding",
    "compute_snr",
    "from_threshold_to_snr",
    "get_sigma",
    "log_cc",
    "get_cc",
    "filter_cc"]

_clusters = [
    "gaussian_3d",
    "precompute_erf",
    "build_reference_spot_3d",
    "initialize_spot_parameter_3d",
    "objective_function",
    "fit_gaussian_3d",
    "simulate_fitted_gaussian_3d",
    "fit_gaussian_mixture",
    "filter_clusters",
    "decompose_clusters",
    "run_decomposition"]

_foci = [
    "convert_spot_coordinates",
    "cluster_spots",
    "extract_foci"]

__all__ = _spots + _clusters + _foci
