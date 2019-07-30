# -*- coding: utf-8 -*-

"""
The bigfish.detection module includes function to detect RNA spot in 2-d and
3-d.
"""

from .spot_detection import (
    log_lm, local_maximum_detection, spots_thresholding, compute_snr,
    from_threshold_to_snr, get_sigma, log_cc, get_cc)
from .cluster_decomposition import (
    gaussian_3d, precompute_erf, build_reference_spot_3d,
    initialize_spot_parameter_3d, objective_function, fit_gaussian_3d,
    simulate_fitted_gaussian_3d, fit_gaussian_mixture, filter_clusters,
    decompose_clusters, run_decomposition)
from .foci_detection import (
    convert_spot_coordinates, cluster_spots, extract_foci)


_spots = ["log_lm", "local_maximum_detection", "spots_thresholding",
          "compute_snr", "from_threshold_to_snr", "get_sigma", "log_cc",
          "get_cc", "filter_cc"]

_clusters = ["gaussian_3d", "precompute_erf", "build_reference_spot_3d",
             "initialize_spot_parameter_3d", "objective_function",
             "fit_gaussian_3d", "simulate_fitted_gaussian_3d",
             "fit_gaussian_mixture", "filter_clusters", "decompose_clusters",
             "run_decomposition"]

_foci = ["convert_spot_coordinates", "cluster_spots", "extract_foci"]

__all__ = _spots + _clusters + _foci
