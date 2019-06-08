# -*- coding: utf-8 -*-

"""
The bigfish.detection module includes function to detect RNA spot in 2-d and
3-d.
"""

from .spot_detection import (log_lm, local_maximum_detection,
                             spots_thresholding, compute_snr,
                             from_threshold_to_snr, get_sigma, log_cc, get_cc,
                             filter_cc)
from .gaussian_fit import (gaussian_3d, build_reference_spot_3d,
                           get_spot_volume, get_spot_surface, precompute_erf,
                           initialize_spot_parameter_3d, objective_function,
                           fit_gaussian_3d, simulate_fitted_gaussian_3d,
                           initialize_grid_3d, compute_background_amplitude,
                           fit_gaussian_mixture, foci_decomposition)

_detection = ["log_lm", "local_maximum_detection", "spots_thresholding",
              "compute_snr", "from_threshold_to_snr", "get_sigma", "log_cc",
              "get_cc", "filter_cc"]

_fit = ["gaussian_3d", "precompute_erf", "build_reference_spot_3d",
        "get_spot_volume", "get_spot_surface", "initialize_spot_parameter_3d",
        "objective_function", "fit_gaussian_3d", "simulate_fitted_gaussian_3d",
        "initialize_grid_3d", "compute_background_amplitude",
        "fit_gaussian_mixture", "foci_decomposition"]

__all__ = _detection + _fit
