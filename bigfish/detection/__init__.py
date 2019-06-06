# -*- coding: utf-8 -*-

"""
The bigfish.detection module includes function to detect RNA spot in 2-d and
3-d.
"""

from .spot_detection import (log_lm, local_maximum_detection,
                             spots_thresholding, compute_snr,
                             from_threshold_to_snr, get_sigma)
from .gaussian_fit import (gaussian_3d, precompute_erf, build_reference_spot,
                           get_spot_volume, get_spot_surface,
                           initialize_spot_parameter_3d, objective_function,
                           fit_gaussian_3d, simulate_fitted_gaussian_3d,
                           initialize_grid_3d)

_detection = ["log_lm", "local_maximum_detection", "spots_thresholding",
              "compute_snr", "from_threshold_to_snr", "get_sigma"]

_fit = ["gaussian_3d", "precompute_erf", "build_reference_spot",
        "get_spot_volume", "get_spot_surface", "initialize_spot_parameter_3d",
        "objective_function", "fit_gaussian_3d", "simulate_fitted_gaussian_3d",
        "initialize_grid_3d"]

__all__ = _detection + _fit
