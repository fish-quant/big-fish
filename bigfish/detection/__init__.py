# -*- coding: utf-8 -*-

"""
The bigfish.detection module includes function to detect RNA spot in 2-d and
3-d.
"""

from .spot_detection import (log_lm, local_maximum_detection,
                             spots_thresholding, compute_snr,
                             from_threshold_to_snr, get_sigma,
                             build_reference_spot)
from .gaussian_fit import (get_spot_volume, get_spot_surface, build_grid,
                           compute_background_amplitude, get_spot_parameter,
                           objective_function, fit_gaussian,
                           simulate_fitted_gaussian, gaussian_3d)

_detection = ["log_lm", "local_maximum_detection", "spots_thresholding",
              "compute_snr", "from_threshold_to_snr", "get_sigma",
              "build_reference_spot"]

_fit = ["get_spot_volume", "get_spot_surface", "build_grid",
        "compute_background_amplitude", "get_spot_parameter",
        "objective_function", "fit_gaussian",
        "simulate_fitted_gaussian", "gaussian_3d"]

__all__ = _detection + _fit
