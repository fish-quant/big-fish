# -*- coding: utf-8 -*-

"""
The bigfish.detection module includes function to detect RNA spot in 2-d and
3-d.
"""

from .detection import (detection, compute_snr, get_sigma, detection_log_lm,
                        detection_log_lm, log_lm, non_maximum_suppression_mask,
                        from_threshold_to_spots, from_threshold_to_snr)
from .gaussian_fit import (get_spot_volume, get_spot_surface, build_grid,
                           compute_background_amplitude, get_spot_parameter,
                           objective_function, fit_gaussian,
                           simulate_fitted_gaussian, gaussian_3d)

_detection = ["detection", "compute_snr", "get_sigma", "detection_log_lm",
              "detection_log_lm", "log_lm", "non_maximum_suppression_mask",
              "from_threshold_to_spots", "from_threshold_to_snr"]

_fit = ["get_spot_volume", "get_spot_surface", "build_grid",
        "compute_background_amplitude", "get_spot_parameter",
        "objective_function", "fit_gaussian",
        "simulate_fitted_gaussian", "gaussian_3d"]

__all__ = _detection + _fit
