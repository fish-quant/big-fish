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

from .cluster_decomposition import decompose_cluster
from .cluster_decomposition import fit_subpixel
from .cluster_decomposition import build_reference_spot
from .cluster_decomposition import modelize_spot
from .cluster_decomposition import precompute_erf
from .cluster_decomposition import get_clustered_region
from .cluster_decomposition import fit_gaussian_mixture

from .foci_detection import detect_foci


_spots = [
    "detect_spots",
    "local_maximum_detection",
    "spots_thresholding",
    "automated_threshold_setting"]

_clusters = [
    "decompose_cluster",
    "fit_subpixel",
    "build_reference_spot",
    "modelize_spot",
    "precompute_erf",
    "get_clustered_region",
    "fit_gaussian_mixture"]

_foci = [
    "detect_foci"]

__all__ = _spots + _clusters + _foci
