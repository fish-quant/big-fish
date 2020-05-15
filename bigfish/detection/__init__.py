# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.detection subpackage includes function to detect RNA spot in 2-d
and 3-d.
"""

from .utils import get_sigma
from .utils import get_radius

from .spot_detection import detect_spots
from .spot_detection import local_maximum_detection
from .spot_detection import spots_thresholding

from .cluster_decomposition import decompose_cluster
from .cluster_decomposition import build_reference_spot
from .cluster_decomposition import modelize_spot
from .cluster_decomposition import precompute_erf
from .cluster_decomposition import get_clustered_region
from .cluster_decomposition import fit_gaussian_mixture

from .foci_detection import detect_foci


_utils = [
    "get_sigma",
    "get_radius"]

_spots = [
    "detect_spots",
    "local_maximum_detection",
    "spots_thresholding"]

_clusters = [
    "decompose_cluster",
    "build_reference_spot",
    "modelize_spot",
    "precompute_erf",
    "get_clustered_region",
    "fit_gaussian_mixture"]

_foci = [
    "detect_foci"]

__all__ = _utils + _spots + _clusters + _foci
