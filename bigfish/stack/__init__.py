# -*- coding: utf-8 -*-

"""
The 'stack' module includes function to read data, preprocess them and build
stack of images.
"""

from .loader import read_tif, read_pickle, build_simulated_dataset


__all__ = ["read_tif",
           "read_pickle",
           "build_simulated_dataset"]
