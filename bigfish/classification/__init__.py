# -*- coding: utf-8 -*-

"""
The bigfish.classification module includes models to classify the localization
patterns of the RNA.
"""

# from .squeezenet import SqueezeNet0
from .features import get_features, get_features_name

# ### Load models ###

_features = ["get_features", "get_features_name"]

# _squeezenet = ["SqueezeNet0"]

__all__ = _features
