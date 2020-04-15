# -*- coding: utf-8 -*-

"""
The bigfish.deep_learning module includes deep learning models and routines.
"""

from .squeezenet import SqueezeNet0, SqueezeNet_qbi


_squeezenet = ["SqueezeNet0", "SqueezeNet_qbi"]

__all__ = _squeezenet
