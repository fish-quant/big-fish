# -*- coding: utf-8 -*-

"""
The bigfish.deep_learning subpackage includes deep learning models and their
routines.
"""

import warnings

# check tensorflow package
try:
    import tensorflow
    tf_version = tensorflow.__version__
    if tf_version != "2.3.0":
        warnings.warn("Available tensorflow version ({0}) differs from "
                      "requirements (2.3.0).".format(tf_version),
                      category=ImportWarning)
except ImportError:
    raise ImportError(
        "Tensorflow package (v2.3.0) is missing. You can install it by "
        "running the command 'pip install tensorflow==2.3.0' in a bash shell.")

# check tensorflow-addons package
try:
    import tensorflow_addons
    tfa_version = tensorflow_addons.__version__
    if tfa_version != "0.12.1":
        warnings.warn("Available tensorflow-addons version ({0}) differs from "
                      "requirements (0.12.1).".format(tfa_version),
                      category=ImportWarning)
except ImportError:
    raise ImportError(
        "Tensorflow addons package (v0.12.1) is missing. You can install it by"
        "running the command 'pip install tensorflow-addons==0.12.1' in a "
        "bash shell.")

from .utils_models import SameConv
from .utils_models import UpConv
from .utils_models import DownBlock
from .utils_models import UpBlock
from .utils_models import Encoder
from .utils_models import Decoder
from .utils_models import EncoderDecoder

from .models_segmentation import load_pretrained_model
from .models_segmentation import check_pretrained_weights
from .models_segmentation import build_compile_3_classes_model
from .models_segmentation import build_compile_distance_model
from .models_segmentation import build_compile_double_distance_model


_utils_models = [
    "SameConv",
    "DownBlock",
    "UpBlock",
    "Encoder",
    "Decoder",
    "EncoderDecoder"]

_models_segmentation = [
    "load_pretrained_model",
    "check_pretrained_weights",
    "build_compile_3_classes_model",
    "build_compile_distance_model",
    "build_compile_double_distance_model"]


__all__ = (_utils_models, _models_segmentation)
