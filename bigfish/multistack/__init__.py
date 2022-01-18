# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.multistack subpackage includes function to process input and output
from different channels.
"""

from .utils import check_recipe
from .utils import check_datamap

from .preprocess import build_stacks
from .preprocess import build_stack
from .preprocess import build_stack_no_recipe

from .colocalization import detect_spots_colocalization
from .colocalization import get_elbow_value_colocalized

from .cell_extraction import identify_objects_in_region
from .cell_extraction import remove_transcription_site
from .cell_extraction import match_nuc_cell
from .cell_extraction import extract_cell
from .cell_extraction import extract_spots_from_frame
from .cell_extraction import summarize_extraction_results

_utils = [
    "check_recipe",
    "check_datamap"]

_preprocess = [
    "build_stacks",
    "build_stack",
    "build_stack_no_recipe"]

_colocalization = [
    "detect_spots_colocalization",
    "get_elbow_value_colocalized"]

_postprocess = [
    "identify_objects_in_region",
    "remove_transcription_site",
    "match_nuc_cell",
    "extract_cell",
    "extract_spots_from_frame",
    "summarize_extraction_results"]

__all__ = _utils + _preprocess + _colocalization + _postprocess
