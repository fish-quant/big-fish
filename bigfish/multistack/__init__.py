# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The bigfish.multistack subpackage includes function to process input and output
from different channels.
"""

from .utils import check_recipe
from .utils import fit_recipe
from .utils import get_path_from_recipe
from .utils import get_nb_element_per_dimension
from .utils import count_nb_fov
from .utils import check_datamap

from .preprocess import build_stacks
from .preprocess import build_stack
from .preprocess import build_stack_no_recipe

from .colocalization import detect_spots_colocalization
from .colocalization import get_elbow_value_colocalized

from .postprocess import identify_objects_in_region
from .postprocess import remove_transcription_site
from .postprocess import match_nuc_cell
from .postprocess import extract_cell
from .postprocess import extract_spots_from_frame
from .postprocess import summarize_extraction_results
from .postprocess import center_mask_coord
from .postprocess import from_boundaries_to_surface
from .postprocess import from_surface_to_boundaries
from .postprocess import from_binary_to_coord
from .postprocess import complete_coord_boundaries
from .postprocess import from_coord_to_frame
from .postprocess import from_coord_to_surface


# TODO complete bigfish.multistack.preprocess documentation

_utils = [
    "check_recipe",
    "fit_recipe",
    "get_path_from_recipe",
    "get_nb_element_per_dimension",
    "count_nb_fov",
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
    "summarize_extraction_results",
    "center_mask_coord",
    "from_boundaries_to_surface",
    "from_surface_to_boundaries",
    "from_binary_to_coord",
    "complete_coord_boundaries",
    "from_coord_to_frame",
    "from_coord_to_surface"]

__all__ = _utils + _preprocess + _colocalization + _postprocess
