# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.postprocess module.
"""

import pytest

import numpy as np
import bigfish.stack as stack

from numpy.testing import assert_array_equal

# TODO add test bigfish.stack.extract_cell
# TODO add test bigfish.stack.extract_spots_from_frame
# TODO add test bigfish.stack.center_mask_coord
# TODO add test bigfish.stack.from_boundaries_to_surface
# TODO add test bigfish.stack.from_surface_to_boundaries
# TODO add test bigfish.stack.from_binary_to_coord
# TODO add test bigfish.stack.complete_coord_boundaries
# TODO add test bigfish.stack.from_coord_to_frame
# TODO add test bigfish.stack.from_coord_to_surface


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("mask_dtype", [
    np.uint8, np.uint16, np.int64, bool])
def test_identify_objects_in_region(ndim, mask_dtype):
    # simulate mask and coordinates
    mask = np.zeros((10, 10), dtype=mask_dtype)
    mask[1:4, 1:5] = np.ones((3, 4), dtype=mask_dtype)
    spots_in = [[2, 1, 1], [5, 2, 2], [2, 3, 1], [9, 3, 4]]
    spots_in = np.array(spots_in, dtype=np.int64)
    spots_out = [[1, 0, 0], [3, 7, 2], [2, 1, 8], [7, 8, 8]]
    spots_out = np.array(spots_out, dtype=np.int64)
    if ndim == 2:
        spots_in = spots_in[:, 1:]
        spots_out = spots_out[:, 1:]
    spots = np.concatenate((spots_in, spots_out))

    # test
    spots_in_, spots_out_ = stack.identify_objects_in_region(mask, spots, ndim)
    assert_array_equal(spots_in_, spots_in)
    assert_array_equal(spots_out_, spots_out)
    assert spots_in_.dtype == spots_in.dtype
    assert spots_out_.dtype == spots_out.dtype


@pytest.mark.parametrize("ndim", [2, 3])
def test_remove_transcription_site(ndim):
    # simulate coordinates or rna and transcription sites
    ts = [[0, 0, 0, 5, 0], [0, 0, 0, 5, 1]]
    ts = np.array(ts, dtype=np.int64)
    rna_out = [[0, 0, 0, -1], [0, 0, 0, -1], [0, 0, 0, -1],
               [0, 0, 0, -1], [0, 0, 0, 3]]
    rna_out = np.array(rna_out, dtype=np.int64)
    rna_in = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
    rna_in = np.array(rna_in, dtype=np.int64)
    if ndim == 2:
        ts = ts[:, 1:]
        rna_in = rna_in[:, 1:]
        rna_out = rna_out[:, 1:]
    rna = np.concatenate((rna_in, rna_out))

    # test
    rna_out_ = stack.remove_transcription_site_rna(rna, ts)
    assert_array_equal(rna_out_, rna_out)
    assert rna_out_.dtype == rna_out.dtype
