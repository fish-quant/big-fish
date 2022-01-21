# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.multistack.postprocess module.
"""

import pytest

import numpy as np

import bigfish.multistack as multistack

from numpy.testing import assert_array_equal


# TODO add test bigfish.multistack.match_nuc_cell
# TODO add test bigfish.multistack.extract_cell
# TODO add test bigfish.multistack.extract_spots_from_frame
# TODO add test bigfish.multistack.summarize_extraction_results

@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("mask_dtype", [
    np.uint8, np.uint16, np.int64, bool])
@pytest.mark.parametrize("spot_dtype", [
    np.int64, np.float64])
def test_identify_objects_in_region(ndim, mask_dtype, spot_dtype):
    # simulate mask and coordinates
    mask = np.zeros((10, 10), dtype=mask_dtype)
    mask[1:4, 1:5] = np.ones((3, 4), dtype=mask_dtype)
    spots_in = [[2, 1, 1], [5, 2, 2], [2, 3, 1], [9, 3, 4]]
    spots_in = np.array(spots_in, dtype=spot_dtype)
    spots_out = [[1, 0, 0], [3, 7, 2], [2, 1, 8], [7, 8, 8]]
    spots_out = np.array(spots_out, dtype=spot_dtype)
    if ndim == 2:
        spots_in = spots_in[:, 1:]
        spots_out = spots_out[:, 1:]
    spots = np.concatenate((spots_in, spots_out))

    # test
    spots_in_, spots_out_ = multistack.identify_objects_in_region(
        mask, spots, ndim)
    assert_array_equal(spots_in_, spots_in)
    assert_array_equal(spots_out_, spots_out)
    assert spots_in_.dtype == spots_in.dtype
    assert spots_out_.dtype == spots_out.dtype


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("mask_dtype", [
    np.uint8, np.uint16, np.int64, bool])
@pytest.mark.parametrize("spot_dtype", [
    np.int64, np.float64])
def test_remove_transcription_site(ndim, mask_dtype, spot_dtype):
    # simulate mask and coordinates
    nuc_mask = np.zeros((10, 10), dtype=mask_dtype)
    nuc_mask[1:4, 1:5] = np.ones((3, 4), dtype=mask_dtype)

    spots_in_nuc_out_ts = [[2, 1, 1, -1], [5, 2, 2, -1], [2, 3, 1, -1]]
    spots_in_nuc_out_ts = np.array(spots_in_nuc_out_ts, dtype=spot_dtype)

    spots_in_nuc_in_ts = [[2, 3, 3, 3], [5, 2, 4, 2], [2, 1, 3, 3]]
    spots_in_nuc_in_ts = np.array(spots_in_nuc_in_ts, dtype=spot_dtype)

    spots_out_nuc_out_foci = [[1, 0, 0, -1], [3, 7, 2, -1], [2, 1, 8, -1]]
    spots_out_nuc_out_foci = np.array(spots_out_nuc_out_foci, dtype=spot_dtype)

    spots_out_nuc_in_foci = [[1, 0, 4, 0], [3, 7, 7, 0], [2, 1, 5, 1]]
    spots_out_nuc_in_foci = np.array(spots_out_nuc_in_foci, dtype=spot_dtype)

    if ndim == 2:
        spots_in_nuc_out_ts = spots_in_nuc_out_ts[:, 1:]
        spots_in_nuc_in_ts = spots_in_nuc_in_ts[:, 1:]
        spots_out_nuc_out_foci = spots_out_nuc_out_foci[:, 1:]
        spots_out_nuc_in_foci = spots_out_nuc_in_foci[:, 1:]
    spots = np.concatenate((spots_in_nuc_out_ts, spots_in_nuc_in_ts,
                            spots_out_nuc_out_foci, spots_out_nuc_in_foci))
    spots_out_ts = np.concatenate((spots_in_nuc_out_ts,
                                   spots_out_nuc_out_foci,
                                   spots_out_nuc_in_foci))

    ts = [[2, 2, 4, 1, 2], [4, 2, 3, 2, 3]]
    ts = np.array(ts, dtype=spot_dtype)

    foci = [[0, 6, 7, 2, 0], [1, 2, 5, 1, 1]]
    foci = np.array(foci, dtype=spot_dtype)

    if ndim == 2:
        ts = ts[:, 1:]
        foci = foci[:, 1:]
    all_clusters = np.concatenate((ts, foci))

    # test
    spots_out_ts_, foci_, ts_ = multistack.remove_transcription_site(
        spots, all_clusters, nuc_mask, ndim)
    assert_array_equal(spots_out_ts_, spots_out_ts)
    assert spots_out_ts_.dtype == spots_out_ts.dtype
    assert_array_equal(foci_, foci)
    assert foci_.dtype == foci.dtype
    assert_array_equal(ts_, ts)
    assert ts_.dtype == ts.dtype
