# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.projection module.
"""

import pytest

import numpy as np
import bigfish.stack as stack

from bigfish.stack.projection import _one_hot_3d

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


# toy images
x_3d = np.array(
    [[[1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0]],

     [[2, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0]],

     [[3, 0, 0, 0, 0],
      [0, 3, 0, 0, 3],
      [0, 3, 0, 0, 0],
      [0, 3, 3, 3, 0],
      [0, 0, 0, 0, 0]]],
    dtype=np.uint8)

x_3d_out_focus = np.array(
    [[[1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1]],

     [[1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0]],

     [[1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1]],

     [[2, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0]],

     [[3, 0, 0, 0, 0],
      [0, 3, 0, 0, 3],
      [0, 3, 0, 0, 0],
      [0, 3, 3, 3, 0],
      [0, 0, 0, 0, 0]],

     [[1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1]]],
    dtype=np.uint8)


# ### test 2-d projection ###

@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.int32, np.int64, np.float32, np.float64])
def test_maximum_projection(dtype):
    x = x_3d.astype(dtype)
    expected_y = np.array([[3, 0, 0, 0, 0],
                           [0, 3, 0, 0, 3],
                           [0, 3, 0, 0, 0],
                           [0, 3, 3, 3, 0],
                           [0, 0, 0, 0, 0]], dtype=dtype)
    y = stack.maximum_projection(x)
    assert_array_equal(y, expected_y)
    assert y.dtype == dtype


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.int32, np.int64, np.float32, np.float64])
def test_mean_projection(dtype):
    # 'return_float' == False
    x = x_3d.astype(dtype)
    expected_y = np.array([[2,   0,   0,   0,   0],
                           [0,   5/3, 0,   0,   1],
                           [0,   5/3, 0,   0,   0],
                           [0,   5/3, 5/3, 5/3, 0],
                           [0,   0,   0,   0,   0]], dtype=dtype)
    y = stack.mean_projection(x)
    assert_array_equal(y, expected_y)
    assert y.dtype == dtype

    # 'return_float' == True
    expected_y = np.array([[2, 0,   0,   0,   0],
                           [0, 5/3, 0,   0,   1],
                           [0, 5/3, 0,   0,   0],
                           [0, 5/3, 5/3, 5/3, 0],
                           [0, 0,   0,   0,   0]], dtype=np.float64)
    y = stack.mean_projection(x, return_float=True)
    assert_array_almost_equal(y, expected_y)
    assert y.dtype in [np.float32, np.float64]


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.int32, np.int64, np.float32, np.float64])
def test_median_projection(dtype):
    x = x_3d.astype(dtype)
    expected_y = np.array([[2, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=dtype)
    y = stack.median_projection(x)
    assert_array_equal(y, expected_y)
    assert y.dtype == dtype


# ### test focus selection ###

@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64])
def test_one_hot_3d(dtype):
    # integer
    indices = np.array(
        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [2, 1, 0, 2, 2],
         [0, 1, 1, 1, 2],
         [0, 0, 0, 1, 0]], dtype=dtype)
    one_hot = _one_hot_3d(indices=indices, depth=4)
    expected_one_hot = np.array(
        [[[0, 1, 1, 1, 1],
          [1, 0, 1, 1, 1],
          [0, 0, 1, 0, 0],
          [1, 0, 0, 0, 0],
          [1, 1, 1, 0, 1]],

         [[1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 1, 1, 1, 0],
          [0, 0, 0, 1, 0]],

         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 0, 0, 1, 1],
          [0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0]],

         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]],
        dtype=np.uint8)
    assert_array_equal(one_hot, expected_one_hot)
    assert one_hot.dtype == indices.dtype

    # boolean
    one_hot = _one_hot_3d(indices=indices, depth=4, return_boolean=True)
    expected_one_hot = expected_one_hot.astype(bool)
    assert_array_equal(one_hot, expected_one_hot)
    assert one_hot.dtype == bool


def test_get_in_focus_indices():
    focus = stack.compute_focus(x_3d_out_focus, neighborhood_size=31)

    # number of slices to keep
    indices_to_keep = stack.get_in_focus_indices(focus, proportion=3)
    assert isinstance(indices_to_keep, list)
    assert len(indices_to_keep) == 3
    assert indices_to_keep[0] == 1
    assert indices_to_keep[1] == 3
    assert indices_to_keep[2] == 4

    # proportion of slices to keep
    indices_to_keep = stack.get_in_focus_indices(focus, proportion=0.5)
    assert isinstance(indices_to_keep, list)
    assert len(indices_to_keep) == 3
    assert indices_to_keep[0] == 1
    assert indices_to_keep[1] == 3
    assert indices_to_keep[2] == 4

    # 0 slice
    indices_to_keep = stack.get_in_focus_indices(focus, proportion=0)
    assert isinstance(indices_to_keep, list)
    assert len(indices_to_keep) == 0

    # error: float > 1
    with pytest.raises(ValueError):
        _ = stack.get_in_focus_indices(focus, proportion=3.)

    # error: 'proportion' < 0
    with pytest.raises(ValueError):
        _ = stack.get_in_focus_indices(focus, proportion=-3)


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.int32, np.int64, np.float32, np.float64])
def test_in_focus_selection(dtype):
    x = x_3d_out_focus.astype(dtype)
    expected_y = x_3d.astype(dtype)
    focus = stack.compute_focus(x_3d_out_focus, neighborhood_size=31)
    y = stack.in_focus_selection(x, focus, proportion=3)
    assert_array_equal(y, expected_y)
    assert y.dtype == dtype


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.int32, np.int64, np.float32, np.float64])
def test_focus_projection(dtype):
    x = x_3d_out_focus.astype(dtype)

    # 'method' == "median"
    expected_y = np.array([[2, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=dtype)
    y = stack.focus_projection(
        x, proportion=3, neighborhood_size=7, method="median")
    assert_array_equal(y, expected_y)
    assert y.dtype == dtype

    # 'method' == "max"
    expected_y = np.array([[3, 0, 0, 0, 0],
                           [0, 3, 0, 0, 3],
                           [0, 3, 0, 0, 0],
                           [0, 3, 3, 3, 0],
                           [0, 0, 0, 0, 0]], dtype=dtype)
    y = stack.focus_projection(
        x, proportion=3, neighborhood_size=7, method="max")
    assert_array_equal(y, expected_y)
    assert y.dtype == dtype

    # error: method not in ["median", "max"]
    with pytest.raises(ValueError):
        _ = stack.focus_projection(
            x, proportion=3, neighborhood_size=7, method="mean")
