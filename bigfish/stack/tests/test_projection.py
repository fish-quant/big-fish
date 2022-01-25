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


# TODO test bigfish.stack.focus_projection
# TODO test bigfish.stack.in_focus_selection
# TODO test bigfish.stack.get_in_focus_indices

# toy images
x = np.array(
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

x_out_focus = np.array(
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

def test_maximum_projection():
    # uint8
    y = stack.maximum_projection(x)
    expected_y = np.array([[3, 0, 0, 0, 0],
                           [0, 3, 0, 0, 3],
                           [0, 3, 0, 0, 0],
                           [0, 3, 3, 3, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    assert_array_equal(y, expected_y)
    assert y.dtype == np.uint8

    # uint16
    y = stack.maximum_projection(x.astype(np.uint16))
    expected_y = expected_y.astype(np.uint16)
    assert_array_equal(y, expected_y)
    assert y.dtype == np.uint16


def test_mean_projection():
    # uint8
    y = stack.mean_projection(x)
    expected_y = np.array([[2, 0, 0, 0, 0],
                           [0, 1, 0, 0, 1],
                           [0, 1, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    assert_array_equal(y, expected_y)
    assert y.dtype == np.uint8

    # uint16
    y = stack.mean_projection(x.astype(np.uint16))
    expected_y = expected_y.astype(np.uint16)
    assert_array_equal(y, expected_y)
    assert y.dtype == np.uint16


def test_median_projection():
    # uint8
    y = stack.median_projection(x)
    expected_y = np.array([[2, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    assert_array_equal(y, expected_y)
    assert y.dtype == np.uint8

    # uint16
    y = stack.median_projection(x.astype(np.uint16))
    expected_y = expected_y.astype(np.uint16)
    assert_array_equal(y, expected_y)
    assert y.dtype == np.uint16


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
