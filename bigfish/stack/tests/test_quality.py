# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.quality module.
"""

import pytest

import numpy as np
import bigfish.stack as stack


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


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("neighborhood_size", [
    31, (5, 11), [11, 5]])
def test_compute_focus_format(dtype, neighborhood_size):
    # 2D input
    image = np.random.normal(loc=0.0, scale=1.0, size=10000)
    image = np.reshape(image, (100, 100)).astype(dtype)
    focus = stack.compute_focus(image, neighborhood_size=neighborhood_size)
    assert focus.dtype == np.float64
    assert focus.shape == image.shape
    assert focus.min() >= 1

    # 3D input
    image = np.random.normal(loc=0.0, scale=1.0, size=100000)
    image = np.reshape(image, (10, 100, 100)).astype(dtype)
    focus = stack.compute_focus(image, neighborhood_size=neighborhood_size)
    assert focus.dtype == np.float64
    assert focus.shape == image.shape
    assert focus.min() >= 1


def test_compute_focus_error():
    image = np.random.normal(loc=0.0, scale=1.0, size=10000)
    image = np.reshape(image, (100, 100))
    # error: sequence 'neighborhood_size' too short
    with pytest.raises(ValueError):
        _ = stack.compute_focus(image, neighborhood_size=[11])
    # error: sequence 'neighborhood_size' too long
    with pytest.raises(ValueError):
        _ = stack.compute_focus(image, neighborhood_size=(5, 11, 31))


def test_compute_focus():
    focus = stack.compute_focus(x_out_focus, neighborhood_size=31)
    average_focus = focus.mean(axis=(1, 2))
    assert round(average_focus[0], 5) == 1.
    assert round(average_focus[1], 5) > 1.
    assert round(average_focus[2], 5) == 1.
    assert round(average_focus[3], 5) > 1.
    assert round(average_focus[4], 5) > 1.
    assert round(average_focus[5], 5) == 1.
