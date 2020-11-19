# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.filter module.
"""

import pytest

import numpy as np
import bigfish.stack as stack

from bigfish.stack.filter import _define_kernel

from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose


# toy images
x = np.array(
    [[3, 2, 0, 0, 0],
     [2, 1, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 2, 1, 5, 0],
     [0, 0, 0, 0, 0]],
    dtype=np.uint8)
y = np.array(
    [[0, 0, 62, 164, 55],
     [0, 0, 120, 235, 181],
     [0, 0, 73, 205, 0],
     [0, 131, 0, 0, 0],
     [0, 0, 0, 0, 0]],
    dtype=np.uint8)


@pytest.mark.parametrize("shape, size", [
    ("diamond", 3), ("disk", 3), ("rectangle", (2, 3)), ("square", 3),
    ("blabla", 3)])
@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64, bool])
def test_kernel(shape, size, dtype):
    # non valid case
    if shape not in ["diamond", "disk", "rectangle", "square"]:
        with pytest.raises(ValueError):
            _define_kernel(shape, size, dtype)

    # valid cases
    else:
        kernel = _define_kernel(shape, size, dtype)
        if shape == "diamond":
            expected_kernel = np.array(
                [[0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0, 0],
                 [0, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 0],
                 [0, 0, 1, 1, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0]],
                dtype=dtype)
        elif shape == "disk":
            expected_kernel = np.array(
                [[0, 0, 0, 1, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 1, 0],
                 [0, 0, 0, 1, 0, 0, 0]],
                dtype=dtype)
        elif shape == "rectangle":
            expected_kernel = np.array(
                [[1, 1, 1],
                 [1, 1, 1]],
                dtype=dtype)
        else:
            expected_kernel = np.array(
                [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]],
                dtype=dtype)
        assert_array_equal(kernel, expected_kernel)
        assert kernel.dtype == dtype


def test_mean_filter():
    # np.uint8
    filtered_x = stack.mean_filter(x,
                                   kernel_shape="square",
                                   kernel_size=3)
    expected_x = np.array(
        [[2, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]],
        dtype=np.uint8)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint8

    # np.uint16
    filtered_x = stack.mean_filter(x.astype(np.uint16),
                                   kernel_shape="square",
                                   kernel_size=3)
    expected_x = expected_x.astype(np.uint16)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint16

    # np.float32
    filtered_x = stack.mean_filter(x.astype(np.float32),
                                   kernel_shape="square",
                                   kernel_size=3)
    expected_x = np.array(
        [[2.333, 1.444, 0.556, 0., 0.],
         [1.556, 1., 0.444, 0., 0.],
         [0.889, 0.778, 1.111, 0.667, 0.556],
         [0.333, 0.444, 1., 0.667, 0.556],
         [0.222, 0.333, 0.889, 0.667, 0.556]],
        dtype=np.float32)
    assert_allclose(filtered_x, expected_x, rtol=1e-02)
    assert filtered_x.dtype == np.float32

    # np.float64
    filtered_x = stack.mean_filter(x.astype(np.float64),
                                   kernel_shape="square",
                                   kernel_size=3)
    expected_x = expected_x.astype(np.float64)
    assert_allclose(filtered_x, expected_x, rtol=1e-02)
    assert filtered_x.dtype == np.float64


def test_median_filter():
    # np.uint8
    filtered_x = stack.median_filter(x,
                                     kernel_shape="square",
                                     kernel_size=3)
    expected_x = np.array(
        [[2, 2, 0, 0, 0],
         [2, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0]],
        dtype=np.uint8)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint8

    # np.uint16
    filtered_x = stack.median_filter(x.astype(np.uint16),
                                     kernel_shape="square",
                                     kernel_size=3)
    expected_x = expected_x.astype(np.uint16)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint16


def test_maximum_filter():
    # np.uint8
    filtered_x = stack.maximum_filter(x,
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = np.array(
        [[3, 3, 2, 0, 0],
         [3, 3, 2, 0, 0],
         [2, 2, 5, 5, 5],
         [2, 2, 5, 5, 5],
         [2, 2, 5, 5, 5]],
        dtype=np.uint8)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint8

    # np.uint16
    filtered_x = stack.maximum_filter(x.astype(np.uint16),
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = expected_x.astype(np.uint16)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint16


def test_minimum_filter():
    # np.uint8
    filtered_x = stack.minimum_filter(x,
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = np.array(
        [[1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        dtype=np.uint8)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint8

    # np.uint16
    filtered_x = stack.minimum_filter(x.astype(np.uint16),
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = expected_x.astype(np.uint16)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint16


def test_log_filter():
    # float64
    y_float64 = stack.cast_img_float64(y)
    filtered_y_float64 = stack.log_filter(y_float64, 2)
    expected_y_float64 = np.array(
        [[0., 0., 0.02995949, 0.06212277, 0.07584532],
         [0., 0., 0.02581818, 0.05134284, 0.06123539],
         [0., 0., 0.01196859, 0.0253716, 0.02853162],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]],
        dtype=np.float64)
    assert_allclose(filtered_y_float64, expected_y_float64, rtol=1e-6)
    assert filtered_y_float64.dtype == np.float64

    # float32
    y_float32 = stack.cast_img_float32(y)
    filtered_y = stack.log_filter(y_float32, 2)
    expected_y = stack.cast_img_float32(expected_y_float64)
    assert_allclose(filtered_y, expected_y, rtol=1e-6)
    assert filtered_y.dtype == np.float32

    # uint8
    filtered_y = stack.log_filter(y, 2)
    expected_y = stack.cast_img_uint8(expected_y_float64)
    assert_array_equal(filtered_y, expected_y)
    assert filtered_y.dtype == np.uint8

    # uint16
    y_uint16 = stack.cast_img_uint16(y)
    filtered_y = stack.log_filter(y_uint16, 2)
    expected_y = stack.cast_img_uint16(expected_y_float64)
    assert_array_equal(filtered_y, expected_y)
    assert filtered_y.dtype == np.uint16


def test_gaussian_filter():
    # float64
    y_float64 = stack.cast_img_float64(y)
    filtered_y_float64 = stack.gaussian_filter(y_float64, 2)
    expected_y_float64 = np.array(
        [[0.08928096, 0.1573019 , 0.22897881, 0.28086597, 0.3001061 ],
         [0.08668051, 0.14896399, 0.21282558, 0.25752308, 0.27253406],
         [0.07634613, 0.12664142, 0.17574502, 0.20765944, 0.2155001 ],
         [0.05890843, 0.09356377, 0.12493327, 0.1427122 , 0.14374558],
         [0.03878372, 0.05873308, 0.07492625, 0.08201409, 0.07939603]],
        dtype=np.float64)
    assert_allclose(filtered_y_float64, expected_y_float64, rtol=1e-6)
    assert filtered_y_float64.dtype == np.float64

    # float32
    y_float32 = stack.cast_img_float32(y)
    filtered_y = stack.gaussian_filter(y_float32, 2)
    expected_y = stack.cast_img_float32(expected_y_float64)
    assert_allclose(filtered_y, expected_y, rtol=1e-6)
    assert filtered_y.dtype == np.float32

    # uint8
    with pytest.raises(ValueError):
        stack.gaussian_filter(y, 2, allow_negative=True)
    filtered_y = stack.gaussian_filter(y, 2)
    expected_y = stack.cast_img_uint8(expected_y_float64)
    assert_array_equal(filtered_y, expected_y)
    assert filtered_y.dtype == np.uint8

    # uint16
    y_uint16 = stack.cast_img_uint16(y)
    with pytest.raises(ValueError):
        stack.gaussian_filter(y_uint16, 2, allow_negative=True)
    filtered_y = stack.gaussian_filter(y_uint16, 2)
    expected_y = stack.cast_img_uint16(expected_y_float64)
    assert_array_equal(filtered_y, expected_y)
    assert filtered_y.dtype == np.uint16


def test_background_removal_mean():
    # np.uint8
    filtered_x = stack.remove_background_mean(x,
                                              kernel_shape="square",
                                              kernel_size=3)
    expected_x = np.array(
        [[1, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 2, 0, 5, 0],
         [0, 0, 0, 0, 0]],
        dtype=np.uint8)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint8

    # np.uint16
    filtered_x = stack.remove_background_mean(x.astype(np.uint16),
                                              kernel_shape="square",
                                              kernel_size=3)
    expected_x = expected_x.astype(np.uint16)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint16


def test_background_removal_gaussian():
    # float64
    y_float64 = stack.cast_img_float64(y)
    filtered_y_float64 = stack.remove_background_gaussian(y_float64, 2)
    expected_y_float64 = np.array(
        [[0., 0., 0.01415845, 0.36227129, 0.],
         [0., 0., 0.25776265, 0.66404555, 0.43726986],
         [0., 0., 0.11052949, 0.59626213, 0.],
         [0., 0.42016172, 0., 0., 0.],
         [0., 0., 0., 0., 0.]],
        dtype=np.float64)
    assert_allclose(filtered_y_float64, expected_y_float64, rtol=1e-6)
    assert filtered_y_float64.dtype == np.float64

    # float32
    y_float32 = stack.cast_img_float32(y)
    filtered_y = stack.remove_background_gaussian(y_float32, 2)
    expected_y = stack.cast_img_float32(expected_y_float64)
    assert_allclose(filtered_y, expected_y, rtol=1e-6)
    assert filtered_y.dtype == np.float32

    # uint8
    with pytest.raises(ValueError):
        stack.gaussian_filter(y, 2, allow_negative=True)
    filtered_y = stack.remove_background_gaussian(y, 2)
    expected_y = stack.cast_img_uint8(expected_y_float64)
    assert_array_equal(filtered_y, expected_y)
    assert filtered_y.dtype == np.uint8

    # uint16
    y_uint16 = stack.cast_img_uint16(y)
    with pytest.raises(ValueError):
        stack.gaussian_filter(y_uint16, 2, allow_negative=True)
    filtered_y = stack.remove_background_gaussian(y_uint16, 2)
    expected_y = stack.cast_img_uint16(expected_y_float64)
    assert_array_equal(filtered_y, expected_y)
    assert filtered_y.dtype == np.uint16


def test_dilation_filter():
    # np.uint8
    filtered_x = stack.dilation_filter(x,
                                       kernel_shape="square",
                                       kernel_size=3)
    expected_x = np.array(
        [[3, 3, 2, 0, 0],
         [3, 3, 2, 0, 0],
         [2, 2, 5, 5, 5],
         [2, 2, 5, 5, 5],
         [2, 2, 5, 5, 5]],
        dtype=np.uint8)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint8

    # np.uint16
    filtered_x = stack.dilation_filter(x.astype(np.uint16),
                                       kernel_shape="square",
                                       kernel_size=3)
    expected_x = expected_x.astype(np.uint16)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint16

    # np.float32
    filtered_x = stack.dilation_filter(x.astype(np.float32),
                                       kernel_shape="square",
                                       kernel_size=3)
    expected_x = expected_x.astype(np.float32)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.float32

    # np.float64
    filtered_x = stack.dilation_filter(x.astype(np.float64),
                                       kernel_shape="square",
                                       kernel_size=3)
    expected_x = expected_x.astype(np.float64)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.float64

    # bool
    filtered_x = stack.dilation_filter(x.astype(bool),
                                       kernel_shape="square",
                                       kernel_size=3)
    expected_x = expected_x.astype(bool)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == bool


def test_erosion_filter():
    # np.uint8
    filtered_x = stack.erosion_filter(x,
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = np.array(
        [[1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        dtype=np.uint8)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint8

    # np.uint16
    filtered_x = stack.erosion_filter(x.astype(np.uint16),
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = expected_x.astype(np.uint16)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.uint16

    # np.float32
    filtered_x = stack.erosion_filter(x.astype(np.float32),
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = expected_x.astype(np.float32)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.float32

    # np.float64
    filtered_x = stack.erosion_filter(x.astype(np.float64),
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = expected_x.astype(np.float64)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == np.float64

    # bool
    filtered_x = stack.erosion_filter(x.astype(bool),
                                      kernel_shape="square",
                                      kernel_size=3)
    expected_x = expected_x.astype(bool)
    assert_array_equal(filtered_x, expected_x)
    assert filtered_x.dtype == bool
