# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.preprocess module.
"""

import pytest

import numpy as np
import bigfish.stack as stack

from numpy.testing import assert_array_equal


# ### Test normalization ###

@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32,
    np.int8, np.int16, np.int32,
    np.float16, np.float32, np.float64])
def test_rescale(dtype):
    # build a 5x5 random matrix with a limited range of values
    tensor = np.random.randint(35, 105, 25).reshape((5, 5)).astype(dtype)

    # rescale tensor
    rescaled_tensor = stack.rescale(tensor)

    # test consistency of the function
    if dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
        i = np.iinfo(dtype)
        min_, max_ = 0, i.max
    else:
        min_, max_ = 0, 1
    assert rescaled_tensor.min() == min_
    assert rescaled_tensor.max() == max_
    assert rescaled_tensor.dtype == dtype
    assert rescaled_tensor.shape == (5, 5)


def test_stretching():
    x = [[51, 51, 51], [102, 102, 102], [153, 153, 153]]

    # integer
    tensor = np.array(x).reshape((3, 3)).astype(np.uint16)
    tensor_rescaled = stack.rescale(tensor,
                                    channel_to_stretch=0,
                                    stretching_percentile=50)
    expected_tensor = np.array([[0, 0, 0],
                                [65535, 65535, 65535],
                                [65535, 65535, 65535]], dtype=np.uint16)
    assert_array_equal(tensor_rescaled, expected_tensor)

    # float
    tensor = np.array(x).reshape((3, 3)).astype(np.float32)
    rescaled_tensor = stack.rescale(tensor,
                                    channel_to_stretch=0,
                                    stretching_percentile=50)
    expected_tensor = np.array([[0., 0., 0.],
                                [1., 1., 1.],
                                [1., 1., 1.]], dtype=np.float32)
    assert_array_equal(rescaled_tensor, expected_tensor)


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64])
def test_cast_uint8(dtype):
    # from integer to np.uint8
    if np.issubdtype(dtype, np.integer):
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tensor = np.array(x).reshape((3, 3)).astype(dtype)
        tensor[2, 2] = np.iinfo(dtype).max

    # from float to np.uint8
    else:
        x = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 1.0]]
        tensor = np.array(x).reshape((3, 3)).astype(dtype)

    # cast in uint8
    if dtype in [np.uint8, np.int8]:
        tensor_uint8 = stack.cast_img_uint8(tensor)
    else:
        with pytest.warns(UserWarning):
            tensor_uint8 = stack.cast_img_uint8(tensor)
    assert tensor_uint8.dtype == np.uint8


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64])
def test_cast_uint16(dtype):
    # from integer to np.uint16
    if np.issubdtype(dtype, np.integer):
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tensor = np.array(x).reshape((3, 3)).astype(dtype)
        tensor[2, 2] = np.iinfo(dtype).max

    # from float to np.uint16
    else:
        x = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 1.0]]
        tensor = np.array(x).reshape((3, 3)).astype(dtype)

    # cast in uint16
    if dtype in [np.uint8, np.int8, np.uint16, np.int16, np.float16]:
        tensor_uint16 = stack.cast_img_uint16(tensor)
    else:
        with pytest.warns(UserWarning):
            tensor_uint16 = stack.cast_img_uint16(tensor)
    assert tensor_uint16.dtype == np.uint16


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64])
def test_cast_float32(dtype):
    # from integer to np.float32
    if np.issubdtype(dtype, np.integer):
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tensor = np.array(x).reshape((3, 3)).astype(dtype)
        tensor[2, 2] = np.iinfo(dtype).max

    # from float to np.float32
    else:
        x = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 1.0]]
        tensor = np.array(x).reshape((3, 3)).astype(dtype)

    # cast in float32
    if dtype == np.float64:
        with pytest.warns(UserWarning):
            tensor_float32 = stack.cast_img_float32(tensor)
    else:
        tensor_float32 = stack.cast_img_float32(tensor)
    assert tensor_float32.dtype == np.float32


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64])
def test_cast_float64(dtype):
    # from integer to np.float64
    if np.issubdtype(dtype, np.integer):
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        tensor = np.array(x).reshape((3, 3)).astype(dtype)
        tensor[2, 2] = np.iinfo(dtype).max

    # from float to np.float64
    else:
        x = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 1.0]]
        tensor = np.array(x).reshape((3, 3)).astype(dtype)

    # cast in float64
    tensor_float64 = stack.cast_img_float64(tensor)
    assert tensor_float64.dtype == np.float64
