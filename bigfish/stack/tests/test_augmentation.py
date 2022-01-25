# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.augmentation module.
"""

import pytest

import numpy as np
import bigfish.stack as stack

from bigfish.stack.augmentation import _identity
from bigfish.stack.augmentation import _flip_h
from bigfish.stack.augmentation import _flip_v
from bigfish.stack.augmentation import _transpose
from bigfish.stack.augmentation import _transpose_inverse
from bigfish.stack.augmentation import _rotation_90
from bigfish.stack.augmentation import _rotation_180
from bigfish.stack.augmentation import _rotation_270

from numpy.testing import assert_array_equal


# toy image
x = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 0, 0, 0]], dtype=np.uint8)


def test_identity():
    y = _identity(x)
    assert_array_equal(y, x)


def test_flip_h():
    # one channel
    y = _flip_h(x)
    expected_y = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [1, 0, 0, 0, 0]], dtype=np.uint8)
    assert_array_equal(y, expected_y)

    # multichannel
    xx = x[..., np.newaxis]
    yy = _flip_h(xx)
    expected_yy = expected_y[..., np.newaxis]
    assert_array_equal(yy, expected_yy)


def test_flip_v():
    # one channel
    y = _flip_v(x)
    expected_y = np.array([[0, 0, 0, 0, 1],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    assert_array_equal(y, expected_y)

    # multichannel
    xx = x[..., np.newaxis]
    yy = _flip_v(xx)
    expected_yy = expected_y[..., np.newaxis]
    assert_array_equal(yy, expected_yy)


def test_transpose():
    # one channel
    y = _transpose(x)
    expected_y = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    assert_array_equal(y, expected_y)

    # multichannel
    xx = x[..., np.newaxis]
    yy = _transpose(xx)
    expected_yy = expected_y[..., np.newaxis]
    assert_array_equal(yy, expected_yy)


def test_transpose_inverse():
    y = _transpose_inverse(x)
    expected_y = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 1]], dtype=np.uint8)
    assert_array_equal(y, expected_y)


def test_rotation_90():
    y = _rotation_90(x)
    expected_y = np.array([[0, 0, 0, 0, 1],
                           [0, 1, 1, 1, 0],
                           [0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    assert_array_equal(y, expected_y)


def test_rotation_180():
    y = _rotation_180(x)
    expected_y = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]], dtype=np.uint8)
    assert_array_equal(y, expected_y)


def test_rotation_270():
    y = _rotation_270(x)
    expected_y = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0],
                           [0, 1, 1, 1, 0],
                           [1, 0, 0, 0, 0]], dtype=np.uint8)
    assert_array_equal(y, expected_y)


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32,np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64,
    bool])
def test_augment_2d_dtype(dtype):
    x = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=dtype)
    y = stack.augment_2d(x)
    assert y.dtype == dtype


def test_augment_2d_function():
    operations = [_identity,
                  _flip_h, _flip_v,
                  _transpose, _transpose_inverse,
                  _rotation_90, _rotation_180, _rotation_270]
    bytecodes = []
    for f in operations:
        bytecode = f.__code__.co_code
        bytecodes.append(bytecode)
    f = stack.augment_2d_function()
    assert f.__code__.co_code in bytecodes
    f = stack.augment_2d_function(identity=True)
    assert f.__code__.co_code == _identity.__code__.co_code


def test_augment_8_times():
    # one channel
    expected_y_identity = x.copy()
    expected_y_flip_h = np.array([[0, 0, 0, 0, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 1, 0, 0, 0],
                                  [0, 1, 0, 0, 0],
                                  [1, 0, 0, 0, 0]], dtype=np.uint8)
    expected_y_flip_v = np.array([[0, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0]], dtype=np.uint8)
    expected_y_transpose = np.array([[1, 0, 0, 0, 0],
                                     [0, 1, 1, 1, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0]], dtype=np.uint8)
    expected_y_transpose_inverse = np.array([[0, 0, 0, 0, 0],
                                             [0, 1, 0, 0, 0],
                                             [0, 1, 0, 0, 0],
                                             [0, 1, 1, 1, 0],
                                             [0, 0, 0, 0, 1]], dtype=np.uint8)
    expected_y_rotation_90 = np.array([[0, 0, 0, 0, 1],
                                       [0, 1, 1, 1, 0],
                                       [0, 1, 0, 0, 0],
                                       [0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    expected_y_rotation_180 = np.array([[0, 0, 0, 0, 0],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 1]], dtype=np.uint8)
    expected_y_rotation_270 = np.array([[0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 0, 0, 1, 0],
                                        [0, 1, 1, 1, 0],
                                        [1, 0, 0, 0, 0]], dtype=np.uint8)
    expected_y = [expected_y_identity,
                  expected_y_flip_h, expected_y_flip_v,
                  expected_y_transpose, expected_y_transpose_inverse,
                  expected_y_rotation_90, expected_y_rotation_180,
                  expected_y_rotation_270]
    augmented_arrays = stack.augment_8_times(x)
    assert isinstance(augmented_arrays, list)
    assert len(augmented_arrays) == len(expected_y)
    for a, b in zip(augmented_arrays, expected_y):
        assert_array_equal(a, b)

    # multichannel
    xx = x[..., np.newaxis]
    expected_yy = [y[..., np.newaxis] for y in expected_y]
    augmented_arrays = stack.augment_8_times(xx)
    assert isinstance(augmented_arrays, list)
    assert len(augmented_arrays) == len(expected_yy)
    for a, b in zip(augmented_arrays, expected_yy):
        assert_array_equal(a, b)


def test_augment_8_times_reversed():
    y = stack.augment_8_times(x)
    y_reversed = stack.augment_8_times_reversed(y)
    assert isinstance(y_reversed, list)
    assert len(y_reversed) == 8
    for a in y_reversed:
        assert_array_equal(a, x.copy())
