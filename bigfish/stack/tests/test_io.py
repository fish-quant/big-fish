# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.io submodule.
"""

import os
import pytest
import mrc
import tempfile

import numpy as np
import bigfish.stack as stack

from numpy.testing import assert_array_equal


@pytest.mark.parametrize("shape", [
    (8, 8), (8, 8, 8), (8, 8, 8, 8), (8, 8, 8, 8, 8)])
@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32,
    np.int8, np.int16, np.int32,
    np.float16, np.float32, np.float64, bool])
@pytest.mark.parametrize("extension", [
    "png", "jpg", "jpeg", "tif", "tiff"])
def test_image(shape, dtype, extension):
    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        test = np.zeros(shape, dtype=dtype)
        path = os.path.join(tmp_dir, "test." + extension)

        # error: boolean multidimensional image
        if (extension in ["png", "jpg", "jpeg", "tif", "tiff"]
                and len(test.shape) > 2
                and test.dtype == bool):
            with pytest.raises(ValueError):
                stack.save_image(test, path)

        # error: non-boolean multidimensional image with 'png', 'jpg' or 'jpeg'
        elif (extension in ["png", "jpg", "jpeg"]
                and len(test.shape) > 2
                and test.dtype != bool):
            with pytest.raises(ValueError):
                stack.save_image(test, path)

        # error: boolean 2-d image with 'tig' and 'tiff'
        elif (extension in ["tif", "tiff"]
                and len(test.shape) == 2
                and test.dtype == bool):
            with pytest.raises(ValueError):
                stack.save_image(test, path)

        # warning: 2-d image with 'png', 'jpg' or 'jpeg'
        elif (extension in ["png", "jpg", "jpeg"]
                and len(test.shape) == 2):
            with pytest.warns(UserWarning):
                stack.save_image(test, path)
                tensor = stack.read_image(path, sanity_check=True)
                assert_array_equal(test, tensor)

        # others valid images
        else:
            stack.save_image(test, path)
            tensor = stack.read_image(path, sanity_check=True)
            assert_array_equal(test, tensor)
            assert test.dtype == tensor.dtype


def test_image_specific():
    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        # non-supported image (1 dimension)
        test = np.array([1, 2, 3], dtype=np.float32)
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(ValueError):
            stack.save_image(test, path)

        # non-supported image (6 dimensions)
        test = np.zeros((8, 8, 8, 8, 8, 8), dtype=np.float32)
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(ValueError):
            stack.save_image(test, path)

        # non-supported image (np.int64)
        test = np.zeros((8, 8), dtype=np.int64)
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(TypeError):
            stack.save_image(test, path)

        # non-supported image (missing values)
        test = np.zeros((8, 8), dtype=np.float32)
        test[0, 0] = np.nan
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(ValueError):
            stack.save_image(test, path)

        # non-supported extensions
        test = np.zeros((8, 8), dtype=np.float32)
        path = os.path.join(tmp_dir, "test.foo")
        with pytest.raises(ValueError):
            stack.save_image(test, path)
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(ValueError):
            stack.save_image(test, path, extension="bar")


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.int32, np.float32])
def test_dv(dtype):
    # build a temporary directory and save a dv file inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        test = np.zeros((8, 8, 8, 8), dtype=dtype)
        path = os.path.join(tmp_dir, "test")
        mrc.imsave(path, test)

        # read and test the dv file
        path = os.path.join(tmp_dir, "test")
        tensor = stack.read_dv(path, sanity_check=True)
        assert_array_equal(test, tensor)
        assert test.dtype == tensor.dtype


@pytest.mark.parametrize("shape", [
    (8, 8), (8, 8, 8), (8, 8, 8, 8), (8, 8, 8, 8, 8)])
@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64, bool])
def test_npy(shape, dtype):
    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        test = np.zeros(shape, dtype=dtype)
        path = os.path.join(tmp_dir, "test.npy")
        stack.save_array(test, path)
        tensor = stack.read_array(path)
        assert_array_equal(test, tensor)
        assert test.dtype == tensor.dtype
