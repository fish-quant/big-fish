# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.io module.
"""

import os
import pytest
import mrc
import tempfile

import numpy as np
import pandas as pd
import bigfish.stack as stack

from numpy.testing import assert_array_equal

# TODO test bigfish.stack.read_cell_extracted
# TODO test bigfish.stack.save_cell_extracted


@pytest.mark.parametrize("shape", [
    (8, 8), (8, 8, 8), (8, 8, 8, 8), (8, 8, 8, 8, 8)])
@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
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
    np.uint8, np.uint16, np.uint32, np.uint64,
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


@pytest.mark.parametrize("shape", [
    (8, 8), (8, 8, 8), (8, 8, 8, 8), (8, 8, 8, 8, 8)])
@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64, bool])
def test_npz(shape, dtype):
    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_1 = np.zeros(shape, dtype=dtype)
        test_2 = np.ones(shape, dtype=dtype)
        path = os.path.join(tmp_dir, "test.npz")
        np.savez(path, test_1=test_1, test_2=test_2)
        data = stack.read_uncompressed(path)
        assert data.files == ["test_1", "test_2"]
        assert_array_equal(test_1, data["test_1"])
        assert_array_equal(test_2, data["test_2"])
        assert test_1.dtype == data["test_1"].dtype
        assert test_2.dtype == data["test_2"].dtype


@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
    np.float16, np.float32, np.float64])
@pytest.mark.parametrize("delimiter", [";", ",", ":", "\t"])
def test_csv_numpy(dtype, delimiter):
    # build a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # save arrays in csv file
        test_array = np.zeros((10, 2), dtype=dtype)
        path = os.path.join(tmp_dir, "test_array.csv")
        stack.save_data_to_csv(test_array, path, delimiter=delimiter)

        # read csv file
        array = stack.read_array_from_csv(path, delimiter=delimiter)
        assert array.dtype == np.float64
        array = stack.read_array_from_csv(path, dtype, delimiter)
        assert_array_equal(test_array, array)
        assert array.dtype == dtype


@pytest.mark.parametrize("delimiter", [";", ",", ":", "\t"])
def test_csv_pandas(delimiter):
    # build a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # save pandas objects in csv file
        test_series = pd.Series([0.1, 2.5, 3.7], name="a")
        test_dataframe = pd.DataFrame({"a": [0, 2, 3],
                                       "b": [0.1, 2.5, 3.7],
                                       "c": ["dog", "cat", "bird"]})
        path = os.path.join(tmp_dir, "test_series.csv")
        stack.save_data_to_csv(test_series, path, delimiter=delimiter)
        path = os.path.join(tmp_dir, "test_dataframe.csv")
        stack.save_data_to_csv(test_dataframe, path, delimiter=delimiter)

        # read csv files
        path = os.path.join(tmp_dir, "test_series.csv")
        df = stack.read_dataframe_from_csv(path, delimiter=delimiter)
        pd.testing.assert_frame_equal(test_series.to_frame(), df)
        path = os.path.join(tmp_dir, "test_dataframe.csv")
        df = stack.read_dataframe_from_csv(path, delimiter=delimiter)
        pd.testing.assert_frame_equal(test_dataframe, df)
