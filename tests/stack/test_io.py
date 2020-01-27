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


def test_io():

    # build different arrays to write and read
    shapes = [(64, 64), (20, 64, 64), (5, 20, 64, 64), (10, 5, 20, 64, 64)]
    dtypes = [np.uint8, np.uint16, np.uint32,
              np.int8, np.int16, np.int32,
              np.float32, np.float64, bool]
    extensions = ["png", "jpg", "jpeg", "tif", "tiff"]

    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:

        for shape in shapes:
            for dtype in dtypes:
                for extension in extensions:
                    test = np.zeros(shape, dtype=dtype)
                    path = os.path.join(tmp_dir, "test." + extension)

                    # multidimensional image
                    if (extension in ["png", "jpg", "jpeg"]
                            and len(test.shape) > 2
                            and test.dtype != bool):
                        with pytest.raises(ValueError):
                            stack.save_image(test, path)

                    # 2-d image
                    elif (extension in ["png", "jpg", "jpeg"]
                            and len(test.shape) == 2):
                        with pytest.warns(UserWarning):
                            stack.save_image(test, path)
                            tensor = stack.read_image(path, sanity_check=True)
                            assert_array_equal(test, tensor)

                    # boolean image
                    elif (extension in ["tif", "tiff"]
                            and len(test.shape) == 2
                            and test.dtype == bool):
                        with pytest.raises(ValueError):
                            stack.save_image(test, path)

                    # multidimensional boolean image
                    elif (extension in ["png", "jpg", "jpeg", "tif", "tiff"]
                            and len(test.shape) > 2
                            and test.dtype == bool):
                        with pytest.raises(ValueError):
                            stack.save_image(test, path)

                    # others valid images
                    else:
                        stack.save_image(test, path)
                        tensor = stack.read_image(path, sanity_check=True)
                        assert_array_equal(test, tensor)
                        assert test.dtype == tensor.dtype

        # non-supported image (1 dimension)
        test = np.array([1, 2, 3], dtype=np.float32)
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(ValueError):
            stack.save_image(test, path)

        # non-supported image (6 dimensions)
        test = np.zeros((1, 10, 5, 20, 64, 64), dtype=np.float32)
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(ValueError):
            stack.save_image(test, path)

        # non-supported image (np.int64)
        test = np.zeros((64, 64), dtype=np.int64)
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(TypeError):
            stack.save_image(test, path)

        # non-supported image (missing values)
        test = np.zeros((64, 64), dtype=np.float32)
        test[0, 0] = np.nan
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(ValueError):
            stack.save_image(test, path)

        # non-supported extensions
        test = np.zeros((64, 64), dtype=np.float32)
        path = os.path.join(tmp_dir, "test.foo")
        with pytest.raises(ValueError):
            stack.save_image(test, path)
        path = os.path.join(tmp_dir, "test")
        with pytest.raises(ValueError):
            stack.save_image(test, path, extension="bar")

    return


def test_dv():
    # build different arrays to write and read
    dtypes = [np.uint16, np.int16, np.int32, np.float32]

    # build a temporary directory and save a dv file inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        for dtype in dtypes:
            test = np.zeros((20, 5, 256, 256), dtype=dtype)
            path = os.path.join(tmp_dir, "test")
            mrc.imsave(path, test)

            # read and test the dv file
            path = os.path.join(tmp_dir, "test")
            tensor = stack.read_dv(path, sanity_check=True)
            assert_array_equal(test, tensor)
            assert test.dtype == tensor.dtype

    return


def test_npy():
    # build different arrays to write and read
    shapes = [(64, 64), (20, 64, 64), (5, 20, 64, 64), (10, 5, 20, 64, 64)]
    dtypes = [np.uint8, np.uint16, np.uint32,
              np.int8, np.int16, np.int32, np.int64,
              np.float32, np.float64, bool]

    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:

        for shape in shapes:
            for dtype in dtypes:
                test = np.zeros(shape, dtype=dtype)
                path = os.path.join(tmp_dir, "test.npy")
                stack.save_array(test, path)
                tensor = stack.read_array(path)
                assert_array_equal(test, tensor)
                assert test.dtype == tensor.dtype

    return
