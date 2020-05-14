# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.preprocess module.
"""

import os
import pytest
import tempfile

import numpy as np
import bigfish.stack as stack

from numpy.testing import assert_array_equal


# ### Test stack building ###

def test_build_stacks_from_recipe():
    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        # field of view 1
        test_nuc = np.zeros((8, 8, 8), dtype=np.uint8)
        test_cyt = np.zeros((8, 8, 8), dtype=np.uint8)
        test_rna = np.zeros((8, 8, 8), dtype=np.uint8)
        path_nuc = os.path.join(tmp_dir, "test_nuc_1.tif")
        path_cyt = os.path.join(tmp_dir, "test_cyt_1.tif")
        path_rna = os.path.join(tmp_dir, "test_rna_1.tif")
        stack.save_image(test_nuc, path_nuc)
        stack.save_image(test_cyt, path_cyt)
        stack.save_image(test_rna, path_rna)

        # field of view 2
        test_nuc = np.zeros((5, 5, 5), dtype=np.uint16)
        test_cyt = np.zeros((5, 5, 5), dtype=np.uint16)
        test_rna = np.zeros((5, 5, 5), dtype=np.uint16)
        path_nuc = os.path.join(tmp_dir, "test_nuc_2.tif")
        path_cyt = os.path.join(tmp_dir, "test_cyt_2.tif")
        path_rna = os.path.join(tmp_dir, "test_rna_2.tif")
        stack.save_image(test_nuc, path_nuc)
        stack.save_image(test_cyt, path_cyt)
        stack.save_image(test_rna, path_rna)

        # define recipe to read tensors
        recipe_1 ={"fov": ["1", "2"],
                   "c": ["nuc", "cyt", "rna"],
                   "opt": "test",
                   "ext": "tif",
                   "pattern": "opt_c_fov.ext"}

        # build tensor without prior information
        tensor = stack.build_stack(recipe_1, input_folder=tmp_dir)
        expected_tensor = np.zeros((1, 3, 8, 8, 8), dtype=np.uint8)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint8

        # build tensor with prior information
        tensor = stack.build_stack(recipe_1,
                                   input_folder=tmp_dir,
                                   input_dimension=3)
        expected_tensor = np.zeros((1, 3, 8, 8, 8), dtype=np.uint8)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint8

        # build tensors with different fields of view
        tensor = stack.build_stack(recipe_1,
                                   input_folder=tmp_dir,
                                   input_dimension=3,
                                   i_fov=0)
        expected_tensor = np.zeros((1, 3, 8, 8, 8), dtype=np.uint8)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint8
        tensor = stack.build_stack(recipe_1,
                                   input_folder=tmp_dir,
                                   input_dimension=3,
                                   i_fov=1)
        expected_tensor = np.zeros((1, 3, 5, 5, 5), dtype=np.uint16)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint16

        # wrong recipe
        recipe_wrong = {"fov": "test",
                        "c": ["nuc", "cyt", "rna"],
                        "ext": "tif",
                        "pattern": "fov_c.ext"}
        with pytest.raises(FileNotFoundError):
            stack.build_stack(recipe_wrong,
                              input_folder=tmp_dir,
                              input_dimension=3)

        # wrong path
        with pytest.raises(FileNotFoundError):
            stack.build_stack(recipe_1,
                              input_folder="/foo/bar",
                              input_dimension=3)


def test_build_stacks_from_datamap():
    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        # field of view 1
        test_nuc = np.zeros((8, 8, 8), dtype=np.uint8)
        test_cyt = np.zeros((8, 8, 8), dtype=np.uint8)
        test_rna = np.zeros((8, 8, 8), dtype=np.uint8)
        path_nuc = os.path.join(tmp_dir, "test_nuc_1.tif")
        path_cyt = os.path.join(tmp_dir, "test_cyt_1.tif")
        path_rna = os.path.join(tmp_dir, "test_rna_1.tif")
        stack.save_image(test_nuc, path_nuc)
        stack.save_image(test_cyt, path_cyt)
        stack.save_image(test_rna, path_rna)

        # field of view 2
        test_nuc = np.zeros((5, 5, 5), dtype=np.uint16)
        test_cyt = np.zeros((5, 5, 5), dtype=np.uint16)
        test_rna = np.zeros((5, 5, 5), dtype=np.uint16)
        path_nuc = os.path.join(tmp_dir, "test_nuc_2.tif")
        path_cyt = os.path.join(tmp_dir, "test_cyt_2.tif")
        path_rna = os.path.join(tmp_dir, "test_rna_2.tif")
        stack.save_image(test_nuc, path_nuc)
        stack.save_image(test_cyt, path_cyt)
        stack.save_image(test_rna, path_rna)

        # define datamap to read  tensors
        recipe_1 = {"fov": ["1", "2"],
                    "c": ["nuc", "cyt", "rna"],
                    "opt": "test",
                    "ext": "tif",
                    "pattern": "opt_c_fov.ext"}
        recipe_2 = {"fov": "2",
                    "c": ["nuc", "cyt", "rna"],
                    "opt": "test",
                    "ext": "tif",
                    "pattern": "opt_c_fov.ext"}
        data_map = [(recipe_1, tmp_dir), (recipe_2, tmp_dir)]

        # build stacks from generator
        generator = stack.build_stacks(data_map, input_dimension=3)
        expected_tensors = [np.zeros((1, 3, 8, 8, 8), dtype=np.uint8),
                            np.zeros((1, 3, 5, 5, 5), dtype=np.uint16),
                            np.zeros((1, 3, 5, 5, 5), dtype=np.uint16)]
        for i, tensor in enumerate(generator):
            expected_tensor = expected_tensors[i]
            assert_array_equal(tensor, expected_tensor)
            assert tensor.dtype == expected_tensor.dtype

        # build stacks from generator with metadata
        generator = stack.build_stacks(data_map,
                                       input_dimension=3,
                                       return_origin=True)
        expected_tensors = [np.zeros((1, 3, 8, 8, 8), dtype=np.uint8),
                            np.zeros((1, 3, 5, 5, 5), dtype=np.uint16),
                            np.zeros((1, 3, 5, 5, 5), dtype=np.uint16)]
        expected_recipes = [recipe_1, recipe_1, recipe_2]
        expected_i_fov = [0, 1, 0]
        for i, (tensor, input_folder, recipe, i_fov) in enumerate(generator):
            expected_tensor = expected_tensors[i]
            assert_array_equal(tensor, expected_tensor)
            assert tensor.dtype == expected_tensor.dtype
            assert input_folder == tmp_dir
            assert recipe == expected_recipes[i]
            assert i_fov == expected_i_fov[i]

        # wrong datamap
        data_map = [(recipe_1, 3), (recipe_2, tmp_dir)]
        generator = stack.build_stacks(data_map, input_dimension=3)
        with pytest.raises(TypeError):
            next(generator)
        data_map = [(recipe_1, "foo/bar"), (recipe_2, tmp_dir)]
        generator = stack.build_stacks(data_map, input_dimension=3)
        with pytest.raises(NotADirectoryError):
            next(generator)


def test_build_stack_from_path():
    # build a temporary directory and save tensors inside
    with tempfile.TemporaryDirectory() as tmp_dir:
        # field of view
        test_nuc = np.zeros((8, 8, 8), dtype=np.uint8)
        test_cyt = np.zeros((8, 8, 8), dtype=np.uint8)
        test_rna = np.zeros((8, 8, 8), dtype=np.uint8)
        path_nuc = os.path.join(tmp_dir, "test_nuc.tif")
        path_cyt = os.path.join(tmp_dir, "test_cyt.tif")
        path_rna = os.path.join(tmp_dir, "test_rna.tif")
        stack.save_image(test_nuc, path_nuc)
        stack.save_image(test_cyt, path_cyt)
        stack.save_image(test_rna, path_rna)

        # build tensor from paths
        paths = [path_nuc, path_cyt, path_rna]
        tensor = stack.build_stack_no_recipe(paths, input_dimension=3)
        expected_tensor = np.zeros((1, 3, 8, 8, 8), dtype=np.uint8)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint8

        # wrong paths
        paths = [path_nuc, path_cyt, "/foo/bar/test_rna.tif"]
        with pytest.raises(FileNotFoundError):
            stack.build_stack_no_recipe(paths, input_dimension=3)


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
