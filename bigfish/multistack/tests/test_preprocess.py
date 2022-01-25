# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.multistack.preprocess module.
"""

import os
import pytest
import tempfile

import numpy as np

import bigfish.stack as stack
import bigfish.multistack as multistack

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
        recipe_1 = {"fov": ["1", "2"],
                    "c": ["nuc", "cyt", "rna"],
                    "opt": "test",
                    "ext": "tif",
                    "pattern": "opt_c_fov.ext"}

        # build tensor without prior information
        tensor = multistack.build_stack(recipe_1, input_folder=tmp_dir)
        expected_tensor = np.zeros((1, 3, 8, 8, 8), dtype=np.uint8)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint8

        # build tensor with prior information
        tensor = multistack.build_stack(recipe_1,
                                        input_folder=tmp_dir,
                                        input_dimension=3)
        expected_tensor = np.zeros((1, 3, 8, 8, 8), dtype=np.uint8)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint8

        # build tensors with different fields of view
        tensor = multistack.build_stack(recipe_1,
                                        input_folder=tmp_dir,
                                        input_dimension=3,
                                        i_fov=0)
        expected_tensor = np.zeros((1, 3, 8, 8, 8), dtype=np.uint8)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint8
        tensor = multistack.build_stack(recipe_1,
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
            multistack.build_stack(recipe_wrong,
                                   input_folder=tmp_dir,
                                   input_dimension=3)

        # wrong path
        with pytest.raises(FileNotFoundError):
            multistack.build_stack(recipe_1,
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
        generator = multistack.build_stacks(data_map, input_dimension=3)
        expected_tensors = [np.zeros((1, 3, 8, 8, 8), dtype=np.uint8),
                            np.zeros((1, 3, 5, 5, 5), dtype=np.uint16),
                            np.zeros((1, 3, 5, 5, 5), dtype=np.uint16)]
        for i, tensor in enumerate(generator):
            expected_tensor = expected_tensors[i]
            assert_array_equal(tensor, expected_tensor)
            assert tensor.dtype == expected_tensor.dtype

        # build stacks from generator with metadata
        generator = multistack.build_stacks(data_map,
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
        generator = multistack.build_stacks(data_map, input_dimension=3)
        with pytest.raises(TypeError):
            next(generator)
        data_map = [(recipe_1, "foo/bar"), (recipe_2, tmp_dir)]
        generator = multistack.build_stacks(data_map, input_dimension=3)
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
        tensor = multistack.build_stack_no_recipe(paths, input_dimension=3)
        expected_tensor = np.zeros((1, 3, 8, 8, 8), dtype=np.uint8)
        assert_array_equal(tensor, expected_tensor)
        assert tensor.dtype == np.uint8

        # wrong paths
        paths = [path_nuc, path_cyt, "/foo/bar/test_rna.tif"]
        with pytest.raises(FileNotFoundError):
            multistack.build_stack_no_recipe(paths, input_dimension=3)
