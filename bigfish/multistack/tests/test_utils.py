# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.utils module.
"""

import os
import pytest
import tempfile

import bigfish.multistack as multistack


# ### Test recipes ###

def test_check_recipe():
    # build a temporary directory with two files
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "experience_1_dapi_fov_1.tif")
        with open(path, 'w') as f:
            f.write("dapi file")
        path = os.path.join(tmp_dir, "experience_1_smfish_fov_1.tif")
        with open(path, 'w') as f:
            f.write("smFISH file")

        # test the consistency of the check function when it should work
        good_recipe_1 = {"fov": "fov_1",
                         "c": ["dapi", "smfish"],
                         "opt": "experience_1",
                         "ext": "tif",
                         "pattern": "opt_c_fov.ext"}
        assert multistack.check_recipe(good_recipe_1, data_directory=None)
        assert multistack.check_recipe(good_recipe_1, data_directory=tmp_dir)

        # case with a good recipe but when a file is missing
        good_recipe_2 = {"fov": "fov_1",
                         "c": ["dapi", "smfish", "cellmask"],
                         "opt": "experience_1",
                         "ext": "tif",
                         "pattern": "opt_c_fov.ext"}
        assert multistack.check_recipe(good_recipe_2, data_directory=None)
        with pytest.raises(FileNotFoundError):
            multistack.check_recipe(good_recipe_2, data_directory=tmp_dir)

        # cases without a 'pattern' key with a string value
        bad_recipe_1 = {"fov": "fov_1",
                        "c": ["dapi", "smfish"],
                        "opt": "experience_1",
                        "ext": "tif"}
        bad_recipe_2 = {"fov": "fov_1",
                        "c": ["dapi", "smfish"],
                        "opt": "experience_1",
                        "ext": "tif",
                        "pattern": ["opt_c_fov.ext"]}
        with pytest.raises(KeyError):
            multistack.check_recipe(bad_recipe_1, data_directory=None)
        with pytest.raises(TypeError):
            multistack.check_recipe(bad_recipe_2, data_directory=None)

        # case with a wrong pattern (repetitive key)
        bad_recipe_3 = {"fov": "fov_1",
                        "c": ["dapi", "smfish"],
                        "opt": "experience_1",
                        "ext": "tif",
                        "pattern": "opt_c_fov_fov.ext"}
        with pytest.raises(ValueError):
            multistack.check_recipe(bad_recipe_3, data_directory=None)

        # case with wrong key or value
        bad_recipe_4 = {"fov": "fov_1",
                        "channel": ["dapi", "smfish"],
                        "optional": "experience_1",
                        "extension": "tif",
                        "pattern": "opt_c_fov.ext"}
        bad_recipe_5 = {"fov": "fov_1",
                        "c": ["dapi", "smfish"],
                        "opt": 1,
                        "ext": "tif",
                        "pattern": "opt_c_fov.ext"}
        with pytest.raises(KeyError):
            multistack.check_recipe(bad_recipe_4, data_directory=None)
        with pytest.raises(TypeError):
            multistack.check_recipe(bad_recipe_5, data_directory=None)


def test_fit_recipe():
    # build a recipe to fit
    good_recipe = {"fov": "fov_1",
                   "c": ["dapi", "smfish"],
                   "opt": "experience_1",
                   "ext": "tif",
                   "pattern": "opt_c_fov.ext"}

    # fit recipe
    new_recipe = multistack.fit_recipe(good_recipe)

    # all keys should be initialized in the new recipe, with a list or a string
    for key in ['fov', 'r', 'c', 'z']:
        assert key in new_recipe
        assert isinstance(new_recipe[key], list)
    for key in ['ext', 'opt']:
        assert key in new_recipe
        assert isinstance(new_recipe[key], str)
    assert 'pattern' in new_recipe
    assert isinstance(new_recipe['pattern'], str)

    # test that fitting an already fitted recipe does not change anything
    new_recip_bis = multistack.fit_recipe(new_recipe)
    assert new_recip_bis == new_recipe


def test_path_from_recipe():
    # build a temporary directory with one file
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "experience_1_dapi_fov_1.tif")
        with open(path, 'w') as f:
            f.write("dapi file")

        # build a recipe to read the file
        good_recipe = {"fov": "fov_1",
                       "c": "dapi",
                       "opt": "experience_1",
                       "ext": "tif",
                       "pattern": "opt_c_fov.ext"}

        # test the path
        path_dapi = multistack.get_path_from_recipe(good_recipe, tmp_dir, c=0)
        assert os.path.isfile(path_dapi)


def test_element_per_dimension():
    # build a recipe to test
    good_recipe = {"fov": "fov_1",
                   "c": ["dapi", "smfish"],
                   "opt": "experience_1",
                   "ext": "tif",
                   "pattern": "opt_c_fov.ext"}

    # test the number of elements to be stacked
    nb_r, nb_c, nb_z = multistack.get_nb_element_per_dimension(good_recipe)
    assert nb_r == 1
    assert nb_c == 2
    assert nb_z == 1


def test_nb_fov():
    # case when 'fov' key is a string
    good_recipe_1 = {"fov": "fov_1",
                     "c": ["dapi", "smfish"],
                     "opt": "experience_1",
                     "ext": "tif",
                     "pattern": "opt_c_fov.ext"}
    nb_fov = multistack.count_nb_fov(good_recipe_1)
    assert nb_fov == 1

    # case when 'fov' key is a list
    good_recipe_2 = {"fov": ["fov_1", "fov_2"],
                     "c": ["dapi", "smfish"],
                     "opt": "experience_1",
                     "ext": "tif",
                     "pattern": "opt_c_fov.ext"}
    nb_fov = multistack.count_nb_fov(good_recipe_2)
    assert nb_fov == 2

    # case when 'fov' key does not exist
    good_recipe_3 = {"c": ["dapi", "smfish"],
                     "opt": "experience_1",
                     "ext": "tif",
                     "pattern": "opt_c_fov.ext"}
    nb_fov = multistack.count_nb_fov(good_recipe_3)
    assert nb_fov == 1

    # case when the 'fov' key is not a string or a list
    with pytest.raises(TypeError):
        bad_recipe = {"fov": 1,
                      "c": ["dapi", "smfish"],
                      "opt": "experience_1",
                      "ext": "tif",
                      "pattern": "opt_c_fov.ext"}
        multistack.count_nb_fov(bad_recipe)


def test_check_datamap():
    # build a temporary directory with two files
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "experience_1_dapi_fov_1.tif")
        with open(path, 'w') as f:
            f.write("dapi file")
        path = os.path.join(tmp_dir, "experience_1_smfish_fov_1.tif")
        with open(path, 'w') as f:
            f.write("smFISH file")

        # test the consistency of the check function
        recipe = {"fov": "fov_1",
                  "c": ["dapi", "smfish"],
                  "opt": "experience_1",
                  "ext": "tif",
                  "pattern": "opt_c_fov.ext"}
        datamap = [(recipe, tmp_dir)]
        assert multistack.check_datamap(datamap)
        datamap = [[recipe, tmp_dir]]
        assert multistack.check_datamap(datamap)
        datamap = [(None, tmp_dir)]
        with pytest.raises(TypeError):
            multistack.check_datamap(datamap)
        datamap = [(recipe, 3)]
        with pytest.raises(TypeError):
            multistack.check_datamap(datamap)
        datamap = [(recipe, "/foo/bar")]
        with pytest.raises(NotADirectoryError):
            multistack.check_datamap(datamap)
        datamap = [(recipe, tmp_dir, None)]
        with pytest.raises(ValueError):
            multistack.check_datamap(datamap)
