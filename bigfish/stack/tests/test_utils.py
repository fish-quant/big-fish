# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.utils module.
"""

import os
import pytest
import tempfile

import bigfish.stack as stack

import numpy as np
import pandas as pd

from bigfish.stack.utils import fit_recipe
from bigfish.stack.utils import get_path_from_recipe
from bigfish.stack.utils import get_nb_element_per_dimension
from bigfish.stack.utils import count_nb_fov


# TODO add test for bigfish.stack.load_and_save_url
# TODO add test for bigfish.stack.check_hash
# TODO add test for bigfish.stack.compute_hash

# ### Test sanity check functions ###

def test_check_parameter():
    # define a function with different parameters to check
    def foo(a, b, c, d, e, f, g, h):
        stack.check_parameter(a=(list, type(None)),
                              b=str,
                              c=int,
                              d=float,
                              e=np.ndarray,
                              f=bool,
                              g=(pd.DataFrame, pd.Series),
                              h=pd.DataFrame)
        return True

    # test the consistency of the check function when it works...
    assert foo(a=[], b="bar", c=5, d=2.5, e=np.array([3, 6, 9]),
               f=True, g=pd.DataFrame(), h=pd.DataFrame())
    assert foo(a=None, b="", c=10, d=2.0, e=np.array([3, 6, 9]),
               f=False, g=pd.Series(), h=pd.DataFrame())

    # ... and when it should raise an error
    with pytest.raises(TypeError):
        foo(a=(), b="bar", c=5, d=2.5, e=np.array([3, 6, 9]),
            f=True, g=pd.DataFrame(), h=pd.DataFrame())
    with pytest.raises(TypeError):
        foo(a=[], b="bar", c=5.0, d=2.5, e=np.array([3, 6, 9]),
            f=True, g=pd.DataFrame(), h=pd.DataFrame())
    with pytest.raises(TypeError):
        foo(a=[], b="bar", c=5, d=2, e=np.array([3, 6, 9]),
            f=True, g=pd.DataFrame(), h=pd.DataFrame())
    with pytest.raises(TypeError):
        foo(a=[], b="bar", c=5, d=2.5, e=[3, 6, 9],
            f=True, g=pd.DataFrame(), h=pd.DataFrame())
    with pytest.raises(TypeError):
        foo(a=[], b="bar", c=5, d=2.5, e=np.zeros((3, 3)),
            f=True, g=pd.DataFrame(), h=pd.Series())


def test_check_df():
    # build a dataframe to test
    df = pd.DataFrame({"A": [3, 6, 9],
                       "B": [2.5, np.nan, 1.3],
                       "C": ["arthur", "florian", "thomas"],
                       "D": [True, True, False]})

    # test the consistency of the check function when it works...
    assert stack.check_df(df,
                          features=["A", "B", "C", "D"],
                          features_without_nan=["A", "C", "D"])
    assert stack.check_df(df,
                          features=["B", "A"],
                          features_without_nan=["C", "D", "A"])
    assert stack.check_df(df,
                          features=None,
                          features_without_nan=["A", "C", "D"])
    assert stack.check_df(df,
                          features=["A", "B", "C", "D"],
                          features_without_nan=None)

    # ... and when it should raise an error
    with pytest.raises(ValueError):
        stack.check_df(df,
                       features=["A", "B", "C", "D", "E"],
                       features_without_nan=["A", "C", "D"])
    with pytest.raises(ValueError):
        stack.check_df(df,
                       features=["A", "B", "C", "D"],
                       features_without_nan=["A", "B", "C", "D"])


def test_check_array():
    # build some arrays to test
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    c = np.array(([1, 2, 3]), dtype=np.float32)
    d = np.array(([1, 2, np.nan]), dtype=np.float32)

    # test number of dimensions
    assert stack.check_array(a, ndim=2)
    assert stack.check_array(b, ndim=[1, 2])
    assert stack.check_array(c, ndim=1)
    with pytest.raises(ValueError):
        stack.check_array(a, ndim=1)

    # test dtypes
    assert stack.check_array(a, dtype=np.float32)
    assert stack.check_array(b, dtype=np.int64)
    assert stack.check_array(c, dtype=[np.float32, np.int64])
    with pytest.raises(TypeError):
        stack.check_array(a, dtype=np.float64)

    # test missing values
    assert stack.check_array(a, allow_nan=False)
    assert stack.check_array(b, allow_nan=True)
    assert stack.check_array(d, allow_nan=True)
    with pytest.raises(ValueError):
        stack.check_array(d, allow_nan=False)


def test_check_range_value():
    # build some arrays to test
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    # test the consistency of the check function when it works...
    assert stack.check_range_value(a, min_=1, max_=None)
    assert stack.check_range_value(a, min_=None, max_=9)

    # ... and when it should raise an error
    with pytest.raises(ValueError):
        stack.check_range_value(a, min_=2, max_=None)
    with pytest.raises(ValueError):
        stack.check_range_value(a, min_=None, max_=8)


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
        assert stack.check_recipe(good_recipe_1, data_directory=None)
        assert stack.check_recipe(good_recipe_1, data_directory=tmp_dir)

        # case with a good recipe but when a file is missing
        good_recipe_2 = {"fov": "fov_1",
                         "c": ["dapi", "smfish", "cellmask"],
                         "opt": "experience_1",
                         "ext": "tif",
                         "pattern": "opt_c_fov.ext"}
        assert stack.check_recipe(good_recipe_2, data_directory=None)
        with pytest.raises(FileNotFoundError):
            stack.check_recipe(good_recipe_2, data_directory=tmp_dir)

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
            stack.check_recipe(bad_recipe_1, data_directory=None)
        with pytest.raises(TypeError):
            stack.check_recipe(bad_recipe_2, data_directory=None)

        # case with a wrong pattern (repetitive key)
        bad_recipe_3 = {"fov": "fov_1",
                        "c": ["dapi", "smfish"],
                        "opt": "experience_1",
                        "ext": "tif",
                        "pattern": "opt_c_fov_fov.ext"}
        with pytest.raises(ValueError):
            stack.check_recipe(bad_recipe_3, data_directory=None)

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
            stack.check_recipe(bad_recipe_4, data_directory=None)
        with pytest.raises(TypeError):
            stack.check_recipe(bad_recipe_5, data_directory=None)


def test_fit_recipe():
    # build a recipe to fit
    good_recipe = {"fov": "fov_1",
                   "c": ["dapi", "smfish"],
                   "opt": "experience_1",
                   "ext": "tif",
                   "pattern": "opt_c_fov.ext"}

    # fit recipe
    new_recipe = fit_recipe(good_recipe)

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
    new_recip_bis = fit_recipe(new_recipe)
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
        path_dapi = get_path_from_recipe(good_recipe, tmp_dir, c=0)
        assert os.path.isfile(path_dapi)


def test_element_per_dimension():
    # build a recipe to test
    good_recipe = {"fov": "fov_1",
                   "c": ["dapi", "smfish"],
                   "opt": "experience_1",
                   "ext": "tif",
                   "pattern": "opt_c_fov.ext"}

    # test the number of elements to be stacked
    nb_r, nb_c, nb_z = get_nb_element_per_dimension(good_recipe)
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
    nb_fov = count_nb_fov(good_recipe_1)
    assert nb_fov == 1

    # case when 'fov' key is a list
    good_recipe_2 = {"fov": ["fov_1", "fov_2"],
                     "c": ["dapi", "smfish"],
                     "opt": "experience_1",
                     "ext": "tif",
                     "pattern": "opt_c_fov.ext"}
    nb_fov = count_nb_fov(good_recipe_2)
    assert nb_fov == 2

    # case when 'fov' key does not exist
    good_recipe_3 = {"c": ["dapi", "smfish"],
                     "opt": "experience_1",
                     "ext": "tif",
                     "pattern": "opt_c_fov.ext"}
    nb_fov = count_nb_fov(good_recipe_3)
    assert nb_fov == 1

    # case when the 'fov' key is not a string or a list
    with pytest.raises(TypeError):
        bad_recipe = {"fov": 1,
                      "c": ["dapi", "smfish"],
                      "opt": "experience_1",
                      "ext": "tif",
                      "pattern": "opt_c_fov.ext"}
        count_nb_fov(bad_recipe)


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
        assert stack.check_datamap(datamap)
        datamap = [[recipe, tmp_dir]]
        assert stack.check_datamap(datamap)
        datamap = [(None, tmp_dir)]
        with pytest.raises(TypeError):
            stack.check_datamap(datamap)
        datamap = [(recipe, 3)]
        with pytest.raises(TypeError):
            stack.check_datamap(datamap)
        datamap = [(recipe, "/foo/bar")]
        with pytest.raises(NotADirectoryError):
            stack.check_datamap(datamap)
        datamap = [(recipe, tmp_dir, None)]
        with pytest.raises(ValueError):
            stack.check_datamap(datamap)


# ### Constants ###

def test_margin_value():
    # test margin value
    assert stack.get_margin_value() >= 2


def test_epsilon_float_32():
    # test epsilon value and dtype
    eps = stack.get_eps_float32()
    assert eps < 1e-5
    assert isinstance(eps, np.float32)
