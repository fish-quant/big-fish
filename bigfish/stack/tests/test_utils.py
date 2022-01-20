# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.utils module.
"""

import pytest

import bigfish.stack as stack

import numpy as np
import pandas as pd


# TODO add test for bigfish.stack.load_and_save_url
# TODO add test for bigfish.stack.check_hash
# TODO add test for bigfish.stack.compute_hash
# TODO add test for bigfish.stack.check_input_data
# TODO add test for bigfish.stack.moving_average
# TODO add test for bigfish.stack.centered_moving_average

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


# ### Constants ###

def test_margin_value():
    # test margin value
    assert stack.get_margin_value() >= 2


def test_epsilon_float_32():
    # test epsilon value and dtype
    eps = stack.get_eps_float32()
    assert eps < 1e-5
    assert isinstance(eps, np.float32)
