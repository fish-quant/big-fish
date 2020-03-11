# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for bigfish.stack.filter submodule.
"""

import pytest

import numpy as np
import bigfish.stack as stack

from bigfish.stack.filter import _define_kernel

from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
