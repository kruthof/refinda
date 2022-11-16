from pandas._testing import assert_series_equal, assert_frame_equal
import pytest
import numpy as np
from refinda.helper.statistics import *
from numpy.testing import assert_allclose
import pandas as pd
import scipy


@pytest.fixture
def get_equal_data():
    np.random.seed(123)
    x = np.random.randint(0,100,10)
    np.random.seed(123)
    y = np.random.randint(0,100,10)
    return [x,y]

@pytest.fixture
def get_non_equal_data():
    np.random.seed(432)
    x = np.random.randint(0,100,10)
    np.random.seed(123)
    y = np.random.randint(0,100,10)
    return [x,y]

def test_z_value_zero():
    data1,data2 = get_equal_data()
    z_value = zvalue_sharp(data1,data2)
    assert z_value == 0.0


def test_z_value_non_zero():
    data1,data2 = get_non_equal_data()
    z_value = zvalue_sharp(data1,data2)
    assert z_value == pytest.approx(-0.7903370505139574)