from __future__ import print_function, division, absolute_import

from dfply import transmute, X
import pandas as pd
import pandas.util.testing as pdt
from toolz.curried import map

from flowly.tz import pipe
from flowly.df import add_dfply_support, as_frame


def test_as_frame():
    actual = as_frame(foo=[1, 2, 3], bar=[4, 5, 6])
    expected = pd.DataFrame({
        'foo': [1, 2, 3],
        'bar': [4, 5, 6],
    })
    pdt.assert_frame_equal(actual, expected)


def test_dfply__example():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    actual = pipe(
        df,
        transmute(c=X.a + X.b),
        rewrites=[add_dfply_support]
    )
    expected = pd.DataFrame({
        'c': [5, 7, 9]
    })

    pdt.assert_frame_equal(actual, expected)


def test_dfply__map():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    actual = pipe(
        [df],
        map(transmute(c=X.a + X.b)),
        list,
        rewrites=[add_dfply_support]
    )
    expected = pd.DataFrame({
        'c': [5, 7, 9]
    })

    assert len(actual) == 1
    pdt.assert_frame_equal(actual[0], expected)
