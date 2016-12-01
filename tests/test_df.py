from __future__ import print_function, division, absolute_import

from flowly import this
import flowly.df as fdf

import pandas as pd
import pandas.util.testing as pdt

from toolz import pipe


def test_filter__example():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [4, 3, 2, 1],
    })

    actual = pipe(
        df,
        fdf.filter(this.a % 2 == 0, this.a < this.b)
    )

    expected = pd.DataFrame({'a': [2], 'b': [3]}, index=[1])
    pdt.assert_frame_equal(actual, expected)


def test_filter__single_condition():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [4, 3, 2, 1],
    })

    actual = pipe(
        df,
        fdf.filter(this.a % 2 == 0)
    )

    expected = pd.DataFrame({'a': [2, 4], 'b': [3, 1]}, index=[1, 3])
    pdt.assert_frame_equal(actual, expected)


def test_filter__callable():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [4, 3, 2, 1],
    })

    actual = pipe(
        df,
        fdf.filter(lambda df: df['a'] % 2 == 0)
    )

    expected = pd.DataFrame({'a': [2, 4], 'b': [3, 1]}, index=[1, 3])
    pdt.assert_frame_equal(actual, expected)


def test_filter__pass_throught():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [4, 3, 2, 1],
    })

    assert pipe(df, fdf.filter()) is df


def test_drop_index():
    df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=[4, 3, 2, 1])
    actual = pipe(df, fdf.drop_index)
    expected = pd.DataFrame({'a': [1, 2, 3, 4]})

    pdt.assert_frame_equal(actual, expected)


def test_drop_columns():
    def _result_columns(*exprs):
        df = pd.DataFrame(
            {'d': [1], 'b': [2], 'a': [3], 'c': [4]},
            columns=['d', 'b', 'a', 'c'],
        )
        result = pipe(df, fdf.drop_columns(*exprs))
        return list(result.columns)

    assert _result_columns('a') == ['d', 'b', 'c']
    assert _result_columns('a', 'c') == ['d', 'b']
    assert _result_columns(this.d, this['b']) == ['a', 'c']
