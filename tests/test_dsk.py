from __future__ import print_function, division, absolute_import
from flowly.dsk import apply
from flowly.tz import chained

import itertools as it

from toolz import compose, concat
from toolz.curried import map, mapcat
from dask.bag import from_sequence

import pytest


def test_unknown_func():
    obj = from_sequence(range(10), npartitions=3)

    with pytest.raises(ValueError):
        apply(obj, None)


def test_sum():
    obj = from_sequence(range(10), npartitions=3)

    actual = apply(obj, sum)
    expected = sum(range(10))

    assert actual.compute() == expected


def test_toolz_map():
    obj = from_sequence(range(10), npartitions=3)

    actual = apply(obj, map(lambda x: x + 3))
    expected = range(3, 13)

    assert actual.compute() == expected


def test_toolz_mapcat():
    obj = from_sequence([["a", "b"], ["c", "d", "e"]], npartitions=2)

    actual = apply(obj, mapcat(lambda s: [c.upper() for c in s]))
    expected = ['A', 'B', 'C', 'D', 'E']

    assert actual.compute() == expected


def test_toolz_compose():
    obj = from_sequence([[1, 2, 3], [4, 5, 6], [7, 8, 9]], npartitions=3)

    actual = apply(
        obj,
        compose(sum, it.chain.from_iterable)
    )

    expected = sum(range(1, 10))
    assert actual.compute() == expected



@pytest.mark.parametrize('impl', [
    concat,
    it.chain.from_iterable,
])
def test_concat(impl):
    obj = from_sequence([[1, 2, 3], [4, 5, 6], [7, 8, 9]], npartitions=3)
    actual = apply(obj, impl)
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert actual.compute() == expected


def test_flowly_tz_chained():
    obj = from_sequence([[1, 2, 3], [4, 5, 6], [7, 8, 9]], npartitions=3)

    actual = apply(
        obj,
        chained(it.chain.from_iterable, lambda obj: obj.sum())
    )

    expected = sum(range(1, 10))
    assert actual.compute() == expected


def test_generic_callable():
    obj = from_sequence(range(10), npartitions=3)

    actual = apply(obj, lambda bag: bag.sum())
    expected = sum(range(10))

    assert actual.compute() == expected
