from __future__ import print_function, division, absolute_import
import itertools as it

from toolz import compose, concat, count
from toolz.curried import map, mapcat
import dask.bag as db

from flowly.dsk import apply
from flowly.tz import chained, apply_concat, apply_map_concat, reduction

import pytest


def test_unknown_func():
    obj = db.from_sequence(range(10), npartitions=3)

    with pytest.raises(ValueError):
        apply(obj, None)


def test_sum():
    obj = db.from_sequence(range(10), npartitions=3)

    actual = apply(obj, sum)
    expected = sum(range(10))

    assert actual.compute() == expected


def test_toolz_count():
    obj = db.from_sequence(range(10), npartitions=3)
    assert apply(obj, count).compute() == 10


def test_toolz_map():
    obj = db.from_sequence(range(10), npartitions=3)

    actual = apply(obj, map(lambda x: x + 3))
    expected = range(3, 13)

    assert actual.compute() == expected


def test_toolz_mapcat():
    obj = db.from_sequence([["a", "b"], ["c", "d", "e"]], npartitions=2)

    actual = apply(obj, mapcat(lambda s: [c.upper() for c in s]))
    expected = ['A', 'B', 'C', 'D', 'E']

    assert actual.compute() == expected


def test_toolz_compose():
    obj = db.from_sequence([[1, 2, 3], [4, 5, 6], [7, 8, 9]], npartitions=3)

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
    obj = db.from_sequence([[1, 2, 3], [4, 5, 6], [7, 8, 9]], npartitions=3)
    actual = apply(obj, impl)
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert actual.compute() == expected


def test_flowly_apply_concat__example():
    obj = db.from_sequence([1, 2, 3, 4], npartitions=3)
    transform = apply_concat([
        lambda x: x.map(lambda i: 2 * i),
        lambda x: x.map(lambda i: 3 * i),
    ])

    actual = sorted(apply(obj, transform).compute())
    expected = sorted([2, 4, 6, 8, 3, 6, 9, 12])

    assert actual == expected


def test_flowly_apply_map_concat__example():
    from dask.async import get_sync
    obj = db.from_sequence([1, 2, 3, 4], npartitions=3)

    transform = apply_map_concat([
        lambda x: 2 * x,
        lambda x: 3 * x,
    ])

    actual = sorted(apply(obj, transform).compute(get=get_sync))
    expected = sorted([2, 4, 6, 8, 3, 6, 9, 12])

    assert actual == expected


def test_flowly_tz_reduce():
    obj = db.from_sequence([1, 2, 3, 4, 5, 6, 7, 8, 9], npartitions=3)

    # compute the mean
    transform = reduction(
        lambda l: (sum(l), len(l),),
        lambda items: sum(s for s, _ in items) / max(1, sum(c for _, c in items))
    )

    actual = apply(obj, transform).compute()

    assert actual == 5.0


def test_flowly_tz_chained():
    obj = db.from_sequence([[1, 2, 3], [4, 5, 6], [7, 8, 9]], npartitions=3)

    actual = apply(
        obj,
        chained(it.chain.from_iterable, lambda obj: obj.sum())
    )

    expected = sum(range(1, 10))
    assert actual.compute() == expected


def test_generic_callable():
    obj = db.from_sequence(range(10), npartitions=3)

    actual = apply(obj, lambda bag: bag.sum())
    expected = sum(range(10))

    assert actual.compute() == expected
