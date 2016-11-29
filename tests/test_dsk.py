from __future__ import print_function, division, absolute_import
import itertools as it
import operator as op

from toolz import compose, concat, count, unique
from toolz.curried import (
    filter,
    map,
    mapcat,
    pluck,
    random_sample,
    remove,
    take,
    topk,
)
import dask.bag as db
from dask.delayed import delayed

from flowly.dsk import apply, item_from_object
from flowly.tz import (
    apply_concat,
    apply_map_concat,
    chained,
    frequencies,
    reduction,
    seq,
)

import pytest


def test_unknown_func():
    obj = db.from_sequence(range(10), npartitions=3)

    with pytest.raises(ValueError):
        apply(obj, None)


@pytest.mark.parametrize('input,output', [
    ([True, True, False, True, False], False),
    ([True, True, True, True, True], True),
    ([False, False, False, False, False], False),
])
def test_all(input, output):
    assert apply(db.from_sequence(input, npartitions=3), all).compute() is output


@pytest.mark.parametrize('input,output', [
    ([True, True, False, True, False], True),
    ([True, True, True, True, True], True),
    ([False, False, False, False, False], False),
])
def test_any(input, output):
    assert apply(db.from_sequence(input, npartitions=3), any).compute() is output


def test_len():
    obj = db.from_sequence(range(10), npartitions=3)
    assert apply(obj, len).compute() == 10


def test_max():
    obj = db.from_sequence(range(10), npartitions=3)
    assert apply(obj, max).compute() == 9


def test_min():
    obj = db.from_sequence(range(10), npartitions=3)
    assert apply(obj, min).compute() == 0


def test_sum():
    obj = db.from_sequence(range(10), npartitions=3)

    actual = apply(obj, sum)
    expected = sum(range(10))

    assert actual.compute() == expected


def test_toolz_pluck():
    obj = db.from_sequence([
        {'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6},
        {'a': 7, 'b': 8}, {'a': 9, 'b': 0}
    ], npartitions=3)

    actual = apply(obj, pluck('a'))
    assert actual.compute() == [1, 3, 5, 7, 9]


def test_toolz_pluck_multiple():
    obj = db.from_sequence([
        {'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6},
        {'a': 7, 'b': 8}, {'a': 9, 'b': 0}
    ], npartitions=3)

    actual = apply(obj, pluck(['a', 'b']))
    assert actual.compute() == [(1, 2), (3, 4), (5, 6), (7, 8), (9, 0)]


def test_toolz_pluck_default():
    obj = db.from_sequence([
        {'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6},
        {'a': 7, 'b': 8}, {'a': 9, 'c': 0}
    ], npartitions=3)

    actual = apply(obj, pluck(['a', 'b'], default=-1))
    assert actual.compute() == [(1, 2), (3, 4), (5, 6), (7, 8), (9, -1)]


def test_toolz_count():
    obj = db.from_sequence(range(10), npartitions=3)
    assert apply(obj, count).compute() == 10


def test_flowly_tz_frequencies():
    obj = db.from_sequence([True, False, True, False, False], npartitions=3)
    assert sorted(apply(obj, frequencies).compute()) == sorted([(True, 2), (False, 3)])


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


def test_toolz_random_sample():
    obj = db.from_sequence(range(10), npartitions=3)
    apply(obj, random_sample(0.51)).compute()


def test_toolz_random_sample__random_state():
    # just test that it does not raise an exception
    obj = db.from_sequence(range(10), npartitions=3)
    apply(obj, random_sample(0.51, random_state=5)).compute()


def test_toolz_filter():
    obj = db.from_sequence(range(10), npartitions=3)
    actual = apply(obj, filter(lambda x: x % 2 == 0))

    assert actual.compute() == [0, 2, 4, 6, 8]


def test_toolz_remove():
    obj = db.from_sequence(range(10), npartitions=3)
    actual = apply(obj, remove(lambda x: x % 2 == 1))

    assert actual.compute() == [0, 2, 4, 6, 8]


def test_toolz_take():
    obj = db.from_sequence(range(10), npartitions=3)
    actual = apply(obj, take(5))

    assert actual.compute() == [0, 1, 2, 3, 4]


def test_toolz_topk():
    obj = db.from_sequence(range(100), npartitions=3)
    actual = apply(obj, topk(5))

    assert sorted(actual.compute()) == sorted([99, 98, 97, 96, 95])


def test_toolz_topk__key():
    obj = db.from_sequence(range(100), npartitions=3)
    actual = apply(obj, topk(5, key=lambda i: -i))

    assert sorted(actual.compute()) == sorted([0, 1, 2, 3, 4])


def test_toolz_compose():
    obj = db.from_sequence([[1, 2, 3], [4, 5, 6], [7, 8, 9]], npartitions=3)

    actual = apply(
        obj,
        compose(sum, it.chain.from_iterable)
    )

    expected = sum(range(1, 10))
    assert actual.compute() == expected


def test_toolz_unique():
    obj = db.from_sequence((1, 2, 1, 3), npartitions=3)
    actual = apply(obj, unique)

    assert sorted(actual.compute()) == [1, 2, 3]


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


def test_flowly_tz_seq():
    obj = item_from_object(42)
    actual = apply(obj, seq)

    assert actual.compute() == [42]


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
