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
    reduce,
    remove,
    take,
    topk,
)
import dask.bag as db
from dask.delayed import delayed

from flowly.dsk import apply, apply_to_local, item_from_object, dask_dict
from flowly.tz import (
    apply_concat,
    apply_map_concat,
    build_dict,
    chained,
    collect,
    frequencies,
    groupby,
    itemsetter,
    kv_transform,
    kv_keymap,
    kv_valmap,
    kv_reduceby,
    kv_reductionby,
    reduceby,
    reduction,
    seq,
)

import pytest

executors = [
    lambda transform, obj, npartitions=None: transform(obj),
    apply_to_local,
]


def test_apply_error():
    # check that the additional print statements works
    # invalid rule object (no match / apply methods)
    rules = [None]

    with pytest.raises(AttributeError):
        apply(None, None, rules)


def test_unknown_func():
    obj = db.from_sequence(range(10), npartitions=3)

    with pytest.raises(ValueError):
        apply(obj, None)


@pytest.mark.parametrize('input,output', [
    ([True, True, False, True, False], False),
    ([True, True, True, True, True], True),
    ([False, False, False, False, False], False),
])
@pytest.mark.parametrize('executor', executors)
def test_all(executor, input, output):
    assert executor(all, input, npartitions=3) is output


@pytest.mark.parametrize('input,output', [
    ([True, True, False, True, False], True),
    ([True, True, True, True, True], True),
    ([False, False, False, False, False], False),
])
@pytest.mark.parametrize('executor', executors)
def test_any(executor, input, output):
    assert executor(any, input, npartitions=3) is output


@pytest.mark.parametrize('executor', executors)
def test_len(executor):
    assert executor(len, range(10), npartitions=3) == 10


@pytest.mark.parametrize('executor', executors)
def test_max(executor):
    assert executor(max, range(10), npartitions=3) == 9


@pytest.mark.parametrize('executor', executors)
def test_min(executor):
    assert executor(min, range(10), npartitions=3) == 0


@pytest.mark.parametrize('executor', executors)
def test_sum(executor):
    assert executor(sum, range(10), npartitions=3) == sum(range(10))


@pytest.mark.parametrize('executor', executors)
def test_toolz_pluck(executor):
    actual = executor(
        pluck('a'),
        [
            {'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6},
            {'a': 7, 'b': 8}, {'a': 9, 'b': 0}
        ],
        npartitions=3,
    )

    assert list(actual) == [1, 3, 5, 7, 9]


@pytest.mark.parametrize('executor', executors)
def test_toolz_pluck_multiple(executor):
    actual = executor(
        pluck(['a', 'b']),
        [
            {'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6},
            {'a': 7, 'b': 8}, {'a': 9, 'b': 0}
        ],
        npartitions=3,
    )

    assert list(actual) == [(1, 2), (3, 4), (5, 6), (7, 8), (9, 0)]


@pytest.mark.parametrize('executor', executors)
def test_toolz_pluck_default(executor):
    actual = executor(
        pluck(['a', 'b'], default=-1),
        [
            {'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6},
            {'a': 7, 'b': 8}, {'a': 9, 'c': 0}
        ],
        npartitions=3,
    )

    assert list(actual) == [(1, 2), (3, 4), (5, 6), (7, 8), (9, -1)]


@pytest.mark.parametrize('executor', executors)
def test_toolz_count(executor):
    assert executor(count, range(10), npartitions=3) == 10


@pytest.mark.parametrize('executor', executors)
def test_flowly_tz_frequencies(executor):
    actual = executor(frequencies, [True, False, True, False, False], npartitions=3)
    assert sorted(actual) == sorted([(True, 2), (False, 3)])


@pytest.mark.parametrize('executor', executors)
def test_toolz_map(executor):
    actual = executor(map(lambda x: x + 3), range(10), npartitions=3)
    assert list(actual) == list(range(3, 13))


@pytest.mark.parametrize('executor', executors)
def test_toolz_mapcat(executor):
    actual = executor(
        mapcat(lambda s: s.upper()), ["ab", "cde"], npartitions=2,
    )
    expected = ['A', 'B', 'C', 'D', 'E']

    assert list(actual) == expected


@pytest.mark.parametrize('executor', executors)
def test_toolz_random_sample(executor):
    executor(random_sample(0.51), range(10), npartitions=3)


@pytest.mark.parametrize('executor', executors)
def test_toolz_random_sample__random_state(executor):
    # just test that it does not raise an exception
    executor(random_sample(0.51, random_state=5), range(10), npartitions=3)


@pytest.mark.parametrize('executor', executors)
def test_toolz_filter(executor):
    actual = executor(filter(lambda x: x % 2 == 0), range(10), npartitions=3)
    assert list(actual) == [0, 2, 4, 6, 8]


@pytest.mark.parametrize('executor', executors)
def test_toolz_reduce(executor):
    assert executor(reduce(op.add), range(100), npartitions=5) == sum(range(100))


@pytest.mark.parametrize('executor', executors)
def test_toolz_remove(executor):
    actual = executor(remove(lambda x: x % 2 == 1), range(10), npartitions=3)
    assert list(actual) == [0, 2, 4, 6, 8]


@pytest.mark.parametrize('executor', executors)
def test_toolz_take(executor):
    actual = executor(take(5), range(10), npartitions=3)
    assert list(actual) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize('executor', executors)
def test_toolz_topk(executor):
    actual = executor(topk(5), range(100), npartitions=3)
    assert sorted(actual) == sorted([99, 98, 97, 96, 95])


@pytest.mark.parametrize('executor', executors)
def test_toolz_topk__key(executor):
    actual = executor(topk(5, key=lambda i: -i), range(100), npartitions=3)
    assert sorted(actual) == sorted([0, 1, 2, 3, 4])


@pytest.mark.parametrize('executor', executors)
def test_toolz_compose(executor):
    actual = executor(
        compose(sum, it.chain.from_iterable),
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        npartitions=3,
    )

    assert actual == sum(range(1, 10))


@pytest.mark.parametrize('executor', executors)
def test_toolz_unique(executor):
    actual = executor(unique, (1, 2, 1, 3), npartitions=3)
    assert sorted(actual) == [1, 2, 3]


@pytest.mark.parametrize('impl', [
    concat,
    it.chain.from_iterable,
])
@pytest.mark.parametrize('executor', executors)
def test_concat(executor, impl):
    actual = executor(
        impl,
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        npartitions=3,
    )
    assert list(actual) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize('executor', executors)
def test_flowly_apply_concat__example(executor):
    transform = apply_concat([
        map(lambda i: 2 * i),
        map(lambda i: 3 * i),
    ])

    actual = executor(transform, [1, 2, 3, 4], npartitions=3)
    assert sorted(actual) == sorted([2, 4, 6, 8, 3, 6, 9, 12])


@pytest.mark.parametrize('executor', executors)
def test_flowly_apply_map_concat__example(executor):
    transform = apply_map_concat([
        lambda x: 2 * x,
        lambda x: 3 * x,
    ])

    actual = executor(transform, [1, 2, 3, 4], npartitions=3)
    assert sorted(actual) == sorted([2, 4, 6, 8, 3, 6, 9, 12])


@pytest.mark.parametrize('executor', executors)
def test_flowly_tz_build_dict(executor):
    transform = build_dict(max=max, min=min, sum=sum)
    actual = executor(transform, [1, 2, 3, 4], npartitions=3)

    assert dict(actual) == dict(min=1, max=4, sum=10)


@pytest.mark.parametrize('executor', executors)
def test_flowly_tz_build_dict__alternative(executor):
    transform = build_dict(dict(max=max), dict(min=min), sum=sum)
    actual = executor(transform, [1, 2, 3, 4], npartitions=3)
    assert dict(actual) == dict(min=1, max=4, sum=10)


@pytest.mark.parametrize('executor', executors)
def test_flowly_tz_groupby(executor):
    result = executor(
        groupby(lambda i: i % 2),
        [1, 2, 3, 4, 5, 6, 7],
        npartitions=3,
    )
    actual = sorted(kv_valmap(sorted)(result))
    assert actual == sorted([(1, [1, 3, 5, 7]), (0, [2, 4, 6])])


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_transform__chained(executor):
    actual = executor(
        kv_transform(
            chained(
                map(lambda i: 2 * i),
                map(lambda i: 5 * i),
            ),
        ),
        [(i % 2, i) for i in range(20)],
        npartitions=10,
    )
    assert sorted(actual) == sorted([(i % 2, 10 * i) for i in range(20)])


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_transform__map(executor):
    actual = executor(
        kv_transform(map(lambda i: 10 * i)),
        [(i % 2, i) for i in range(20)],
        npartitions=10,
    )
    assert sorted(actual) == sorted([(i % 2, 10 * i) for i in range(20)])


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_transform__mapcat(executor):
    actual = executor(
        kv_transform(mapcat(lambda i: [10 * i, 20 * i])),
        [(i % 2, i) for i in range(20)],
        npartitions=10,
    )
    assert sorted(actual) == sorted(it.chain(
        [(i % 2, 10 * i) for i in range(20)],
        [(i % 2, 20 * i) for i in range(20)]
    ))


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_transform__reduce(executor):
    actual = executor(
        kv_transform(reduce(lambda a, b: a + b)),
        [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]],
        npartitions=3,
    )

    assert sorted(actual) == sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_transform__reduction(executor):
    actual = executor(
        kv_transform(reduction(None, sum)),
        [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]],
        npartitions=3,
    )

    assert sorted(actual) == sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_transform__concat(executor):
    actual = executor(
        kv_transform(concat),
        [(i % 2, [1 * i, 2 * i, 3 * i]) for i in [1, 2, 3, 4, 5, 6, 7]],
        npartitions=3,
    )

    assert sorted(actual) == sorted([
        (i % 2, factor * i)
        for i in [1, 2, 3, 4, 5, 6, 7]
        for factor in [1, 2, 3]
    ])


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_transform__collect(executor):
    actual = executor(
        kv_transform(
            chained(
                collect,
                map(sorted)
            )
        ),
        [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]],
        npartitions=3,
    )

    assert sorted(actual) == sorted([
        (0, [2, 4, 6]),
        (1, [1, 3, 5, 7]),
    ])


def test_flowly_kv_transform__unknown_function():
    with pytest.raises(ValueError):
        kv_transform(lambda foo: foo)


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_keymap(executor):
    actual = executor(
        kv_keymap(lambda i: 10 * i),
        [(i % 2, i) for i in range(20)],
        npartitions=10,
    )
    assert sorted(actual) == sorted([(10 * (i % 2), i) for i in range(20)])


@pytest.mark.parametrize('executor', executors)
def test_flowly_kv_valmap(executor):
    actual = executor(
        kv_valmap(lambda i: 10 * i),
        [(i % 2, i) for i in range(20)],
        npartitions=10,
    )
    assert sorted(actual) == sorted([(i % 2, 10 * i) for i in range(20)])


@pytest.mark.parametrize('executor', executors)
def test_kv_reduceby(executor):
    actual = executor(
        kv_reduceby(lambda a, b: a + b),
        [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]],
        npartitions=3,
    )

    assert sorted(actual) == sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])


@pytest.mark.parametrize('executor', executors)
def test_kv_reductionby__no_perpartition(executor):
    actual = executor(
        kv_reductionby(None, sum),
        [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]],
        npartitions=3,
    )

    assert sorted(actual) == sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])


@pytest.mark.parametrize('executor', executors)
def test_kv_reductionby__with_perpartition(executor):
    actual = executor(
        kv_reductionby(
            lambda x: x,
            lambda parts: sum(i for part in parts for i in part),
        ),
        [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]],
        npartitions=3,
    )

    assert sorted(actual) == sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])


def test_flowly_tz_update_dict():
    obj = dict(l=db.from_sequence([1, 2, 3, 4], npartitions=3))

    transform = itemsetter(
        # varargs are also allowed
        dict(max=chained(op.itemgetter('l'), max)),
        min=chained(op.itemgetter('l'), min),
        sum=chained(op.itemgetter('l'), sum),
    )

    actual = apply(transform, obj).compute()

    assert actual == dict(l=[1, 2, 3, 4], min=1, max=4, sum=10)


@pytest.mark.parametrize('executor', executors)
def test_flowly_tz_reduceby(executor):
    actual = executor(
        reduceby(lambda i: i % 2, lambda a, b: a + b),
        [1, 2, 3, 4, 5, 6, 7],
        npartitions=3,
    )

    assert sorted(actual) == sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])


@pytest.mark.parametrize('executor', executors)
def test_flowly_tz_reduce(executor):
    # compute the mean
    transform = reduction(
        lambda l: (sum(l), len(l),),
        lambda items: sum(s for s, _ in items) / max(1, sum(c for _, c in items))
    )

    assert executor(transform, [1, 2, 3, 4, 5, 6, 7, 8, 9], npartitions=3) == 5.0


def test_flowly_tz_seq():
    obj = item_from_object(42)
    actual = apply(seq, obj)

    assert actual.compute() == [42]


@pytest.mark.parametrize('executor', executors)
def test_flowly_tz_chained(executor):
    actual = executor(
        chained(it.chain.from_iterable, sum),
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        npartitions=3
    )

    assert actual == sum(range(1, 10))


def test_generic_callable():
    obj = db.from_sequence(range(10), npartitions=3)

    actual = apply(lambda bag: bag.sum(), obj)
    expected = sum(range(10))

    assert actual.compute() == expected


def test_dsk_dict__copy():
    d = dask_dict(a=delayed(42), b=delayed(13)).copy().compute()
    assert d == dict(a=42, b=13)


def test_toolz_groupby__is_not_supported():
    from toolz.curried import groupby

    transform = groupby(lambda i: i % 2)
    data = db.from_sequence([1, 2, 3, 4, 5, 6, 7], npartitions=3)

    with pytest.raises(ValueError):
        apply(transform, data).compute()


def test_toolz_reduceby__is_not_supported():
    from toolz.curried import reduceby

    transform = reduceby(lambda i: i % 2, lambda a, b: a + b)
    data = db.from_sequence([1, 2, 3, 4, 5, 6, 7], npartitions=3)

    with pytest.raises(ValueError):
        apply(transform, data).compute()
