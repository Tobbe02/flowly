from __future__ import print_function, division, absolute_import

import operator as op

import pytest

from flowly.tz import (
    apply_concat,
    apply_map_concat,
    build_dict,
    chained,
    frequencies,
    groupby,
    itemsetter,
    kv_keymap,
    kv_reduceby,
    kv_reductionby,
    kv_valmap,
    optional,
    raise_,
    reduceby,
    reduction,
    seq,
    show,
    try_call,
)


def test_chained__example():
    transform = chained(
        lambda a: 2 * a,
        lambda a: a - 3
    )

    assert transform(5) == 7


def test_chained__composition():
    transform = chained(lambda a: a * 2) + chained(lambda a: a - 3)
    assert transform(5) == 7


def test_apply_concat__example():
    transform = apply_concat([
        lambda x: [2 * i for i in x],
        lambda x: [3 * i for i in x],
    ])

    actual = sorted(transform([1, 2, 3, 4]))
    expected = sorted([2, 4, 6, 8, 3, 6, 9, 12])

    assert actual == expected


def test_apply_map_concat__example():
    transform = apply_map_concat([
        lambda x: 2 * x,
        lambda x: 3 * x,
    ])

    actual = sorted(transform([1, 2, 3, 4]))
    expected = sorted([2, 4, 6, 8, 3, 6, 9, 12])

    assert actual == expected


def test_build_dict():
    transform = build_dict(
        dict(a=lambda x: 2 * x),
        b=lambda x: 3 * x,
    )

    assert transform(4) == dict(a=8, b=12)


def test_update_dict():
    transform = itemsetter(
        dict(a=chained(op.itemgetter('i'), lambda x: 2 * x)),
        b=chained(op.itemgetter('i'), lambda x: 3 * x),
    )

    assert transform(dict(i=4)) == dict(i=4, a=8, b=12)


def test_frequencies():
    actual = sorted(frequencies([1, 1, 2, 2, 2]))
    expected = sorted([(1, 2), (2, 3)])
    assert actual == expected


def test_groupby():
    transform = groupby(lambda i: i % 2)
    actual = sorted(transform([1, 2, 3, 4, 5, 6, 7]))
    expected = sorted([(1, [1, 3, 5, 7]), (0, [2, 4, 6])])

    assert actual == expected


def test_reduceby():
    seq = [1, 2, 3, 4, 5, 6, 7]
    transform = reduceby(lambda i: i % 2, lambda a, b: a + b)
    actual = sorted(transform(seq))
    expected = sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])

    assert actual == expected


def test_kv_keymap():
    seq = [(i % 2, i) for i in [1, 2, 3, 4]]
    transform = kv_keymap(lambda k: 10 * k)
    actual = sorted(transform(seq))
    expected = sorted([(10, 1), (0, 2), (10, 3), (0, 4)])

    assert actual == expected


def test_kv_reduceby():
    seq = [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]]
    transform = kv_reduceby(lambda a, b: a + b)
    actual = sorted(transform(seq))
    expected = sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])

    assert actual == expected


def test_kv_valmap():
    seq = [(i % 2, i) for i in [1, 2, 3, 4]]
    transform = kv_valmap(lambda k: 10 * k)
    actual = sorted(transform(seq))
    expected = sorted([(1, 10), (0, 20), (1, 30), (0, 40)])

    assert actual == expected


def test_kv_reductionby__no_perpartition():
    seq = [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]]
    transform = kv_reductionby(None, sum)
    actual = sorted(transform(seq))
    expected = sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])

    assert actual == expected


def test_kv_reductionby__with_perpartition():
    seq = [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]]
    transform = kv_reductionby(
        lambda x: x,
        lambda parts: sum(i for part in parts for i in part),
    )
    actual = sorted(transform(seq))
    expected = sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])

    assert actual == expected


def test_reduction():
    obj = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # compute the mean
    transform = reduction(
        lambda l: (sum(l), len(l),),
        lambda items: sum(s for s, _ in items) / max(1, sum(c for _, c in items))
    )

    assert transform(obj) == 5.0


def test_seq():
    assert seq(1) == [1]
    assert seq(1, 2) == [1, 2]
    assert seq(1, 2, 3) == [1, 2, 3]


def test_optional__example():
    assert optional(None).or_else(5).get() == 5
    assert optional(3).or_else(5).get() == 3
    assert optional(3).get() == 3
    assert +optional(None).or_else(5) == 5

    assert +optional(3).transform(lambda a, b: a * b, b=2).get() == 6
    assert +optional(None).transform(lambda a, b: a + b, b=2).or_else(42).get() == 42

    assert +optional(3).or_else_call(lambda: 42).get() == 3
    assert +optional(None).or_else_call(lambda: 42).get() == 42


def test_optional__get_raises():
    with pytest.raises(ValueError):
        optional(None).get()


def test_try_call__example():
    assert try_call(raise_, ValueError).recover(lambda _: 42).get() == 42
    assert try_call(lambda: 13).recover(lambda _: 42).get() == 13


def test_try_call__then():
    assert 13 == (
        +try_call(raise_, ValueError)
        .then(lambda _: 42)
        .recover(lambda _: 13)
    )

    assert +try_call(lambda: 13).then(lambda v: v + 8) == 21


def test_try_call__get_for_failure_raises():
    with pytest.raises(ValueError):
        try_call(raise_, ValueError).get()


def test_show():
    assert show | 42 == 42


def test_show_format():
    assert show % '-- %d -- ' | 13 == 13
