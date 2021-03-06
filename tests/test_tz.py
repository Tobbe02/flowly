from __future__ import print_function, division, absolute_import

import logging
import operator as op

import pytest

from flowly.hashing import functional_hash
from flowly.tz import (
    pipe,
    aggregate,
    apply_concat,
    apply_map_concat,
    build_dict,
    chained,
    frequencies,
    groupby,
    get_all_items,
    get_all_optional_items,
    itemsetter,
    kv_aggregateby,
    kv_keymap,
    kv_reduceby,
    kv_reductionby,
    kv_valmap,
    kv_transform,
    optional,
    printf,
    raise_,
    reduceby,
    reduction,
    reductionby,
    seq,
    show,
    timed,
    tupled,
    try_call,
)


def test_apply():
    assert pipe(5, lambda x: 2 * x, lambda x: x - 3) == 7
    assert pipe(5, lambda x: 2 * x, rewrites=[lambda f: lambda x: x + 5]) == 10


def test_chained__example():
    transform = chained(
        lambda a: 2 * a,
        lambda a: a - 3
    )

    assert transform(5) == 7


def test_chained__repr():
    repr(chained(
        lambda a: 2 * a,
        lambda a: a - 3
    ))


def test_chained__example_iter():
    transform = chained(*chained(
        lambda a: 2 * a,
        lambda a: a - 3
    ))

    assert transform(5) == 7


def test_chained__hash():
    hash_1 = functional_hash(chained(lambda a: 2 * a, lambda a: a - 3))
    hash_2 = functional_hash(chained(lambda a: 2 * a, lambda a: a - 3))

    assert hash_1 == hash_2


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


def test_kv_reductionby__no_perpartition_single_argument():
    seq = [(i % 2, i) for i in [1, 2, 3, 4, 5, 6, 7]]
    transform = kv_reductionby(sum)
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


def test_kv_aggregateby__no_perpartition():
    actual = pipe(
        [(0, {'v': 1}), (1, {'v': 2}), (0, {'v': 3}), (1, {'v': 4}), (0, {'v': 5})],
        kv_aggregateby({'v': [min, max, sum]}),
    )
    expected = [
        (0, {('v', 'max'): 5, ('v', 'min'): 1, ('v', 'sum'): 9}),
        (1, {('v', 'max'): 4, ('v', 'min'): 2, ('v', 'sum'): 6}),
    ]

    assert actual == expected


def test_kv_transform__aggregate():
    actual = pipe(
        [(0, {'v': 1}), (1, {'v': 2}), (0, {'v': 3}), (1, {'v': 4}), (0, {'v': 5})],
        kv_transform(aggregate({'v': [min, max, sum]})),
    )
    expected = [
        (0, {('v', 'max'): 5, ('v', 'min'): 1, ('v', 'sum'): 9}),
        (1, {('v', 'max'): 4, ('v', 'min'): 2, ('v', 'sum'): 6}),
    ]

    assert actual == expected


def test_reductionby__no_perpartition():
    seq = [1, 2, 3, 4, 5, 6, 7]
    transform = reductionby(lambda x: x % 2, None, sum)
    actual = sorted(transform(seq))
    expected = sorted([
        (1, sum([1, 3, 5, 7])),
        (0, sum([2, 4, 6])),
    ])

    assert actual == expected


def test_reductionby__with_perpartition():
    seq = [1, 2, 3, 4, 5, 6, 7]
    transform = reductionby(
        lambda x: x % 2,
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


def test_reduction__no_perpartition():
    # compute the mean
    transform = reduction(None, lambda s: sum(s) / max(1, len(s)))
    assert transform([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 5.0


def test_aggregate__no_perpartion():
    actual = pipe(
        [{'value': 1}, {'value': 2}, {'value': 3}],
        aggregate({'value': [sum, min, max]}),
    )
    expected = {('value', 'min'): 1, ('value', 'sum'): 6, ('value', 'max'): 3}

    assert actual == expected


def test_seq():
    assert seq(1) == [1]
    assert seq(1, 2) == [1, 2]
    assert seq(1, 2, 3) == [1, 2, 3]


def test_printf():
    printf('{0:.2%} {key} {foo}', 0.005, key='value', foo='bar')


def test_printf__empty():
    printf()


def test_printf__wrap():
    printf.wrap('{0:.2%} {key} {foo}', 0.005, key='value', foo='bar')


def test_printf__wrap__empty():
    printf.wrap()


def test_timed():
    with timed():
        pass


def test_timed_raises():
    with pytest.raises(ValueError):
        with timed():
            raise ValueError()


def test_timed_with_tag():
    with timed(tag='operation'):
        pass


def test_timed_with_tag_and_level():
    with timed(tag='operation', level=logging.DEBUG):
        pass


def test_tupled():
    assert tupled(lambda a, b: a + b)((1, 2)) == 3


def test_optional__example():
    assert optional(None).or_else(5).get() == 5
    assert optional(3).or_else(5).get() == 3
    assert optional(3).get() == 3
    assert +optional(None).or_else(5) == 5

    assert +optional(3).transform(lambda a, b: a * b, b=2).get() == 6
    assert +optional(None).transform(lambda a, b: a + b, b=2).or_else(42).get() == 42

    assert +optional(3).or_else_call(lambda: 42) == 3
    assert +optional(None).or_else_call(lambda: 42) == 42

    assert +optional(3).pipe(lambda x: [x]) == [3]
    assert +optional(None).pipe(lambda x: [x]) == [None]

    assert +optional(3).pipe() == 3
    assert +optional(3).pipe().or_else(6) == 3
    assert +optional(None).pipe().or_else(6) == 6

    repr(optional(3))
    repr(optional(None))


def test_empty_optional():
    assert +optional().or_else(7) == 7


def test_optional_identity():
    assert +optional(optional(3)) == 3


def test_optional_all():
    assert +optional.all(1, 2, 3).or_else('nothing') == [1, 2, 3]
    assert +optional.all(None, 2, 3).or_else('nothing') == 'nothing'


def test_optional_any():
    assert +optional.any(1, 2, 3).or_else('nothing') == 1
    assert +optional.any(None, 2, 3).or_else('nothing') == 2
    assert +optional.any(None, None, 3).or_else('nothing') == 3
    assert +optional.any(None, None, None).or_else('nothing') == 'nothing'


def test_optional__get_raises():
    with pytest.raises(ValueError):
        optional(None).get()


def test_try_call__example():
    assert try_call(raise_, ValueError).recover(lambda _: 42).get() == 42
    assert try_call(lambda: 13).recover(lambda _: 42).get() == 13

    repr(try_call(raise_, ValueError))
    repr(try_call(lambda: 13))


def test_try_call__identity():
    assert try_call.of(try_call(raise_, ValueError)).recover(lambda _: 42).get() == 42
    assert try_call.of(try_call(lambda: 13)).recover(lambda _: 42).get() == 13


def test_try_call__then():
    assert 13 == (
        +try_call(raise_, ValueError)
        .then(lambda _: 42)
        .recover(lambda _: 13)
    )

    assert +try_call(lambda: 13).then(lambda v: v + 8) == 21


def test_try_call__all():
    f = try_call(raise_, ValueError)
    a = try_call(lambda: 1)
    b = try_call(lambda: 2)
    c = try_call(lambda: 3)

    assert +try_call.all(a, b, c).recover(lambda _: None) == [1, 2, 3]
    assert +try_call.all(f, b, c).recover(lambda _: None) is None
    assert +try_call.all(f, f, c).recover(lambda _: None) is None
    assert +try_call.all(f, f, f).recover(lambda _: None) is None


def test_try_call__any():
    f = try_call(raise_, ValueError)
    a = try_call(lambda: 1)
    b = try_call(lambda: 2)
    c = try_call(lambda: 3)

    assert +try_call.any(a, b, c).recover(lambda _: None) == 1
    assert +try_call.any(f, b, c).recover(lambda _: None) == 2
    assert +try_call.any(f, f, c).recover(lambda _: None) == 3
    assert +try_call.any(f, f, f).recover(lambda _: None) is None


def test_try_call__get_for_failure_raises():
    with pytest.raises(ValueError):
        try_call(raise_, ValueError).get()


def test_show():
    assert show | 42 == 42


def test_show_format():
    assert show % '-- %d -- ' | 13 == 13


def test_get_all_items():
    data = [
        dict(a=1, b=2, c=3),
        dict(a=4, b=5, c=6),
    ]

    a, b = get_all_items('a', 'b')(data)

    assert a == [1, 4]
    assert b == [2, 5]


def test_get_all_optional_items():
    data = [
        dict(a=1, b=2, c=3),
        dict(a=4, b=5, c=6),
    ]

    a, n, b = get_all_optional_items('a', None, 'b')(data)

    assert a == [1, 4]
    assert b == [2, 5]
    assert n is None
