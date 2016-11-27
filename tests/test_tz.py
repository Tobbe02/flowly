from __future__ import print_function, division, absolute_import
from flowly.tz import chained, optional, raise_, try_call, show

import pytest


def test_chained__example():
    transform = chained(
        lambda a: 2 * a,
        lambda a: a - 3
    )

    assert transform(5) == 7


def test_chained__composition():
    transform = chained(lambda a: a * 2) + chained(lambda a: a -3)
    assert transform(5) == 7


def test_optional__example():
    assert optional(None).or_else(lambda: 5).get() == 5
    assert optional(3).or_else(lambda: 5).get() == 3
    assert optional(3).get() == 3
    assert +optional(None).or_else(lambda: 5) == 5


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
