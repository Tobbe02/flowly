from __future__ import print_function, division, absolute_import

try:
    import __builtin__ as builtins

except ImportError:
    import builtins

import math
import operator as op

import dask.bag as db
import pytest

from flowly.dsk import apply
from flowly.tz import apply_concat, chained, seq

executors = [
    lambda graph, obj: graph(obj),
    lambda graph, obj: apply(graph, obj).compute(),
]


@pytest.mark.parametrize('executor', executors)
def test_dags(executor):
    # build dags by using itemgetter and dicts
    scope = dict(
        a=db.from_sequence(range(0, 10), npartitions=3),
        b=db.from_sequence(range(10, 20), npartitions=3),
        c=db.from_sequence(range(20, 30), npartitions=3),
    )

    graph = chained(
        apply_concat([
            chained(op.itemgetter('a'), sum, seq),
            chained(op.itemgetter('b'), sum, seq),
            chained(op.itemgetter('c'), sum, seq),
        ]),
        apply_concat([
            chained(max, seq),
            chained(min, seq),
            chained(sum, seq),
        ])
    )

    actual = executor(graph, scope)
    assert sorted(actual) == sorted([
        sum(range(20, 30)),
        sum(range(0, 10)),
        sum(range(0, 30)),
    ])


def test_pipeline_example():
    from functools import reduce
    import operator as op

    data = range(100)
    result1 = math.sqrt(
        reduce(
            op.add,
            builtins.map(
                lambda x: x ** 2.0,
                builtins.filter(
                    lambda x: x % 2 == 0,
                    data,
                )
            )
        )
    )

    from toolz.curried import filter, map, reduce
    from flowly.tz import chained

    transform = chained(
        filter(lambda x: x % 2 == 0),
        map(lambda x: x ** 2.0),
        reduce(op.add),
        math.sqrt,
    )

    result2 = transform(data)

    assert result1 == result2
