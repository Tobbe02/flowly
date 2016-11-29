from __future__ import print_function, division, absolute_import

import operator as op

import dask.bag as db
import pytest

from flowly.dsk import apply
from flowly.tz import apply_concat, chained, seq

executors = [
    lambda graph, obj: graph(obj),
    lambda graph, obj: apply(obj, graph).compute(),
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
