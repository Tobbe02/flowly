"""Execute transformations via dask.
"""
import itertools as it

try:
    import __builtins__ as builtins

except ImportError:
    import builtins

import dask
import dask.bag as db
from dask.delayed import delayed

import toolz
import toolz.functoolz
from toolz import (
    count,
    merge,
    partition_all,
)

from .tz import (
    apply_concat,
    apply_map_concat,
    build_dict,
    chained,
    frequencies,
    groupby,
    itemsetter,
    reduction,
    seq,
)

id_sequence = it.count()


def item_from_object(obj):
    return db.Item.from_delayed(delayed(obj))


# TODO: change argument order for currying
def apply(bag, transform, rules=None):
    """Translate the dask object via the given transformation.
    """
    if rules is None:
        rules = _default_rules()

    for rule in rules:
        if rule.match(transform, rules):
            return rule.apply(bag, transform, rules)

    raise ValueError('cannot handle transform')


def _default_rules():
    return [
        adict(
            name='builtins.sum', match=_match_equal(builtins.sum), apply=_methodcaller('sum'),
        ),
        adict(
            name='builtins.any', match=_match_equal(builtins.all), apply=_methodcaller('all'),
        ),
        adict(
            name='builtins.any', match=_match_equal(builtins.any), apply=_methodcaller('any'),
        ),
        adict(
            name='builtins.len', match=_match_equal(builtins.len), apply=_methodcaller('count'),
        ),
        adict(
            name='builtins.max', match=_match_equal(builtins.max), apply=_methodcaller('max'),
        ),
        adict(
            name='builtins.min', match=_match_equal(builtins.min), apply=_methodcaller('min'),
        ),
        adict(
            name='toolz.concat',
            match=lambda transform, rules: transform == toolz.concat,
            apply=lambda bag, transform, rules: bag.concat(),
        ),
        adict(
            name='toolz.compose',
            match=lambda transform, rules: isinstance(transform, toolz.functoolz.Compose),
            apply=_apply__toolz__compose,
        ),
        adict(
            name='toolz.count', match=_match_equal(count), apply=_methodcaller('count'),
        ),
        adict(
            name='toolz.frequencies',
            match=_match_equal(frequencies),
            apply=_methodcaller('frequencies'),
        ),
        adict(
            name='toolz.curried.filter',
            match=_match_curried(builtins.filter),
            apply=lambda bag, transform, rules: bag.filter(*transform.args, **transform.keywords)
        ),
        adict(
            name='toolz.curried.map',
            match=_match_curried(builtins.map),
            apply=lambda bag, transform, rules: bag.map(*transform.args, **transform.keywords),
        ),
        adict(
            name='toolz.curried.mapcat',
            match=_match_curried(toolz.mapcat),
            apply=lambda bag, transform, rules: bag.map(*transform.args).concat(),
        ),
        adict(
            name='toolz.curried.pluck',
            match=_match_curried(toolz.pluck),
            apply=lambda bag, transform, rules: bag.pluck(*transform.args, **transform.keywords),
        ),
        adict(
            name='toolz.curried.random_sample',
            match=_match_curried(toolz.random_sample),
            apply=lambda bag, transform, rules: bag.random_sample(*transform.args, **transform.keywords),
        ),
        adict(
            name='toolz.curried.remove',
            match=_match_curried(toolz.remove),
            apply=lambda bag, transform, rules: bag.remove(*transform.args, **transform.keywords),
        ),
        adict(
            name='toolz.curried.take',
            match=_match_curried(toolz.take),
            apply=lambda bag, transform, rules: bag.take(
                *transform.args,
                **merge(transform.keywords, dict(compute=False, npartitions=-1))
            ),
        ),
        adict(
            name='toolz.curried.topk',
            match=_match_curried(toolz.topk),
            apply=lambda bag, transform, rules: bag.topk(*transform.args, **transform.keywords),
        ),
        adict(
            name='toolz.unique',
            match=_match_equal(toolz.unique),
            apply=_methodcaller('distinct'),
        ),
        adict(
            name='itertools.chain.from_iterable',
            match=lambda transform, rules: transform == it.chain.from_iterable,
            apply=lambda bag, transform, rules: bag.concat(),
        ),
        adict(
            name='flowly.tz.apply_concat',
            match=_match_isinstance(apply_concat),
            apply=_apply__flowly__tz__apply_concat,
        ),
        adict(
            name='flowly.tz.apply_map_concat',
            match=_match_isinstance(apply_map_concat),
            apply=_apply__flowly__tz__apply_map_concat,
        ),
        adict(
            name='flowly.tz.build_dict',
            match=_match_isinstance(build_dict),
            apply=_build_dask_dict,
        ),
        adict(
            name='flowly.tz.build_dict',
            match=_match_isinstance(itemsetter),
            apply=_update_dask_dict,
        ),
        adict(
            name='flowly.tz.chained',
            match=_match_isinstance(chained),
            apply=_apply__flowly__tz__chained,
        ),
        adict(
            name='flowly.tz.groupby',
            match=_match_isinstance(groupby),
            # TODO: inject remaining arguments into groupby
            apply=lambda bag, transform, rules: bag.groupby(transform.key),
        ),
        adict(
            name='flowly.tz.reduction',
            match=_match_isinstance(reduction),
            apply=lambda bag, transform, rules: bag.reduction(
                transform.perpartition, transform.aggregate, split_every=transform.split_every,
            ),
        ),
        adict(
            name='flowly.tz.seq',
            match=_match_equal(seq),
            apply=lambda item, transform, rules: db.from_delayed([
                item.apply(lambda i: [i]).to_delayed()
            ])
        ),
        # TODO: let any curried callable fallback to the callable itself, if not args were given
        adict(
            name='callable',
            match=lambda transform, rules: callable(transform),
            apply=lambda bag, transform, rules: transform(bag),
        )
    ]


def _match_equal(obj):
    return lambda transform, rules: transform == obj


def _match_isinstance(kls):
    return lambda transform, rules: isinstance(transform, kls)


def _match_curried(func):
    return lambda transform, rules: (
        isinstance(transform, toolz.curry) and (transform.func is func)
    )


def _methodcaller(name):
    return lambda bag, transform, rules: getattr(bag, name)()


def _apply__toolz__compose(bag, transform, rules):
    funcs = [transform.first] + list(transform.funcs)
    return _apply_funcs(bag, funcs, rules)


def _apply__flowly__tz__apply_concat(bag, transform, rules):
    return db.concat([apply(bag, func, rules=rules) for func in transform.funcs])


def _apply__flowly__tz__apply_map_concat(bag, transform, rules):
    # TODO: handle impure functions
    return db.concat([
        bag.map_partitions(_apply_map_concat_impl, funcs=list(funcs), _flowly_id=flowly_id)
        # TODO: fix chunk_size
        for flowly_id, funcs in zip(id_sequence, partition_all(10, transform.funcs))
    ])


def _apply_map_concat_impl(obj, funcs, _flowly_id):
    return [
        func(item)
        for item in obj
        for func in funcs
    ]


def _apply__flowly__tz__chained(bag, transform, rules):
    return _apply_funcs(bag, transform.funcs, rules)


def _apply_funcs(bag, funcs, rules):
    for func in funcs:
        bag = apply(bag, func, rules=rules)

    return bag


def _build_dask_dict(obj, transform, rules):
    result = {}

    for assignment in transform.assigments:
        for k, func in assignment.items():
            result[k] = apply(obj, func, rules=rules)

    return dask_dict(result)


def _update_dask_dict(obj, transform, rules):
    obj = obj.copy()

    for assignment in transform.assigments:
        for k, func in assignment.items():
            obj[k] = apply(obj, func, rules=rules)

    return dask_dict(obj)


class adict(dict):
    __getattr__ = dict.__getitem__


class dask_dict(dict):
    def copy(self):
        return dask_dict(self)

    def compute(self, **kwargs):
        items = list(self.items())
        keys = [key for key, _ in items]
        values = [value for _, value in items]
        values = dask.compute(*values, **kwargs)

        return dask_dict(zip(keys, values))