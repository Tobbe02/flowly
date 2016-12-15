"""Execute transformations via dask.
"""
from __future__ import print_function, division, absolute_import

import functools as ft
import itertools as it
import operator as op
import logging

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
    kv_keymap,
    kv_reduceby,
    kv_reductionby,
    kv_valmap,
    raise_,
    reduceby,
    reduction,
    reductionby,
    seq,
)

_logger = logging.getLogger(__name__)
id_sequence = it.count()


def item_from_object(obj):
    """Helper create a :class:`dask.bag.Item` from a python object.
    """
    return db.Item.from_delayed(delayed(obj))


def apply_to_local(transform, obj, npartitions=None, get=None, rules=None):
    """Distribute obj, then apply the transform, finally compute the result.

    :param Callable[Any,Any] transform:
        the transformation to apply.

    :param Sequence[Any] obj:
        the list of objects to transform.

    :param Optional[int] npartitions:
        the number of partitions to split the original sequence into.

    :param Callable[Any,Any,Any] get:
        the get function to use when computing the resulting dask object.
        To execute in parallel, use the ``get`` method of
        ``distributed.Client``.
        See :class:`flowly.dst.LocalCluster` for a simple way to start a local
        cluster and to construct the client.

    :param Optional[Iterable] rules:
        See :func:`flowly.dsk.apply`.
    """
    obj = db.from_sequence(obj, npartitions=npartitions)
    obj = apply(transform, obj, rules=rules)
    return obj.compute(get=get)


def apply(transform, obj, rules=None):
    """Translate the dask object via the given transformation.

    :param Callable[Any,Any] transform:
        the transformation to apply.

    :param Any object:
        the object to transform.

    :param Optional[Iterable] rules:
        an iterable containing rules to interpret the given transformation
        on top of the bassed in object. If not given,
        :func:`flowly.dsk.get_default_rules` will be called to retrieve the
        rules to apply.

        Each rule should be an object with two functions ``match`` and
        ``apply``. The first matching rule will be applied and no subsequent
        rule will be checked.
    """
    if rules is None:
        rules = get_default_rules()

    for rule in rules:
        try:
            if rule.match(obj, transform, rules):
                return rule.apply(obj, transform, rules)

        except:
            print('error for rule %s', rule)
            raise

    raise ValueError('cannot handle transform')


def get_default_rules():
    """
    Return the list of default rules used to interpret expressions in
    :func:`apply()<flowly.dsk.apply>`.

    Each rules has an additional property ``.name``, that may be useful when
    modifying the rules.
    """
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
            match=_match_equal(toolz.concat),
            apply=lambda bag, transform, rules: bag.concat(),
        ),
        adict(
            name='toolz.compose',
            match=_match_isinstance(toolz.functoolz.Compose),
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
            name='toolz.curried.reduce',
            match=_match_curried(ft.reduce),
            apply=lambda bag, transform, rules: bag.reduction(
                lambda i: i,
                lambda partitions: ft.reduce(transform.args[0], it.chain.from_iterable(partitions)),
            )
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
            name='toolz.curried.groupby',
            match=_match_curried(toolz.groupby),
            apply=lambda *args, **kwargs: raise_(ValueError, 'use flowly.tz.groupby')
        ),
        adict(
            name='toolz.curried.reduceby',
            match=_match_curried(toolz.reduceby),
            apply=lambda *args, **kwargs: raise_(ValueError, 'use flowly.tz.reduceby')
        ),
        adict(
            name='toolz.unique',
            match=_match_equal(toolz.unique),
            apply=_methodcaller('distinct'),
        ),
        adict(
            name='itertools.chain.from_iterable',
            match=_match_equal(it.chain.from_iterable),
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
            name='flowly.tz.itemsetter',
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
            name='flowlfy.tz.kv_keymap',
            match=_match_isinstance(kv_keymap),
            apply=lambda bag, transform, rules: apply(
                toolz.curried.map(lambda t: (transform.func(t[0]), t[1])), bag, rules=rules,
            ),
        ),
        adict(
            name='flowly.tz.kv_valmap',
            match=_match_isinstance(kv_valmap),
            apply=lambda bag, transform, rules: apply(
                toolz.curried.map(lambda t: (t[0], transform.func(t[1]))), bag, rules=rules,
            ),
        ),
        adict(
            name='flowly.tz.kv_reduceby',
            match=_match_isinstance(kv_reduceby),
            apply=_apply_kv_reduceby,
        ),
        adict(
            name='flowly.tz.kv_reductionby',
            match=_match_isinstance(kv_reductionby),
            apply=_apply_kv_reductionby,
        ),
        adict(
            name='flowly.tz.reduceby',
            match=_match_isinstance(reduceby),
            apply=lambda bag, transform, rules: (
                bag
                .groupby(transform.key)
                .map(lambda t: (t[0], ft.reduce(transform.binop, t[1])))
            ),
        ),
        adict(
            name='flowly.tz.reduction',
            match=_match_isinstance(reduction),
            apply=lambda bag, transform, rules: bag.reduction(
                transform.perpartition, transform.aggregate, split_every=transform.split_every,
            ),
        ),
        adict(
            name='flowly',
            match=_match_isinstance(reductionby),
            apply=_apply_reductionby,
        ),
        adict(
            name='flowly.tz.seq',
            match=_match_equal(seq),
            apply=lambda item, transform, rules: db.from_delayed([
                item.apply(lambda i: [i]).to_delayed()
            ])
        ),
        # TODO: let any curried callable fallback to the callable itself, if not args were given
        # TODO: add option to skip arbitrary callables and add marker functions to annotate them
        adict(
            name='callable',
            match=lambda bag, transform, rules: callable(transform),
            apply=lambda bag, transform, rules: transform(bag),
        )
    ]


def _match_equal(obj):
    return lambda bag, transform, rules: transform == obj


def _match_isinstance(kls):
    return lambda bag, transform, rules: isinstance(transform, kls)


def _match_curried(func):
    return lambda bag, transform, rules: (
        isinstance(transform, toolz.curry) and (transform.func is func)
    )


def _methodcaller(name):
    return lambda bag, transform, rules: getattr(bag, name)()


def _apply__toolz__compose(bag, transform, rules):
    funcs = [transform.first] + list(transform.funcs)
    return _apply_funcs(bag, funcs, rules)


def _apply__flowly__tz__apply_concat(bag, transform, rules):
    return db.concat([apply(func, bag, rules=rules) for func in transform.funcs])


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
        bag = apply(func, bag, rules=rules)

    return bag


def _build_dask_dict(obj, transform, rules):
    result = {}

    for assignment in transform.assigments:
        for k, func in assignment.items():
            result[k] = apply(func, obj, rules=rules)

    return dask_dict(result)


def _update_dask_dict(obj, transform, rules):
    obj = obj.copy()

    for assignment in transform.assigments:
        for k, func in assignment.items():
            obj[k] = apply(func, obj, rules=rules)

    return dask_dict(obj)


def _apply_reductionby(obj, transform, rules):
    if transform.perpartition is not None:
        # TODO: optimize perpartition call
        impl = chained(
            groupby(transform.key),
            toolz.curried.map(lambda t: (
                t[0],
                transform.aggregate([transform.perpartition(t[1])])
            )),
        )

    else:
        impl = chained(
            groupby(transform.key),
            toolz.curried.map(lambda t: (t[0], transform.aggregate(t[1]))),
        )

    return apply(impl, obj, rules=rules)


def _apply_kv_reduceby(obj, transform, rules):
    return (
        obj
        .groupby(op.itemgetter(0))
        .map(lambda t: (
            t[0], ft.reduce(transform.binop, (i for (_, i) in t[1]))
        ))
    )


def _apply_kv_reductionby(obj, transform, rules):
    if transform.perpartition is not None:
        impl = reductionby(
            key=op.itemgetter(0),
            split_every=transform.split_every,
            perpartition=lambda t: transform.perpartition([i for _, i in t]),

            # NOTE: perpartition strips the key
            aggregate=lambda t: transform.aggregate(t),
        )

    else:
        impl = reductionby(
            key=op.itemgetter(0),
            split_every=transform.split_every,
            perpartition=None,
            aggregate=lambda t: transform.aggregate([i for _, i in t]),
        )

    return apply(impl, obj, rules=rules)


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
