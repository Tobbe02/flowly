"""Execute transformations via dask.
"""
import itertools as it
import functools as ft

try:
    import __builtins__ as builtins

except ImportError:
    import builtins

import dask.bag as db

import toolz
import toolz.functoolz
from toolz import pipe, compose, concat
from toolz.curried import map, mapcat

from .tz import (
    chained,
    show,
    apply_concat,
    apply_map_concat,
)


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
            name='builtins.sum',
            match=lambda transform, rules: transform == builtins.sum,
            apply=lambda bag, transform, rules: bag.sum(),
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
            name='toolz.curried.map',
            match=lambda transform, rules: (
                isinstance(transform, toolz.curry) and (transform.func is builtins.map)
            ),
            apply=lambda bag, transform, rules: bag.map(*transform.args, **transform.keywords),
        ),
        adict(
            name='toolz.curried.mapcat',
            match=lambda transform, rules: (
                isinstance(transform, toolz.curry) and (transform.func is toolz.mapcat)
            ),
            apply=lambda bag, transform, rules: bag.map(*transform.args).concat(),
        ),
        adict(
            name='itertools.chain.from_iterable',
            match=lambda transform, rules: transform == it.chain.from_iterable,
            apply=lambda bag, transform, rules: bag.concat(),
        ),
        adict(
            name='flowly.tz.apply_concat',
            match=lambda transform, rules: isinstance(transform, apply_concat),
            apply=_apply__flowly__tz__apply_concat,
        ),
        adict(
            name='flowly.tz.apply_map_concat',
            match=lambda transform, rules: isinstance(transform, apply_map_concat),
            apply=_apply__flowly__tz__apply_map_concat,
        ),
        adict(
            name='flowly.tz.chained',
            match=lambda transform, rules: isinstance(transform, chained),
            apply=_apply__flowly__tz__chained,
        ),
        adict(
            name='callable',
            match=lambda transform, rules: callable(transform),
            apply=lambda bag, transform, rules: transform(bag),
        )
    ]


def _apply__toolz__compose(bag, transform, rules):
    funcs = [transform.first] + list(transform.funcs)
    return _apply_funcs(bag, funcs, rules)


def _apply__flowly__tz__apply_concat(bag, transform, rules):
    return db.concat([
        apply(bag, func, rules=rules) for func in transform.funcs
    ])


def _apply__flowly__tz__apply_map_concat(bag, transform, rules):
    return db.concat([
        apply(bag, map(func), rules=rules) for func in transform.funcs
    ])


def _apply__flowly__tz__chained(bag, transform, rules):
    return _apply_funcs(bag, transform.funcs, rules)


def _apply_funcs(bag, funcs, rules):
    for func in funcs:
        bag = apply(bag, func, rules=rules)

    return bag


class adict(dict):
    __getattr__ = dict.__getitem__
