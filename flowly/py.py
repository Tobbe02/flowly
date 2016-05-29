from __future__ import print_function
from ._base import side_effect, lit, _unset, callable_base, flowly_base, pipe, eval_expr

import itertools as it
import functools as ft

from six.moves import builtins


print = side_effect(builtins.print)

abs = lit(builtins.abs)
chr = lit(builtins.chr)
len = lit(builtins.len)
ord = lit(builtins.ord)


class map(flowly_base):
    """Transform an iterable by a flowly expression.

    Example::

        +(fy.pipe(obj) | fpy.map(this["foo"] + this["bar"]))
    """
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, obj):
        return (+(pipe(i) | self.expr) for i in obj)


class filter(flowly_base):
    """Filter an iterable by a flowly expresion.

    Example::

        +(fy.pipe(obj) | fpy.filter(this.startswith("h")))
    """
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, obj):
        return (i for i in obj if +(pipe(i) | self.expr))


class reduce(object):
    """Reduce an iterable by a flowly expression.

    Inside the reduction expression, the reduction arguments are passed as the
    ``left`` and ``right`` attributes of  ``this``.

    Example::

        +(fy.pipe(obj))
    """
    def __init__(self, expr, initializer=_unset):
        self.expr = expr
        self.initializer = initializer

    def func(self, left, right):
        return +(pipe(_reduce_arg(left, right)) | self.expr)

    def __call__(self, obj):
        if self.initializer is _unset:
            return ft.reduce(self.func, obj)

        else:
            return ft.reduce(self.func, obj, eval_expr(obj, self.initializer))


class _reduce_arg(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right


def first(obj):
    return next(iter(obj))


def flatmap(obj):
    return it.chain.from_iterable(obj)
