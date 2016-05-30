from __future__ import print_function
from ._base import side_effect, lit, _unset, callable_base, flowly_base, pipe, eval_

import itertools as it
import functools as ft
import operator

from six.moves import builtins


print = side_effect(builtins.print)
setattr = side_effect(builtins.setattr)
setitem = side_effect(operator.setitem)

abs = lit(builtins.abs)
chr = lit(builtins.chr)
len = lit(builtins.len)
ord = lit(builtins.ord)
range = lit(builtins.range)


class update(object):
    def __init__(self, **items):
        self.items = items

    def __call__(self, obj):
        obj = obj.copy()
        obj.update(self.items)
        return obj


class assign(object):
    def __init__(self, **assignments):
        self.assignments = assignments

    def __call__(self, obj):
        for key, val in self.assignments.items():
            builtins.setattr(obj, key, +(pipe(obj) | eval_(val)))

        return obj


class do_(object):
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, obj):
        +(pipe(obj) | self.expr)
        return obj


class raise_(object):
    def __init__(self, exc):
        self.exc = exc

    def _flowly_eval_(self, obj):
        exc = +(pipe(obj) | eval_(self.exc))
        raise exc


# TODO: implement for ... else
def for_(name):
    return _for1(name)


class _for1(object):
    def __init__(self, name):
        self.name = name

    def in_(self, iterable):
        return _for2(self.name, iterable)


class _for2(object):
    def __init__(self, name, iterable):
        self.name = name
        self.iterable = iterable

    def __call__(self, body):
        return _for(self.name, self.iterable, body)


class _for(object):
    def __init__(self, name, iterable, body):
        self.name = name
        self.iterable = iterable
        self.body = body

    def _flowly_eval_(self, obj):
        iterable = +(pipe(obj) | eval_(self.iterable))

        # TODO: implement scoping?
        for item in iterable:
            builtins.setattr(obj, self.name, item)
            +(pipe(obj) | self.body)

        return obj


class while_(object):
    def __init__(self, guard):
        self.guard = guard

    def __call__(self, body):
        return _while(self.guard, body)


class _while(object):
    def __init__(self, guard, body):
        self.guard = guard
        self.body = body

    def _flowly_eval_(self, obj):
        while +(pipe(obj) | self.guard):
            +(pipe(obj) | self.body)

        return obj


class if_(object):
    def __init__(self, cond):
        self.cond = cond

    def __call__(self, body):
        return _if([(self.cond, body)])


class _if(object):
    def __init__(self, cond_body_pairs):
        self.cond_body_pairs = cond_body_pairs

    def elif_(self, cond):
        return _elif1(self.cond_body_pairs, cond)

    def else_(self, body):
        return _else(self.cond_body_pairs, body)

    def _flowly_eval_(self, obj):
        for cond, body in self.cond_body_pairs:
            if +(pipe(obj) | eval_(cond)):
                +(pipe(obj) | body)
                break
        return obj


class _elif1(object):
    def __init__(self, cond_body_pairs, cond):
        self.cond_body_pairs = list(cond_body_pairs)
        self.cond = cond

    def __call__(self, body):
        return _if(self.cond_body_pairs + [(self.cond, body)])


class _else(object):
    def __init__(self, cond_body_pairs, body):
        self.cond_body_pairs = cond_body_pairs
        self.body = body

    def _flowly_eval_(self, obj):
        for cond, body in self.cond_body_pairs:
            if +(pipe(obj) | eval_(cond)):
                +(pipe(obj) | body)
                break

        else:
            +(pipe(obj) | self.body)

        return obj


class try_(object):
    def __init__(self, body, catch_pairs=()):
        self.body = body
        self.catch_pairs = list(catch_pairs)

    def except_(self, exc_class):
        return _try_catch1(self.body, self.catch_pairs, exc_class)

    def finally_(self, body):
        return _try_catch_finally(self.body, self.catch_pairs, body)


class _try_catch1(object):
    def __init__(self, body, catch_pairs, exc_class):
        self.body = body
        self.catch_pairs = catch_pairs
        self.exc_class = exc_class

    def __call__(self, body):
        return _try_catch(self.body, self.catch_pairs + [(self.exc_class, body)])


class _try_catch(object):
    def __init__(self, body, catch_pairs):
        self.body = body
        self.catch_pairs = catch_pairs

    def except_(self, exc_class):
        return _try_catch1(self.body, self.catch_pairs, exc_class)

    def else_(self, body):
        return _try_catch_else(self.body, self.catch_pairs, body)

    def finally_(self, body):
        return _try_catch_finally(self.body, self.catch_pairs, body)

    def _flowly_eval_(self, obj):
        return _try_catch_impl(obj, self.body, self.catch_pairs)


class _try_catch_else(object):
    def __init__(self, body, catch_pairs, else_body):
        self.body = body
        self.catch_pairs = catch_pairs
        self.else_body = else_body

    def finally_(self, body):
        return _try_catch_finally(self.body, self.catch_pairs, body, self.else_body)

    def _flowly_eval_(self, obj):
        return _try_catch_impl(obj, self.body, self.catch_pairs, _unset, self.else_body)


class _try_catch_finally(object):
    def __init__(self, body, catch_pairs, finally_body, else_body=_unset):
        self.body = body
        self.catch_pairs = catch_pairs
        self.finally_body = finally_body
        self.else_body = else_body

    def _flowly_eval_(self, obj):
        return _try_catch_impl(obj, self.body, self.catch_pairs, self.finally_body, self.else_body)


def _try_catch_impl(obj, body, catch_pairs, finally_body=_unset, else_body=_unset):
    try:
        +(pipe(obj) | body)

    except Exception as e:
        for catch_class, catch_body in catch_pairs:
            if isinstance(e, catch_class):
                +(pipe(obj) | catch_body)
                break

        else:
            raise

    else:
        if else_body is not _unset:
            +(pipe(obj) | else_body)

    finally:
        if finally_body is not _unset:
            +(pipe(obj) | finally_body)

    return obj


# TODO: implement with statement


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

        +(fy.pipe(obj) | fpy.reduce(this.left + this.right))
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
            return ft.reduce(self.func, obj, +(pipe(obj) | eval_(self.initializer)))


class _reduce_arg(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right


def first(obj):
    return next(iter(obj))


def flatmap(obj):
    return it.chain.from_iterable(obj)
