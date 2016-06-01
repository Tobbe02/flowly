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
all = lit(builtins.all)
any = lit(builtins.any)


def _as_tuple(*args):
    return tuple(args)


def _as_list(*args):
    return list(args)


as_tuple = lit(_as_tuple)
as_list = lit(_as_list)


def _and(*args):
    return builtins.all(args)


def _or(*args):
    return builtins.any(args)


and_ = lit(_and)
or_ = lit(_or)


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


# TODO: remove suffix _
class do(object):
    def __init__(self, expr):
        self.expr = expr

    def _flowly_eval_(self, obj):
        +(pipe(obj) | self.expr)
        return obj


class raise_(object):
    def __init__(self, exc):
        self.exc = exc

    def _flowly_eval_(self, obj):
        exc = +(pipe(obj) | eval_(self.exc))
        raise exc


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

    def do(self, body):
        return self(do(body))


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
            obj = +(pipe(obj) | self.body)

        return obj


class while_(object):
    def __init__(self, guard):
        self.guard = guard

    def __call__(self, body):
        return _while(self.guard, body)

    def do(self, body):
        return self(do(body))


class _while(object):
    def __init__(self, guard, body):
        self.guard = guard
        self.body = body

    def _flowly_eval_(self, obj):
        while +(pipe(obj) | eval_(self.guard)):
            obj = +(pipe(obj) | self.body)

        return obj


class if_(object):
    def __init__(self, cond):
        self.cond = cond

    def __call__(self, body):
        return _if([(self.cond, body)])

    def do(self, body):
        return self(do(body))


class _if_else_proxy(object):
    def __init__(self, cond_body_pairs):
        self.cond_body_pairs = cond_body_pairs

    def __call__(self, body):
        return _else(self.cond_body_pairs, body)

    def do(self, body):
        return self(do(body))


class _if(object):
    def __init__(self, cond_body_pairs):
        self.cond_body_pairs = cond_body_pairs
        self.else_ = _if_else_proxy(cond_body_pairs)

    def elif_(self, cond):
        return _elif1(self.cond_body_pairs, cond)

    def _flowly_eval_(self, obj):
        return _if_else_impl(obj, self.cond_body_pairs, _unset)


class _elif1(object):
    def __init__(self, cond_body_pairs, cond):
        self.cond_body_pairs = list(cond_body_pairs)
        self.cond = cond

    def __call__(self, body):
        return _if(self.cond_body_pairs + [(self.cond, body)])

    def do(self, body):
        return self(do(body))


class _else(object):
    def __init__(self, cond_body_pairs, body):
        self.cond_body_pairs = cond_body_pairs
        self.body = body

    def _flowly_eval_(self, obj):
        return _if_else_impl(obj, self.cond_body_pairs, self.body)


def _if_else_impl(obj, cond_body_pairs, else_body):
    result = obj
    for cond, body in cond_body_pairs:
        if +(pipe(obj) | eval_(cond)):
            result = +(pipe(obj) | eval_(body))
            break

    else:
        if else_body is not _unset:
            result = +(pipe(obj) | eval_(else_body))

    return result


class try_(object):
    def __init__(self, body, catch_pairs=()):
        self.body = body
        self.catch_pairs = list(catch_pairs)

    def except_(self, exc_class):
        return _try_catch1(self.body, self.catch_pairs, exc_class)

    def finally_(self, body):
        return _try_catch_finally(self.body, self.catch_pairs, body)

    @classmethod
    def do(cls, body):
        return cls(do(body))


class _try_catch1(object):
    def __init__(self, body, catch_pairs, exc_class):
        self.body = body
        self.catch_pairs = catch_pairs
        self.exc_class = exc_class

    def __call__(self, body):
        return _try_catch(self.body, self.catch_pairs + [(self.exc_class, body)])

    def do(self, body):
        return self(do(body))


class _else_proxy(object):
    def __init__(self, try_obj):
        self.try_obj = try_obj

    def __call__(self, body):
        return _try_catch_else(self.try_obj.body, self.try_obj.catch_pairs, body)

    def do(self, body):
        return self(do(body))


class _try_catch(object):
    def __init__(self, body, catch_pairs):
        self.body = body
        self.catch_pairs = catch_pairs

        self.else_ = _else_proxy(self)

    def except_(self, exc_class):
        return _try_catch1(self.body, self.catch_pairs, exc_class)

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
        result = +(pipe(obj) | body)

    except Exception as e:
        for catch_class, catch_body in catch_pairs:
            if isinstance(e, catch_class):
                result = +(pipe(obj) | catch_body)
                break

        else:
            raise

    else:
        if else_body is not _unset:
            result = +(pipe(obj) | else_body)

    finally:
        if finally_body is not _unset:
            +(pipe(obj) | finally_body)

    return result


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
