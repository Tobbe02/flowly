"""Additional hashing primitives to detect uniqueness
"""
from __future__ import print_function, division, absolute_import

import functools as ft
import hashlib
import itertools as it
import types

from ._dispatch import Dispatch

try:
    import dis
    dis.get_instructions

except AttributeError:
    import dis3 as dis

try:
    import builtins

except ImportError:  # pragma: no cover
    import __builtin__ as builtins

try:
    string_types = [str, unicode]

except NameError:  # pragma: no cover
    string_types = [str]


__all__ = [
    'base_hash',
    'functional_hash',
    'base_system',
    'functional_system',
    'compute_hash',
    'composite_hash',
]


def base_hash(obj):
    return compute_hash(base_system, obj)


def functional_hash(obj):
    return compute_hash(functional_system, obj)


def is_lambda(func):
    return func.__name__ == (lambda: None).__name__


def compute_hash(hash_system, obj):
    h = hashlib.sha1()

    for part in hash_system(obj, hash_system):
        h.update(part)

    return h.hexdigest()


def composite_hash(func):
    @ft.wraps(func)
    def impl(obj, base_system):
        parts = func(obj, base_system)

        return it.chain(
            [repr(len(parts)).encode('utf8')],
            it.chain.from_iterable(
                base_system(part, base_system) for part in parts
            ),
        )

    return impl


base_system = Dispatch()


@base_system.default
@composite_hash
def base_system_object(o, _):
    return type(o), o.__reduce__()


@base_system.bind(type)
def base_system_type(t, _):
    if t.__module__ == '__main__':
        raise ValueError('cannot hash classes defined in __main__')

    return [
        u'T:{}:{}'.format(t.__module__, t.__name__).encode('utf8'),
    ]


@base_system.bind(type(None))
def base_system_none_type(obj, _):
    return [b'0']


@base_system.bind(string_types)
def base_system_str(o, base_system):
    return it.chain(
        [b'2'],
        base_system(type(o), base_system),
        [o.encode('utf8')],
    )


@base_system.bind(bytes)
def base_system_bytes(o, base_system):
    return it.chain(
        [b'2'],
        base_system(type(o), base_system),
        [o],
    )


@base_system.bind([int, float, bool])
def base_system_primitives(o, base_system):
    return it.chain(
        [b'2'],
        base_system(type(o), base_system),
        [repr(o).encode('utf8')],
    )


@base_system.bind([list, tuple])
@composite_hash
def base_system_list(l, _):
    return [type(l)] + list(l)


@base_system.bind(dict)
@composite_hash
def base_system_dict(d, _):
    return [type(d)] + sorted(d.items())


@base_system.bind(types.FunctionType)
@composite_hash
def base_system_function(func, _):
    if not is_lambda(func) and func.__module__ != '__main__':
        return type(func), u'{}:{}'.format(func.__module__, func.__name__)

    # TODO: handle referenced gloabal variables
    if func.__closure__ is not None:
        closure = tuple(c.cell_contents for c in func.__closure__)

    else:
        closure = None

    return (
        type(func),
        func.__code__,
        closure,
        func.__dict__,
        get_func_globals(func),
    )


def get_func_globals(func):
    """Get all globals referenced in a function.
    """
    source = {"True": True, "False": False, "None": None}
    source.update(builtins.__dict__)
    source.update(func.__globals__)

    return {k: source[k] for k in get_globals(func)}


def get_globals(func):
    result = set()

    for inst in dis.get_instructions(func):
        if inst.opname == 'LOAD_GLOBAL':
            result.add(inst.argval)

    return result


@base_system.bind(types.BuiltinFunctionType)
@composite_hash
def base_system_builtin_function_type(func, _):
    return type(func), u'{}:{}'.format(func.__module__, func.__name__)


@base_system.bind(types.CodeType)
@composite_hash
def base_system_code(code, _):
    return type(code), [
        (k, getattr(code, k))
        for k in dir(code)
        if k.startswith('co')
    ]


@base_system.bind_if(lambda obj: type(obj).__module__.startswith('pandas.'))
def base_system_pandas(base_system):
    import pandas as pd
    import dask.base

    @base_system.bind([pd.Index, pd.DataFrame, pd.Series])
    @composite_hash
    def base_system_pandas(obj, _):
        return dask.base.normalize_token(obj)


@base_system.bind_if(lambda obj: type(obj).__module__.startswith('numpy.'))
def base_system_numpy(base_system):
    import numpy as np
    import dask.base

    @base_system.bind([np.ndarray, np.dtype, np.generic])
    @composite_hash
    def base_system_numpy(obj, _):
        return dask.base.normalize_token(obj)


functional_system = base_system.inherit()


@functional_system.bind(types.CodeType)
@composite_hash
def functional_system_code(code, _):
    skip = {'co_filename', 'co_firstlineno'}

    return type(code), [
        (k, getattr(code, k))
        for k in dir(code)
        if k.startswith('co') and k not in skip
    ]
