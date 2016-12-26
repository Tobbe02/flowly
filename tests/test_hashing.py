from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd
import pytest

from flowly.hashing import functional_hash, base_hash, get_hash_parts, functional_system

global_func = None


def times_2():
    return lambda x: 2 * x


def times_3():
    return lambda x: 3 * x


def test_functional_hash__lambda():
    hash_1 = functional_hash(lambda x: 2 * x)
    hash_2 = functional_hash(lambda x: 2 * x)

    assert hash_1 == hash_2


def test_functional_hash__lambda__closure():
    def gen_func(a):
        return lambda x: x + a

    assert functional_hash(gen_func(0)) == functional_hash(gen_func(0))
    assert functional_hash(gen_func(0)) != functional_hash(gen_func(1))


def test_functional_hash__lamba_globals():
    global global_func

    global_func = times_2()
    hash_1 = functional_hash(lambda x: global_func(x) - 3)
    hash_2 = functional_hash(lambda x: global_func(x) - 3)

    global_func = times_3()
    hash_3 = functional_hash(lambda x: global_func(x) - 3)

    assert hash_1 == hash_2
    assert hash_1 != hash_3


def test_functional_hash__builtins():
    assert functional_hash(any) == functional_hash(any)


def test_base_hash__lambda():
    "hashes of lambda functions differ if defined on different lines"
    hash_1 = base_hash(lambda x: 2 * x)
    hash_2 = base_hash(lambda x: 2 * x)

    assert hash_1 != hash_2


def test_general_object():
    class foo_object(object):
        def __init__(self, foo):
            self.foo = foo

    o1 = foo_object('bar')
    o2 = foo_object('bar')
    o3 = foo_object('baz')

    print('o1')
    print(get_hash_parts(functional_system, o1))
    print(get_hash_parts(functional_system, o1, recursive=False))

    print('o3')
    print(get_hash_parts(functional_system, o3))
    print(get_hash_parts(functional_system, o3, recursive=False))

    assert functional_hash(o1) == functional_hash(o2)
    assert functional_hash(o1) != functional_hash(o3)


def test_base_system__examples():
    base_hash(2.0)
    base_hash(True)
    base_hash(False)
    base_hash({'1', '2', 3})
    base_hash({int: 'int', float: 'float', 'int': int, 'float': float})

    assert base_hash(True) != base_hash(1)


def test_base_system__pandas():
    assert (
        base_hash(pd.DataFrame({'a': [1, 2, 3]})) ==
        base_hash(pd.DataFrame({'a': [1, 2, 3]}))
    )

    assert base_hash(pd.Series([1, 2, 3])) == base_hash(pd.Series([1, 2, 3]))


def test_base_system__numpy():
    assert base_hash(np.array([1, 2, 3])) == base_hash(np.array([1, 2, 3]))


def test_base_system__classes_in_main():
    class foo(object):
        pass

    foo.__module__ = '__main__'

    with pytest.raises(ValueError):
        base_hash(foo)


def test_recursive_hash():
    o = {}
    o['o'] = o

    functional_hash(o)

    l = []
    l.append(l)

    functional_hash(l)


def test_hash_modules():
    h1 = base_hash(np)
    h2 = base_hash(np)

    assert h1 == h2
