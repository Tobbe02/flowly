from __future__ import print_function, division, absolute_import

from flowly.hashing import functional_hash, base_hash


global_func = None


def times_2(x):
    return 2 * x


def times_3(x):
    return 3 * x


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

    global_func = times_2
    hash_1 = functional_hash(lambda x: global_func(x) - 3)
    hash_2 = functional_hash(lambda x: global_func(x) - 3)

    global_func = times_3
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
    o1 = foo_object('bar')
    o2 = foo_object('bar')
    o3 = foo_object('baz')

    assert functional_hash(o1) == functional_hash(o2)
    assert functional_hash(o1) != functional_hash(o3)


class foo_object(object):
    def __init__(self, foo):
        self.foo = foo
