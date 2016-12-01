"""Additional toolz.
"""
from __future__ import print_function, division, absolute_import

import itertools as it

# TODO: add proper __all__


class ShowImpl(object):
    """Inspired by q"""
    def __init__(self, fmt='{!r}'):
        self.fmt = fmt

    def __call__(self, obj):
        print(self.fmt.format(obj))
        return obj

    def __or__(self, obj):
        return self(obj)

    def __mod__(self, fmt):
        return ShowImpl(fmt)


show = ShowImpl()


class itemsetter(object):
    def __init__(self, *prototypes, **assigments):
        self.assigments = list(prototypes) + [dict(assigments)]

    def __call__(self, obj):
        obj = obj.copy()
        for assigment in self.assigments:
            for k, func in assigment.items():
                obj[k] = func(obj)

        return obj


class build_dict(object):
    def __init__(self, *prototypes, **assigments):
        self.assigments = list(prototypes) + [dict(assigments)]

    def __call__(self, obj):
        res = {}
        for assigment in self.assigments:
            for k, func in assigment.items():
                res[k] = func(obj)

        return res


class chained(object):
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, obj):
        for func in self.funcs:
            obj = func(obj)

        return obj

    def __add__(self, other):
        return chained(*(list(self.funcs) + list(other.funcs)))


class _apply_concat_base(object):
    def __init__(self, funcs, chunks=None):
        self.funcs = list(funcs)
        self.chunks = chunks


class apply_concat(_apply_concat_base):
    """Apply the functions in parallel and concatenate the results.

    .. note::

        The order of the result is not guaranteed.

        Also care has to be taken for iterable arguments. They must be iterable
        repeatedly or or only a single function may iterate over the object.

    Equivalent to::

        it.chain.from_iterable(func(obj, *args, **kwargs) for func in funcs)

    Each function should map from an iterable to an iterable. The result will
    be the concatenation of all items in the results for all functions.

    :param Iterable[Callable[Any,...]] funcs:
        The functions to apply. Needs to be finite.

    :param Optional[int] chunks:
        A hint to parallel executors about the desired chunk size.
    """
    def __call__(self, obj):
        return list(it.chain.from_iterable(func(obj) for func in self.funcs))


class apply_map_concat(_apply_concat_base):
    """TODO: describe, decide which way to order

    Equivalent to::

        it.chain.from_iterable((func(item) for func in funcs) for item in obj)

    Each function should map from a single item to a transformed item. The
    result will be the concatenation of the transformed items of all functions.

    Note: the order is not guaranteed.

    :param Iterable[Callable[Any,...]] funcs:
        The functions to apply. Needs to be finite.

    :param Optional[int] chunks:
        A hint to parallel executors about the desired chunk size.
    """
    def __call__(self, obj):
        return it.chain.from_iterable(
            (func(item) for func in self.funcs)
            for item in obj
        )


def frequencies(obj):
    """In contrast to ``toolz.frequencies`` return ``(item, count)`` pairs.
    """
    result = {}

    for item in obj:
        result[item] = result.get(item, 0) + 1

    return list(result.items())


class groupby(object):
    def __init__(self, key):
        self.key = key

    def __call__(self, obj):
        result = {}
        for item in obj:
            result.setdefault(self.key(item), []).append(item)

        return list(result.items())


class reduction(object):
    def __init__(self, perpartition, aggregate, split_every=None):
        self.perpartition = perpartition
        self.aggregate = aggregate
        self.split_every = split_every

    def __call__(self, obj):
        return self.aggregate([self.perpartition(obj)])


def seq(*items):
    """Turn one or multiple values into a list.
    """
    return list(items)


def optional(val):
    return Just(val) if val is not None else Nothing()


def try_call(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)

    except Exception as e:
        return Failure(e)

    else:
        return Success(result)


def raise_(exc_class, *args, **kwargs):
    raise exc_class(*args, **kwargs)


class _Gettable(object):
    def __pos__(self):
        return self.get()

    def get(self):  # pragma: no cover
        raise NotImplementedError()


class _Maybe(_Gettable):
    pass


class Nothing(_Maybe):
    def __init__(self):
        pass

    def get(self):
        raise ValueError()

    def or_else(self, factory):
        return optional(factory())


class Just(_Maybe):
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def or_else(self, factory):
        return self


class _Try(_Gettable):
    pass


class Success(_Try):
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def then(self, func, *args, **kwargs):
        return try_call(func, self.value, *args, **kwargs)

    def recover(self, func, *args, **kwargs):
        return self


class Failure(_Try):
    def __init__(self, exception):
        self.exception = exception

    def get(self):
        raise self.exception

    def then(self, func, *args, **kwargs):
        return self

    def recover(self, func, *args, **kwargs):
        return try_call(func, self.exception, *args, **kwargs)