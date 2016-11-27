"""Additional toolz.
"""
from __future__ import print_function, division, absolute_import


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


class chained(object):
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, obj):
        for func in self.funcs:
            obj = func(obj)

        return obj

    def __add__(self, other):
        return chained(*(list(self.funcs) + list(other.funcs)))


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
