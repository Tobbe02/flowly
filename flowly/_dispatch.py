from __future__ import print_function, division, absolute_import

import inspect


class Dispatch(object):
    def __init__(self, mapping=None, parent=None):
        if mapping is None:
            mapping = {}

        self.mapping = mapping
        self.parent = parent

    def inherit(self):
        self_type = type(self)
        return self_type({}, self)

    def bind(self, type):
        def impl(func):
            if not isinstance(type, list):
                types = [type]

            else:
                types = type

            for t in types:
                self.mapping[t] = func
            return func

        return impl

    def default(self, impl):
        self.default_func = impl

    def default_func(self, obj, *args, **kwargs):   # pragma: no cover
        raise ValueError("cannot lookup {} for {}".format(type(obj), obj))

    def __call__(self, obj, *args, **kwargs):
        f = self.lookup(obj)
        return f(obj, *args, **kwargs)

    def lookup(self, obj):
        t = type(obj)

        try:
            return self.mapping[t]

        except KeyError:
            pass

        for alt in inspect.getmro(t):
            try:
                return self.mapping[t]

            except KeyError:
                pass

        if self.parent is not None:
            return self.parent.lookup(obj)

        return self.default_func
