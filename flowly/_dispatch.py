from __future__ import print_function, division, absolute_import

import inspect


class Dispatch(object):
    def __init__(self, mapping=None, parent=None):
        if mapping is None:
            mapping = {}

        self.mapping = mapping
        self.conditional_bindings = []
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

    def bind_if(self, condition):
        def impl(func):
            self.conditional_bindings.append((condition, func))
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

        for match in self._lookup_walk_mro(t):
            return match

        if self.parent is not None:
            return self.parent.lookup(obj)

        for _ in self._lookup_execute_conditional_bindings(obj):
            return self.lookup(obj)

        return self.default_func

    def _lookup_walk_mro(self, t):
        for alt in inspect.getmro(t):
            try:
                match = self.mapping[alt]

            except KeyError:
                pass

            else:
                self.mapping[t] = match
                yield match

    def _lookup_execute_conditional_bindings(self, obj):
        if not self.conditional_bindings:
            return

        matching_binders, non_matching_binders = _split(lambda t: t[0](obj), self.conditional_bindings)

        if len(matching_binders) > 1:
            raise RuntimeError("multiple condition bindings matched: {}".format(matching_binders))

        # prune conditional bindings
        self.conditional_bindings = non_matching_binders

        for condition, binder in matching_binders:
            if condition(obj):
                binder(self)
                yield None


def _split(pred, seq):
    result = {True: [], False: []}

    for item in seq:
        result[pred(item)].append(item)

    return result[True], result[False]
