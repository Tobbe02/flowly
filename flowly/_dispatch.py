from __future__ import print_function, division, absolute_import

import inspect


class Dispatch(object):
    def __init__(self, mapping=None, parent=None, name='<unnamed>'):
        if mapping is None:
            mapping = {}

        self.name = name
        self.mapping = mapping
        self.conditional_bindings = {}
        self.parent = parent

    def inherit(self, name='<unnamed child>'):
        self_type = type(self)
        return self_type({}, self, name=name)

    def bind(self, type, func=None):
        def impl(func):
            if not isinstance(type, list):
                types = {type}

            else:
                types = set(type)

            duplicates = types & set(self.mapping)
            if duplicates:
                raise ValueError('rebinding of {} in {}, use inheritance'
                                 .format(duplicates, self.name))

            for t in types:
                self.mapping[t] = func

            return func

        if func is not None:
            impl(func)
            return self

        return impl

    def bind_conditional(self, module, func=None):
        def impl(func):
            self.conditional_bindings[module] = func
            return func

        if func is not None:
            impl(func)
            return self

        return impl

    def default(self, impl):
        self.default_func = impl
        return self

    def default_func(self, obj, *args, **kwargs):
        if self.parent is None:
            raise ValueError("cannot lookup {} for {}".format(type(obj), obj))

        return self.parent.default_func(obj, *args, **kwargs)

    def __call__(self, obj, *args, **kwargs):
        f = self.lookup(obj)
        return f(obj, *args, **kwargs)

    def lookup(self, obj):
        t = type(obj)
        match = self._lookup_exact(t)

        if match is not None:
            return match

        for match in self._lookup_walk_mro(t):
            return match

        return self.default_func

    def _lookup_exact(self, t):
        try:
            return self.mapping[t]

        except KeyError:
            pass

        if self._lookup_execute_conditional_bindings(t):
            return self._lookup_exact(t)

        if self.parent is None:
            return

        return self.parent._lookup_exact(t)

    def _lookup_walk_mro(self, t):
        for alt in inspect.getmro(t):
            try:
                match = self.mapping[alt]

            except KeyError:
                pass

            else:
                self.mapping[t] = match
                yield match

        if self.parent is not None:
            for match in self.parent._lookup_walk_mro(t):
                yield match

    def _lookup_execute_conditional_bindings(self, t):
        root, _1, _2,  = t.__module__.partition('.')

        try:
            func = self.conditional_bindings.pop(root)

        except KeyError:
            return False

        else:
            func(self)
            return True
