import itertools as it
import operator


def pretty(obj):
    try:
        impl = obj._flowly_pp_

    except AttributeError:
        raise ValueError("cannot describe {!r}".format(obj))

    return "\n".join(impl())


class flowly_base(object):
    def _flowly_pp_(self, indent=0):
        def describe(obj, indent):
            try:
                impl = obj._flowly_pp_

            except AttributeError:
                pass

            else:
                return impl(indent + 1)

            return []

        prefix = "  " * indent

        if indent == 0:
            yield "{}{!r}".format(prefix, self)

        for (k, v) in self._flowly_items_():
            yield "{}- {}: {!r}".format(prefix, k, v)

            for line in describe(v, indent):
                yield line

    def _flowly_items_(self):
        raise NotImplementedError()


class _unset(object):
    """sentinel object to indicate unset arguments
    """
    pass


def eval_expr(obj, expr):
    try:
        eval_func = expr._flowly_eval_

    except AttributeError:
        return expr

    else:
        return eval_func(obj)


class eval_(object):
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, obj):
        return eval_expr(obj, self.expr)


class expr(flowly_base):
    def __init__(self):
        pass

    def _flowly_eval_(self, obj):
        raise NotImplementedError()

    def __lt__(self, other):
        return lit(operator.lt)(self, other)

    def __le__(self, other):
        return lit(operator.le)(self, other)

    def __eq__(self, other):
        return lit(operator.eq)(self, other)

    def __ne__(self, other):
        return lit(operator.ne)(self, other)

    def __ge__(self, other):
        return lit(operator.ge)(self, other)

    def __gt__(self, other):
        return lit(operator.gt)(self, other)

    def __and__(self, other):
        return lit(operator.and_)(self, other)

    def __rand__(self, other):
        return lit(operator.and_)(other, self)

    def __or__(self, other):
        return lit(operator.or_)(self, other)

    def __ror__(self, other):
        return lit(operator.or_)(other, self)

    def __neg__(self):
        return lit(operator.neg)(self)

    def __pos__(self):
        return lit(operator.pos)(self)

    def __abs__(self):
        return lit(operator.abs)(self)

    def __invert__(self):
        return lit(operator.invert)(self)

    def __getattr__(self, name):
        return lit(getattr)(self, name)

    def __getitem__(self, key):
        return lit(operator.getitem)(self, key)

    def __call__(self, *args, **kwargs):
        return call_expr(self, *args, **kwargs)

    def __len__(self):
        # __len__ has to return an int, it cannot be work with proxy objects
        raise NotImplementedError("len(...) is not supported")

    # auto-generated
    # for op in [
    #     "add", "sub", "mul", "matmul", "div", "truediv", "floordiv", "mod",
    #     "divmod", "pow", "lshift", "rshift", "xor"
    # ]:
    #     print("    def __{}__(self, other):".format(op))
    #     print("        return lit(operator.{})(self, other)".format(op))
    #     print("")
    #     print("    def __r{}__(self, other):".format(op))
    #     print("        return lit(operator.{})(other, self)".format(op))
    #     print("")

    def __add__(self, other):
        return lit(operator.add)(self, other)

    def __radd__(self, other):
        return lit(operator.add)(other, self)

    def __sub__(self, other):
        return lit(operator.sub)(self, other)

    def __rsub__(self, other):
        return lit(operator.sub)(other, self)

    def __mul__(self, other):
        return lit(operator.mul)(self, other)

    def __rmul__(self, other):
        return lit(operator.mul)(other, self)

    def __matmul__(self, other):
        return lit(operator.matmul)(self, other)

    def __rmatmul__(self, other):
        return lit(operator.matmul)(other, self)

    def __div__(self, other):
        return lit(operator.div)(self, other)

    def __rdiv__(self, other):
        return lit(operator.div)(other, self)

    def __truediv__(self, other):
        return lit(operator.truediv)(self, other)

    def __rtruediv__(self, other):
        return lit(operator.truediv)(other, self)

    def __floordiv__(self, other):
        return lit(operator.floordiv)(self, other)

    def __rfloordiv__(self, other):
        return lit(operator.floordiv)(other, self)

    def __mod__(self, other):
        return lit(operator.mod)(self, other)

    def __rmod__(self, other):
        return lit(operator.mod)(other, self)

    def __divmod__(self, other):
        return lit(operator.divmod)(self, other)

    def __rdivmod__(self, other):
        return lit(operator.divmod)(other, self)

    def __pow__(self, other):
        return lit(operator.pow)(self, other)

    def __rpow__(self, other):
        return lit(operator.pow)(other, self)

    def __lshift__(self, other):
        return lit(operator.lshift)(self, other)

    def __rlshift__(self, other):
        return lit(operator.lshift)(other, self)

    def __rshift__(self, other):
        return lit(operator.rshift)(self, other)

    def __rrshift__(self, other):
        return lit(operator.rshift)(other, self)

    def __xor__(self, other):
        return lit(operator.xor)(self, other)

    def __rxor__(self, other):
        return lit(operator.xor)(other, self)


class call_expr(expr):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _flowly_eval_(self, obj):
        func = eval_expr(obj, self.func)
        args = [eval_expr(obj, arg) for arg in self.args]
        kwargs = {k: eval_expr(obj, arg) for (k, arg) in self.kwargs.items()}

        return func(*args, **kwargs)

    def _flowly_items_(self):
        return chained(
            [("func", self.func)], enumerate(self.args), self.kwargs.items()
        )


class lit(expr):
    def __init__(self, value):
        self.value = value

    def _flowly_eval_(self, obj):
        return self.value

    def _flowly_items_(self):
        return [("value", self.value)]


class side_effect(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return side_effect_call(self.func, args, kwargs)


class side_effect_call(object):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj):
        expr = lit(self.func)(*self.args, **self.kwargs)
        eval_expr(obj, expr)
        return obj


class callable_base(flowly_base):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def call(self, obj, *args, **kwargs):
        raise NotImplementedError()

    @property
    def expr(self):
        return lit(self.call)(this, *self.args, **self.kwargs)

    def _flowly_eval_(self, obj):
        return eval_expr(obj, self.expr)

    def _flowly_items_(self):
        return chained(enumerate(self.args), self.kwargs.items())


class this_impl(expr):
    def _flowly_eval_(self, obj):
        return obj

    def _flowly_items_(self):
        return []


this = this_impl()


def create_callable_pipe(*args):
    return callable_pipe(args)


def pipe(obj=_unset):
    return bound_pipe(obj) if obj is not _unset else unbound_pipe()


pipe.func = create_callable_pipe


class bound_pipe(object):
    """Allow easy chaining of transformations of an object.
    """
    def __init__(self, obj):
        self.obj = obj

    def __or__(self, transform):
        return bound_pipe(pipe_eval(self.obj, transform))

    def __pos__(self):
        return self.obj

    def __neg__(self):
        return None


class unbound_pipe(flowly_base):
    def __init__(self, transforms=()):
        self.transforms = list(transforms)

    def __or__(self, transform):
        return unbound_pipe(self.transforms + [transform])

    def _flowly_items_(self):
        return list(enumerate(self.transforms))


    def _flowly_eval_(self, obj):
        current = bound_pipe(obj)

        for transform in self.transforms:
            current = current | transform

        return +current


class callable_pipe(flowly_base):
    def __init__(self, args, transforms=()):
        self.args = args
        self.transforms = list(transforms)

    def __or__(self, transform):
        return callable_pipe(self.args, self.transforms + [transform])

    def _flowly_items_(self):
        return list(enumerate(self.transforms))

    def __call__(self, *args, **kwargs):
        current = bound_pipe(build_argument(self.args, args, kwargs))

        for transform in self.transforms:
            current = current | transform

        return +current


def build_argument(argspec, args, kwargs):
    named_args = dict(zip(argspec, args))
    for name, arg in kwargs.items():
        if name in named_args:
            raise TypeError("keyword argument {} specified multiple times".format(name))

        named_args[name] = arg

    missing_args = set(argspec) - set(named_args)
    if missing_args:
        raise TypeError("missing arguments {}".format(missing_args))

    result = argument(args[len(argspec):])
    for name, value in named_args.items():
        setattr(result, name, value)

    return result


class argument(object):
    def __init__(self, args):
        self.__args = args

    def __getitem__(self, idx):
        return self.__args[idx]

    def __len__(self):
        return len(self.__args)

    def __iter__(self):
        return iter(self.__args)


def pipe_eval(obj, expr):
    try:
        eval_func = expr._flowly_eval_

    except AttributeError:
        # prevent exception chaining
        pass

    else:
        return eval_func(obj)

    return expr(obj)


def chained(*args):
    return list(it.chain(*args))


def wrap(module_like):
    """Wrap every callable inside a module-like object by an unbound_callable.
    """
    return namespace(
        (name, unbound_callable(var))
        for name, var in vars(module_like).items()
        if not name.startswith("_") and callable(var)
    )


class namespace(object):
    def __init__(self, items):
        for key, value in items:
            setattr(self, key, value)


class unbound_callable(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return bound_callable(self.func, args, kwargs)


class bound_callable(object):
    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _flowly_eval_(self, obj):
        return self.func(obj, *self.args, **self.kwargs)
