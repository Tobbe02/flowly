"""Additional toolz.
"""
from __future__ import print_function, division, absolute_import

try:
    import __builtins__ as builtins

except ImportError:
    import builtins

import contextlib
import functools as ft
import inspect
import itertools as it
import logging
import operator as op
import time

import toolz

_logger = logging.getLogger(__name__)

__all__ = [
    'Failure',
    'Just',
    'Nothing',
    'Success',
    'apply_concat',
    'apply_map_concat',
    'build_dict',
    'chained',
    'contextlib',
    'division',
    'frequencies',
    'groupby',
    'itemsetter',
    'kv_transform',
    'kv_keymap',
    'kv_reduceby',
    'kv_reductionby',
    'kv_valmap',
    'optional',
    'printf',
    'raise_',
    'reduceby',
    'reduction',
    'reductionby',
    'seq',
    'show',
    'timed',
    'try_call',
]


class _ShowImpl(object):
    def __init__(self, fmt='{!r}'):
        self.fmt = fmt

    def __call__(self, obj):
        print(self.fmt.format(obj))
        return obj

    def __or__(self, obj):
        return self(obj)

    def __mod__(self, fmt):
        return _ShowImpl(fmt)


show = _ShowImpl()


def printf(fmt, *args, **kwargs):
    """Wrapper around print / str.format for interactive use.
    """
    print(fmt.format(*args, **kwargs))


class itemsetter(object):
    """Set key/values of mapping where the values are created by callables.

    :param Dict[Any,Callable[Any,Any]] prototypes:
        each prototype is a mapping from keys to callables.

    :param Callable[Any,Any] assigments:
        all assignment are converted into a new prototype appended to the
        existing ones.

    For example::

        >>> import operator as op
        >>> from flowly.tz import chained, itemsetter
        >>> transform = itemsetter(
        ...     sum=chained(op.itemgetter('values'), sum),
        ...     max=chained(op.itemgetter('values'), max),
        ...     min=chained(op.itemgetter('values'), min),
        ... )
        >>> obj = {'values': [1, 2, 3, 4, 5]}
        >>> transform(obj)
        ... {'values': [1, 2, 3, 4, 5], 'sum': 15, 'max': 5, 'min': 1}
    """
    def __init__(self, *prototypes, **assigments):
        self.assigments = list(prototypes) + [assigments]

    def __call__(self, obj):
        obj = obj.copy()
        for assigment in self.assigments:
            for k, func in assigment.items():
                obj[k] = func(obj)

        return obj


class build_dict(object):
    """Build a mapping where the values are created by callables.

    :param Dict[Any,Callable[Any,Any]] prototypes:
        each prototype is a mapping from keys to callables.

    :param Callable[Any,Any] assigments:
        all assignment are converted into a new prototype appended to the
        existing ones.


    For example::

        >>> import operator as op
        >>> from flowly.tz import chained, build_dict
        >>> transform = build_dict(
        ...     sum=sum,
        ...     max=max,
        ...     min=min,
        ...     values=lamba x: x,
        ... )
        >>> transform([1, 2, 3, 4, 5])
        ... {'values': [1, 2, 3, 4, 5], 'sum': 15, 'max': 5, 'min': 1}
    """
    def __init__(self, *prototypes, **assigments):
        self.assigments = list(prototypes) + [assigments]

    def __call__(self, obj):
        res = {}
        for assigment in self.assigments:
            for k, func in assigment.items():
                res[k] = func(obj)

        return res


class chained(object):
    """Represent the composition of functions.

    When the resulting object is called with a single argument, the passed
    object is transformed by passing it through all given functions.
    For example::

        a = chained(
            math.sqrt,
            math.log,
            math.cos,
        )(5.0)

    is equivalent to::

        a = 5.0
        a = math.sqrt(a)
        a = math.log(a)
        a = math.cos(a)

    Different chains can be composed via ``+``.
    For example, the chain above can be written as::

        chained(math.sqrt, math.log) + chained(math.cos)

    """
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
    """Apply different functions to each item and concatenate all results.

    Equivalent to::

        it.chain.from_iterable((func(item) for func in funcs) for item in obj)

    Each function should map from a single item to a transformed item.
    The result will be the concatenation of the transformed items of all
    functions in arbitrary order.
    Often it is useful to use non-pure functions, i.e., sampling procedures.
    Per default, dask will remove duplicate invocations of non-pure functions.
    flowly takes care to pass a unique component for each job and thereby will
    ensure that identical invocations of the function will not be skipped by
    dask when using :func:`flowly.dsk.apply`.

    :param Iterable[Callable[Any,...]] funcs:
        The functions to apply. Needs to be finite.

    :param Optional[int] chunks:
        A hint to parallel executors about the desired chunk size.


    For example, consider the application of some subsampling procedure with a
    reduction step::

        >>> from flowly.tz import apply_map_concat
        >>> from flowly.dsk import apply
        >>> transform = apply_map_concat([
        ...     chained(
        ...         subsample,
        ...         reduction
        ...     )
        ...     for _ in range(1000)
        ... ])
        >>> apply(transform, bag_of_objects)

    These steps would apply the subsample and reduction steps 1000 times to each
    object in the passed bag in parallel.
    The end result will be a list of the reduction results.
    """
    def __call__(self, obj):
        return it.chain.from_iterable(
            (func(item) for func in self.funcs)
            for item in obj
        )


def frequencies(obj):
    """In contrast to ``toolz.frequencies``, return ``(item, count)`` pairs.
    """
    result = {}

    for item in obj:
        result[item] = result.get(item, 0) + 1

    return list(result.items())


class groupby(object):
    """In contrast to ``toolz.groupby``, return ``(key, values)`` pairs.
    """
    def __init__(self, key):
        self.key = key

    def __call__(self, obj):
        result = {}
        for item in obj:
            result.setdefault(self.key(item), []).append(item)

        return list(result.items())


# TODO: add support for initial value
class reduceby(object):
    """In contrast to ``toolz.reduceby``, return ``(key, value)`` pairs.
    """
    def __init__(self, key, binop):
        self.key = key
        self.binop = binop

    def __call__(self, seq):
        return [
            (key, ft.reduce(self.binop, subseq))
            for (key, subseq) in groupby(self.key)(seq)
        ]


class reduction(object):
    """Reduce values via aggregate functions.

    :param Optional[Callable[Any,Iterable[Any]]] perpartition:
        a function that reduces a list of values to a single object.
        This function is used to generate intermediates before passing them
        to aggregate.
        If passed as ``None``, all values will be collected in a single list
        before applying the ``aggregate`` function.

    :param Callable[Any,Iterable[Any]] aggregate:
        a function that reduces a list of values to a single object.
        This function is used after perpartition to compute the global
        aggregate.

    :param Optional[int] split_every:
        a hint to the executor how to group intermediates.
    """
    def __init__(self, perpartition, aggregate, split_every=None):
        self.perpartition = perpartition
        self.aggregate = aggregate
        self.split_every = split_every

    def __call__(self, obj):
        if self.perpartition is None:
            return self.aggregate(obj)

        else:
            return self.aggregate([self.perpartition(obj)])


class reductionby(object):
    """Like :func:`flowly.tz.reduction` but on a group basis.

    :param Callable[Any,Any] key:
        return a key to group by.

    The other arguments are the same as for :func:`flowly.tz.reduction`.
    """
    def __init__(self, key, perpartition, aggregate, split_every=None):
        self.key = key
        self.perpartition = perpartition
        self.aggregate = aggregate
        self.split_every = split_every

    def __call__(self, obj):
        grouped = groupby(self.key)(obj)
        if self.perpartition is None:
            return [
                (key, self.aggregate(group))
                for (key, group) in grouped
            ]

        else:
            return [
                (key, self.aggregate([self.perpartition(group)]))
                for (key, group) in grouped
            ]


def seq(*items):
    """Turn one or multiple values into a list.

    Examples::

        >>> seq(1)
        [1]
        >>> seq(1, 2, 3)
        [1, 2, 3]
        >>> seq([1], [2])
        [[1], [2]]
    """
    return list(items)


def kv_transform(transform):
    """Let a transform operate on a list of key-value pairs, grouped by the key.

    This function may be useful to execute pre-defined transformations on top
    of different groups.
    It can transform the following building blocks:

    - :func:`toolz.concat` and
      :func:`itertools.chain.from_iterable()<itertools.chain>`:
      transform a list of the form ``[(k, [a, b, c]), ...]`` to
      ``[(k, a), (k, b), (k, c), ...]``.
    - :func:`toolz.curried.map()<map>`:
      transform a list of the form ``[(k, v), ...]`` to ``[(k, func(v)), ...]``
    - :func:`toolz.curried.reduce(binop)<functools.reduce>`:
      transform a list of the form ``[(k, a), (k, b), (k, c), ...]`` to
      ``[(k, binop(binop(a, b), c))]``.
    - :func:`flowly.tz.reduction`:
      transform a list of the form ``[(k, a), (k, b), (k, c), ...]`` to
      ``[(k, aggregate([a, b, c]), ...]``.
      The ``perpartition`` and  ``split_every`` are also supported.
    - :func:`flowly.tz.chained`: transform all steps of the chain.

    For example::

        # compute the sum of squares of a list of numbers
        sum_of_squares = chained(
            map(lambda x: x ** 2.0),
            reduction(None, sum),
        )

        # compute the sum of squares of even and odd numbers separately
        even_odd_sum_of_squares = chained(
            map(lambda x: (x % 2, x)),
            kv_transform(sum_of_squares),
        )

    """
    if isinstance(transform, chained):
        return chained(*[
            kv_transform(child) for child in transform.funcs
        ])

    elif isinstance(transform, toolz.curry) and (transform.func is builtins.map):
        return kv_valmap(transform.args[0])

    elif isinstance(transform, toolz.curry) and (transform.func is ft.reduce):
        return kv_reduceby(transform.args[0])

    elif isinstance(transform, reduction):
        return kv_reductionby(
            transform.perpartition, transform.aggregate,
            split_every=transform.split_every,
        )

    elif isinstance(transform, toolz.curry) and (transform.func is toolz.mapcat):
        return chained(
            toolz.curried.map(lambda item: [
                (item[0], transformed)
                for transformed in transform.args[0](item[1])
            ]),
            toolz.concat,
        )

    elif transform == it.chain.from_iterable or transform == toolz.concat:
        return chained(
            toolz.curried.map(lambda item: [
                (item[0], value) for value in item[1]
            ]),
            toolz.concat,
        )

    else:
        raise ValueError('cannot translate %s to a kv transform' % transform)


class kv_keymap(object):
    """Apply a function to the key part of each key-value pair.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, obj):
        return (
            (self.func(key), value)
            for (key, value) in obj
        )


class kv_valmap(object):
    """Apply a function to the value part of each key-value pair.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, obj):
        return (
            (key, self.func(value))
            for (key, value) in obj
        )


class kv_reduceby(object):
    """Reduce for key value pairs, only values are seend by the reducer.

    In contrast to :func:`flowly.tz.reduceby`, the data should be a list of
    key-value pairs.
    The groups are formed by the key part of each item and the reducer is only
    applied to the values.
    This function is designed to map from a list of key-value pairs to another
    list of key-value pairs.

    :param Callable[Any,Any,Any] binop:
        a binary function that maps to values to an aggregated value.
    """
    def __init__(self, binop):
        self.binop = binop

    def __call__(self, obj):
        return [
            (key, ft.reduce(self.binop, [val for _, val in group]))
            for (key, group) in groupby(op.itemgetter(0))(obj)
        ]


class kv_reductionby(object):
    """Reduction for key value pairs, only values are seend by the reducer.

    In contrast to :func:`flowly.tz.reductionby`, the data should be a list of
    key-value pairs.
    The groups are formed by the key part of each item and the reduction is only
    applied to the values.
    This function is designed to map from a list of key-value pairs to another
    list of key-value pairs.
    """
    def __init__(self, perpartition, aggregate, split_every=None):
        self.perpartition = perpartition
        self.aggregate = aggregate
        self.split_every = split_every

    def __call__(self, obj):
        if self.perpartition is None:
            reducer = self.aggregate

        else:
            def reducer(obj):
                return self.aggregate([self.perpartition(obj)])

        return [
            (key, reducer([val for _, val in group]))
            for (key, group) in groupby(op.itemgetter(0))(obj)
        ]


def optional(val):
    """Wrap any value with the optional moand.

    Usage::

        val = +optional(val).or_else(default_value).get()

        val = +optional(val).or_else_call(expensive_function, arg1, arg2).get()
    """
    return Just(val) if val is not None else Nothing()


def try_call(func, *args, **kwargs):
    """Try to call the function and return a Try monad.

    Usage::

        result = +try_call(func, arg1, arg2).recover(altnative_operation).get()
    """
    try:
        result = func(*args, **kwargs)

    except Exception as e:
        return Failure(e)

    else:
        return Success(result)


def raise_(exc_class, *args, **kwargs):
    """Construct and raise an exception as an expression.
    """
    raise exc_class(*args, **kwargs)


@contextlib.contextmanager
def timed(tag=None):
    """Time a codeblock and log the result.

    Usage::

        with timed():
            long_running_operation()

    :param any tag:
        an object used to identify the timed code block. It is printed with
        the time taken.
    """
    if tag is None:
        msg = 'took %s s'

    else:
        msg = '{} took %s s'.format(tag)

    logger = _get_caller_logger()
    start = time.time()
    yield
    end = time.time()
    logger.info(msg, end - start)


def _get_caller_logger(depth=2):
    stack = inspect.stack()

    if depth >= len(stack):  # pragma: no cover
        return logging.getLogger(__name__)

    # NOTE: python2 returns raw tuples, index 0 is the frame
    frame = stack[depth][0]
    name = frame.f_globals.get('__name__')
    return logging.getLogger(name)


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

    def transform(self, func, *args, **kwargs):
        return self

    def get(self):
        raise ValueError()

    def or_else(self, val):
        return optional(val)

    def or_else_call(self, func, *args, **kwargs):
        return optional(func(*args, **kwargs))


class Just(_Maybe):
    def __init__(self, value):
        self.value = value

    def transform(self, func, *args, **kwargs):
        return optional(func(self.value, *args, **kwargs))

    def get(self):
        return self.value

    def or_else(self, val):
        return self

    def or_else_call(self, func, *args, **kwargs):
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
