Writing Data Pipelines
======================

Often data processing involves multiple transformations of data into more and
more complex objects.

From functions calls to DAGs
----------------------------

Python offers a number of utility functions for lists, like ``map``,
``filter``, and ``reduce``. For example say you wanted to sum the square of all
even numbers between 0 and 99. In terms of said utility functions you can
express this operation as::

    from functools import reduce
    import math
    import operator as op

    data = range(100)
    math.sqrt(
      reduce(op.add, map(lambda x: x ** 2.0,
      filter(lambda x: x % 2 == 0, data)))
    )

These operations can be rewritten by using the functionality offered by
``flowly`` and ``toolz`` into::

    from toolz.curried import filter, map, reduce
    from flowly.tz import chained

    transform = chained(
        filter(lambda x: x % 2 == 0),
        map(lambda x: 2 * x),
        reduce(op.add),
        math.sqrt,
    )

    data = range(100)
    transform(data)

:func:``flowly.tz.chained`` represents the application of multiple functions
one after the other and the curried namespace of ``toolz`` offers support bind
the first argument of said utility functions without executing it immediately.

The second variant arguably simplifies the structure of the program and has to
additional benefit of being easier to compose. ``transform`` can be placed
anywhere into an even large chain of transformations without having to be
changed. Finally, the second variant separates definition of the operations from
the execution. This way the operations can be reinterpreted as will be done
below when considering parallel execution.

While it may seem that this method only allows to write linear pipelines,
general DAGs can easily built be using dictionaries. To simplify this pattern
``flowly`` offers a couple of helpers:

- :func:`flowly.tz.build_dict`
- :func:`flowly.tz.itemsetter`

Consider the following example::

    TODO: add an example


Parallel execution
------------------

Flowly comes with builtin support for dask for many existing elements of
computation graphs. Applying the transformation to a ``dask.bag.Bag`` is a
simple matter of calling ``flowly.dsk.apply``:

    import dask.bag as db
    from flowly.dsk import apply

    data = db.from_sequence([1, 2, 3, 4], npartitions=2)
    result = apply(transform, data)

To compute the result using the standard dask executor use::

    print(result.compute())

Since dask also supports distributed executors, computing the result on a
cluster becomes a matter of simply writing::

    from distributed import Client
    client = Client('127.0.0.1:8786')
    result.compute(get=client.get)


The DAG primitives that are understood can easily be adapted by specifying the
``rules`` argument to :func:`flowly.dsk.apply`. Out of the box, the following
DAG primitives are supported:

- ``builtins.any``
- ``builtins.all``
- ``builtins.min``
- ``builtins.max``
- ``builtins.sum``
- ``itertools.chain.from_iterable``
- ``toolz.compose``
- ``toolz.concat``
- ``toolz.count``
- ``toolz.unique`` (without key function)
- ``toolz.curried.filter``
- ``toolz.curried.map``
- ``toolz.curried.mapcat``
- ``toolz.curried.pluck``
- ``toolz.curried.random_sample``
- ``toolz.curried.remove``. Note, the ``intial`` keyword is currently
    unsupported and ``reudce`` requires to collect all partitions on a
    single worker, which may be inefficient. If possible, prefer
    ``flowly.tz.reduction`` which enables additional optimizations.
- ``toolz.curried.remove``
- ``toolz.curried.take``
- ``toolz.curried.topk``
- :func:`flowly.tz.apply_concat`
- :func:`flowly.tz.apply_map_concat`
- :func:`flowly.tz.build_dict`
- :func:`flowly.tz.chained`
- :func:`flowly.tz.frequencies`
- :func:`flowly.tz.groupby`
- :func:`flowly.tz.itemsetter`
- :func:`flowly.tz.reduceby`
- :func:`flowly.tz.reduction`
- :func:`flowly.tz.seq`

Any other callable is applied as is to the given object, i.e.,
``apply(show, obj)``merely calls ``flowly.tz.show`` on object. This fallback
is offered to make extending pipelines with custom code easy.

Reference
---------

Buildings Blocks
~~~~~~~~~~~~~~~~

.. autofunction:: flowly.tz.apply_concat

.. autofunction:: flowly.tz.apply_map_concat

.. autofunction:: flowly.tz.build_dict

.. autofunction:: flowly.tz.chained

.. autofunction:: flowly.tz.frequencies

.. autofunction:: flowly.tz.groupby

.. autofunction:: flowly.tz.itemsetter

.. autofunction:: flowly.tz.reduceby

.. autofunction:: flowly.tz.reduction

.. autofunction:: flowly.tz.seq

Dask Support
~~~~~~~~~~~~

.. autofunction:: flowly.dsk.apply

.. autofunction:: flowly.dsk.get_default_rules
