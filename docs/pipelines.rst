Building Data Pipelines
=======================

Often data processing involves multiple transformations of data into more and
more complex objects.

From function calls to graphs
-----------------------------

Python offers a number of utility functions for lists, like ``map``,
``filter``, and ``reduce``.
For example say you wanted to compute the square root  of the sum the square of
all even numbers between 0 and 99.
In terms of said utility functions you can express this operation as::

    from functools import reduce
    import math
    import operator as op

    data = range(100)
    math.sqrt(
        reduce(
            op.add,
            map(lambda x: x ** 2.0, filter(lambda x: x % 2 == 0, data))
        )
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

:func:`flowly.tz.chained` represents the application of multiple functions,
one after the other, and the curried namespace of ``toolz`` allows to bind
the first argument of said utility functions without executing them immediately.

The second variant arguably simplifies the structure of the program and has to
additional benefit of being easier to compose.
``transform`` can easily be placed into larger chains of transformations without
having to be changed.
Finally, the second variant separates definition of the operations from
the execution.
This way the operations can be reinterpreted, i.e., for parallel execution, as
will be discussed below.

While it may seem that this method only allows to write linear pipelines,
general DAGs can easily built be using dictionaries. To simplify this pattern
``flowly`` offers the following helpers:

- :func:`flowly.tz.build_dict`
- :func:`flowly.tz.itemsetter`

Finally, flowly offser support for both data and operation parallelism via its
:func:`flowly.tz.apply_map_concat` function.
It applies a list of functions over a list of objects in parallel and
concatenates the results into a single, large iterable.

Parallel execution
------------------

Flowly can reinterpret many existing elements of computation graphs to be
executed on top of dask bags.
Applying the transformation to a :class:`dask.bag.Bag` is a simple matter of
calling :func:`flowly.dsk.apply` and calling
:meth:`compute<dask.bag.Bag.compute>` on the result::

    import dask.bag as db
    from flowly.dsk import apply

    data = db.from_sequence([1, 2, 3, 4], npartitions=2)
    result = apply(transform, data)
    print(result.compute())

Since dask also supports distributed executors, computing the result on a
cluster becomes a matter of simply writing::

    from distributed import Client
    client = Client('127.0.0.1:8786')
    print(result.compute(get=client.get))

The DAG primitives that are understood can easily be adapted by specifying the
``rules`` argument to :func:`flowly.dsk.apply`. Out of the box, the following
DAG primitives are supported:

- :func:`builtins.any()<any>`
- :func:`builtins.all()<all>`
- :func:`builtins.min()<min>`
- :func:`builtins.max()<max>`
- :func:`builtins.sum()<sum>`
- :func:`itertools.chain.from_iterable()<itertools.chain>`:
  often called "flatten" in  other frameworks.
- :func:`toolz.compose()<toolz.functoolz.compose>`
- :func:`toolz.concat()<toolz.itertoolz.concat>`:
  often called "flatten" in  other frameworks.
- :func:`toolz.count()<toolz.itertoolz.count>`
- :func:`toolz.unique()<toolz.itertoolz.unique>` (without key function)
- :func:`toolz.curried.filter()<filter>`
- :func:`toolz.curried.map()<map>`
- :func:`toolz.curried.mapcat()<toolz.itertoolz.mapcat>`:
  often called `flatmap` in other frameworks.
  First apply a function and then flatten the result.
- :func:`toolz.curried.pluck()<toolz.itertoolz.pluck>`
- :func:`toolz.curried.random_sample()<toolz.itertoolz.random_sample>`
- :func:`toolz.curried.reduce()<functools.reduce>`.
  The ``intial`` keyword is currently unsupported and ``reudce``
  requires to collect all partitions on a single worker, which may be
  inefficient.
  If possible, prefer :func:`flowly.tz.reduction` which enables additional
  optimizations.
- :func:`toolz.curried.remove()<toolz.itertoolz.remove>`
- :func:`toolz.curried.take()<toolz.itertoolz.take>`
- :func:`toolz.curried.topk()<toolz.itertoolz.topk>`
- :func:`flowly.tz.apply_concat`
- :func:`flowly.tz.apply_map_concat`
- :func:`flowly.tz.build_dict`
- :func:`flowly.tz.chained`
- :func:`flowly.tz.frequencies`
- :func:`flowly.tz.groupby`
- :func:`flowly.tz.itemsetter`
- :func:`flowly.tz.kv_keymap`
- :func:`flowly.tz.kv_reduceby`
- :func:`flowly.tz.reductionby`
- :func:`flowly.tz.kv_valmap`
- :func:`flowly.tz.reduceby`
- :func:`flowly.tz.reduction`
- :func:`flowly.tz.seq`

Any other callable is applied as is to the given object, i.e.,
``apply(show, obj)`` merely calls :func:`flowly.tz.show` on the object.
This fallback is offered to make extending pipelines with custom code easy.

Difference between reduction and reduce functions
-------------------------------------------------

While the reduction and reduce function offer similar functionality they are
fundamentally different.
The former take functions that operations on lists of values, e.g., :func:`sum`,
where the latter take function that operate on two values at a time, e.g.,
:func:`operator.add`.

Working with key-value pairs
----------------------------

Often it is natural to work with key-value pairs, where the key expresses some
group membership.
Often using the general reduction, reduce, and map functionality introduces
additional overhead, since the require additional steps to operate on the value
part of the items.

To simplify the process flowly offers the following helper functions:

* :func:`flowly.tz.kv_aggregateby`
* :func:`flowly.tz.kv_keymap`
* :func:`flowly.tz.kv_valmap`
* :func:`flowly.tz.kv_reduceby`
* :func:`flowly.tz.kv_reductionby`

All these methods take a list of key-value pairs as an input and return another
list of key value pairs.
For the reduction-by and reduce-by steps, the key of each item is used to define
the groups.

To lift an existing transform to operate on values grouped by the key part of a
list of key-value pairs use :func:`flowly.tz.kv_transform`.
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


Reference
---------

Execution
~~~~~~~~~

.. autofunction:: flowly.tz.apply

.. autofunction:: flowly.tz.pipe

Buildings Blocks
~~~~~~~~~~~~~~~~

.. autofunction:: flowly.tz.aggregate

.. autofunction:: flowly.tz.apply_concat

.. autofunction:: flowly.tz.apply_map_concat

.. autofunction:: flowly.tz.build_dict

.. autofunction:: flowly.tz.chained

.. function:: flowly.tz.collect

    shorthand for ``reduction(None, list)``.
    Will collect all values into a single list.

.. autofunction:: flowly.tz.frequencies

.. autofunction:: flowly.tz.groupby

.. autofunction:: flowly.tz.itemsetter

.. autofunction:: flowly.tz.kv_aggregateby

.. autofunction:: flowly.tz.kv_keymap

.. autofunction:: flowly.tz.kv_reduceby

.. autofunction:: flowly.tz.kv_reductionby

.. autofunction:: flowly.tz.kv_transform

.. autofunction:: flowly.tz.kv_valmap

.. autofunction:: flowly.tz.reduceby

.. autofunction:: flowly.tz.reduction

.. autofunction:: flowly.tz.reductionby

.. autofunction:: flowly.tz.seq

Dask Support
~~~~~~~~~~~~

.. autofunction:: flowly.dsk.apply_to_local

.. autofunction:: flowly.dsk.apply

.. autofunction:: flowly.dsk.get_default_rules
