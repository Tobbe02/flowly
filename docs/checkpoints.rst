Checkpointing
=============

.. warning::

  Checkpoint support is currently in development.
  While the API most likely will not change, there is a chance that
  some transformations are not yet properly supported.
  Please open an issue on github if you find such instances.

  Currently, there is also no support for dask pipelines executed via
  :func:`flowly.dsk.apply`.

By inserting checkpoints, parts of transformations are only executed if either
the input or the transformation itself have changed.
Consider the case, where the object is first transformed by a function with
considerable runtime and then the result is plotted.
The following pipeline will only execute the ``long_running_transform``
once for every input argument::

    from flowly.checkpoint import checkpoint
    from flowly.tz import chained

    _checkpoints = {}

    transform = chained(
        long_running_transform,
        checkpoint(target=_checkpoints),
        plot,
    )

To use the checkpointing, the transform has to be rewritten prior to use, this
step can be performed by using :func:`flowly.tz.apply`::

    from flowly.checkpoint import add_checkpoints
    from flowly.tz import apply

    apply(transform, obj, rewrites=[add_checkpoints])

Checkpoint support can be added as the default with :func:`functools.partial`,
as in ``apply = partial(apply, rewrites=[add_checkpoints])``.

Flowly comes with special support for interactive environments, such as
notebooks.
In particular, transforms defined inside the notebook will be reexecuted if
they are redefined.

In notebooks, the ``_checkpoints`` variable should be defined in a cell by
itself.
Thereby it will collect all execution results over time.
Memory can be freed by calling ``_checkpoints.clear()`` to remove all execution
results or by using :func:`flowly.checkpoint.clear_checkpoints` for more fine
grained control.


Stateful transformations
------------------------

At times transformations inside a chain may be stateful.
For example, requests to external services may keep track of prior requests to
implement rate limiting.
In such cases, checkpointing will not work as expected, since the internal state
of the transform will change with each request.
To circumvent such problems, use the decorator
:func:`flowly.hashing.ignore_globals`.

Say `client` implements rate limiting, but the result of ``execute_request`` is
full determined by the argument.
Then the use of ``client`` should be decorated as::

    @ignore_globals('client')
    def execute_request(arg):
        return client.request(arg=arg)

Even though, the state of ``client`` will change with each request, it will
only be executed once for each value of ``arg``.

Reference
---------

.. autofunction:: flowly.checkpoint.add_checkpoints

.. autofunction:: flowly.checkpoint.clear_checkpoints

.. autofunction:: flowly.checkpoint.with_checkpoint

.. autoclass:: flowly.checkpoint.checkpoint
