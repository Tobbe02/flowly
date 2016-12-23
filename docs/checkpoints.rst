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

Reference
---------

.. autofunction:: flowly.checkpoint.add_checkpoints

.. autofunction:: flowly.checkpoint.clear_checkpoints

.. autofunction:: flowly.checkpoint.with_checkpoint

.. autoclass:: flowly.checkpoint.checkpoint
