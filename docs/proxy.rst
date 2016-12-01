Building expression graphs
==========================

At times it may be convenient to build expression graphs. First, expression
graphs add the possibility for optimizations. Second, they allow to construct
domain specific languages to improve readability. For example::

    from flowly import this

    pipe(
      df,
      filter(this.a == this.b),
      drop_columns(this.c, this.b),
    )

flowly supports such a pattern with its :obj:`flowly.this` object. It can be
used in almost arbitrary expressions. The result of evaluating this expression
will then be a graph that can be run independently. Currently, most support for
this pattern can be found in the dataframe support of flowly.

To interface with existing code, use the helpers :func:`flowly.unary`,
:func:`fowly.binary`, and :func:`flowly.predicate`.

Reference
---------

.. object :: flowly.this

    A proxy object. When used in expressions, it generates expression graphs.

.. autofunction:: flowly.unary

.. autofunction:: flowly.binary

.. function:: flowly.predicate

    An alias for :func:`flowly.unary`.
