Working with dataframes
=======================

Overview
--------

dplyr verbs:

- :func:`flowly.df.filter`
- :obj:`flowly.df.slice` and :obj:`flowly.df.islice`

Other helpers:

- :func:`flowly.df.drop_columns`
- :func:`flowly.df.drop_index`

Differences to ...
------------------

- Pure callables
- No extensions of dataframes

Reference
---------

.. autofunction:: flowly.df.drop_columns

.. autofunction:: flowly.df.drop_index

.. autofunction:: flowly.df.filter

.. object:: flowly.df.slice, flowly.df.islice

    .. note::

        pandas slice interpretations are retained.

    Examples::

        pipe(df, fdf.slice[1:5])
        pipe(df, slice(5))

        pipe(df, islice[1:5])
        pipe(df, islice(5))
