Miscellaneous helpers
=====================

Output Helper:

- :func:`flowly.ipy.printmk`
- :func:`flowly.tz.printf`
- :func:`flowly.tz.show`

Notebook utilities:

- :func:`flowly.ipy.init`
- :func:`flowly.ipy.notify`
- :func:`flowly.ipy.download_csv`

Monads:

- :func:`flowly.tz.optional`
- :func:`flowly.tz.try_call`

Hashing:

- :func:`flowly.hashing.base_hash`
- :func:`flowly.hashing.functional_hash`
- :func:`flowly.hashing.ignore_globals`

Uncategorized:

- :func:`flowly.tz.raise_`
- :func:`flowly.tz.timed`

Reference
---------

.. autofunction:: flowly.hashing.base_hash

.. autofunction:: flowly.hashing.functional_hash

.. autofunction:: flowly.hashing.ignore_globals

.. autofunction:: flowly.ipy.download_csv

.. autofunction:: flowly.ipy.init

.. autofunction:: flowly.ipy.notify

.. autofunction:: flowly.ipy.printmk

.. autofunction:: flowly.tz.printf

.. autofunction:: flowly.tz.optional

.. autofunction:: flowly.tz.raise_

.. function:: flowly.tz.show(obj)

    Helper to show the value of expression inspired by q. It aims to be easy to
    integrate print statements into existing expressions without having to
    rewrite the code just for debugging purposes.

    Usage::

        from flowly.tz import show
        a = 3.

        # prints 4.0 to stdout
        result = show | 1 + a

        from pandas import pd
        df = pd.DataFrame(...)

        # print the dataframe to stdout
        df.pipe(show)

    To change the format string use the modulo operator as in::

        # prints 5.00 % to stdout
        p = show % '{:.2%}'| 0.05

.. autofunction:: flowly.tz.timed(tag=None)

.. autofunction:: flowly.tz.try_call
