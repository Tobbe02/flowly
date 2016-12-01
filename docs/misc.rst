Miscellaneous helpers
=====================

TODO: add overview

Reference
---------

.. autofunction:: flowly.tz.optional

.. autofunction:: flowly.tz.try_call

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

.. autofunction:: flowly.ipy.add_toc
