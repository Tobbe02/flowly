Distributed Helpers
===================

Helpers to work with `distributed`_.

Example
-------

To start the distributed scheduler together four workers and compute the result
of a dask value use::

    from flowly.dst import LocalCluster

    with LocalCluster(workers=4) as cluster:
        result = value.compute(get=cluster.get)

Reference
---------

.. autoclass:: flowly.dst.LocalCluster

.. _distributed: https://distributed.readthedocs.io/en/latest/
