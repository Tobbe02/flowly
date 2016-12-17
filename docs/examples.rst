Examples
========

The following examples demonstrate some common usages.
Since distributed allows to pickle lambdas and functions defined in
``__main__``, all examples work as-is in IPython notebooks.

Trivial parallelization
-----------------------

To apply the function ``func`` across a list of objects in parallel across
``16`` distributed workers, use::

    from flowly.dsk import apply_to_local
    from flowly.dst import LocalCluster

    from toolz.curried import map

    with LocalCluster(workers=16) as cluster:
        results = apply_to_local(
            map(func), objects,
            get=cluster.get,
            npartitions=100,
        )

.. seealso ::

    * :func:`toolz.curried.map`
    * :func:`flowly.dsk.apply_to_local`
    * :class:`flowly.dst.LocalCluster`


Word Count
----------

Of course, also the hello-world of parallel processing can be implemented using
flowly::

    # generate random input documents
    import random
    words = ['the', 'brown', 'fox', 'jumped', 'over', 'the', 'hedge']
    documents = [
        ' '.join(random.choice(words) for _ in range(random.randint(0, 100)))
        for _ in range(100)
    ]

    # count words
    from flowly.tz import kv_reductionby, chained
    from toolz import concat
    from toolz.curried import map

    count_words = chained(
        map(lambda doc: [(word, 1) for word in doc.split(' ')]),
        concat,
        kv_reductionby(sum, sum),
    )

    with LocalCluster() as cluster:
        word_counts = apply_to_local(
            count_words, documents,
            npartitions=20,
            get=cluster.get,
        )

.. seealso ::

    * :func:`toolz.concat`
    * :func:`flowly.dsk.apply_to_local`
    * :func:`flowly.tz.kv_reductionby`
    * :class:`flowly.dst.LocalCluster`

Bootstrap Analysis
------------------

The following example is a variation of the trivial parallelization example.
Suppose, you got two different groups that you watch over two days.
For each group you collect samples of some observable.
To compute the certainty of the results you could use bootstrapping.
The following example, computes 1000 bootstrap samples in parallel over 4
different input datasets and groups the final results by the group key.

::

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from flowly.tz import apply_map_concat, reductionby, show
    from flowly.dsk import apply_map_concat
    from flowly.dst import LocalCluster

    # generate random data
    objects = [
        {'group': 1, 'day': 0,
         'data': pd.DataFrame({'value': np.random.normal(1 + 5, 10, size=100)})},
        {'group': 2, 'day': 0,
         'data': pd.DataFrame({'value': np.random.normal(2 + 5, 10, size=100)})},
        {'group': 1, 'day': 1,
         'data': pd.DataFrame({'value': np.random.normal(1 + 10, 10, size=100)})},
        {'group': 2, 'day': 1,
         'data': pd.DataFrame({'value': np.random.normal(2 + 10, 10, size=100)})},
    ]


    # define the analysis steps
    def compute_bootstrapped_mean(d):
        d = d.copy()
        data = d.pop('data')

        d['value'] = (
            data
            .sample(frac=1.0, replace=True)
            ['value'].mean()
        )

        return d

    transform = chained(
        # compute 1000 bootstrap samples
        apply_map_concat([
            compute_bootstrapped_mean
            for _ in range(1000)
        ]),

        # collect the results into a single dataframe by the group key
        reductionby(lambda d: d['group'], None, pd.DataFrame),
    )

    # execute them in parallel
    with LocalCluster() as cluster:
        bootstrapped_results = apply_to_local(
            transform, objects,
            get=cluster.get,
        )

    bootstrapped_results = dict(bootstrapped_results)

    # and plot the results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='day', y='value', data=bootstrapped_results[1])
    plt.title('Group 1')

    plt.subplot(1, 2, 2)
    sns.boxplot(x='day', y='value', data=bootstrapped_results[2])
    plt.title('Group 2')


.. seealso ::

    * :func:`flowly.dsk.apply_to_local`
    * :class:`flowly.dst.LocalCluster`
    * :func:`flowly.tz.apply_map_concat`
    * :func:`flowly.tz.reductionby`
