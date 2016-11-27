from ._base import flowly_base, eval_expr, callable_base, expr, pipe, lit, wrapped, this

import collections
import functools as ft
import itertools as it

import numpy as np
import pandas as pd
import six

# TODO: add proper __all__

cut = lit(pd.cut)
qcut = lit(pd.qcut)


from .mpl import plot_params


@wrapped
def drop_index(obj):
    return obj.reset_index(drop=True)


@wrapped
def drop_columns(obj, *cols):
    cols = set(cols)
    return obj[[col for col in obj.columns if col not in cols]]


@wrapped
def query(obj, mask):
    return obj[mask]


class colgen(type):
    def __getattr__(self, name):
        return col(name)


class col(six.with_metaclass(colgen, expr)):
    def __init__(self, name):
        self.name = name

    def _flowly_eval_(self, df):
        return df[self.name]

    def _flowly_items_(self):
        return [("name", self.name)]


class assign_base(object):
    def __init__(self, *args, **kwargs):
        self.expressions = {}
        self.expressions.update(dict(enumerate(args)))
        self.expressions.update(kwargs)

    def _eval(self, df):
        return {k: eval_expr(df, v) for (k, v) in self.expressions.items()}

class assign(assign_base):
    def __call__(self, df):
        return df.assign(**self._eval(df))


class select(assign_base):
    def __call__(self, df):
        return pd.DataFrame(self._eval(df))


def eval_(expr):
    return ft.partial(eval_expr, expr=expr)


class foreach_column(object):
    def __init__(self, expr):
        self.expr = expr

    def _flowly_eval_(self, obj):
        data = {
            col: +(pipe(obj[col]) | self.expr)
            for col in obj.columns
        }

        max_dim = max(len(np.shape(v)) for v in data.values())

        if max_dim == 0:
            return pd.Series(data)

        elif max_dim == 1:
            return pd.DataFrame(data)

        else:
            raise ValueError("cannot combine results with max dim {}".format(max_dim))


class staged_aggregate(object):
    def __init__(self, what, keys, stages, **kwargs):
        self.what = what
        self.keys = keys
        self.stages = stages

        self.df = kwargs.get("df", False)
        self.name = kwargs.get("name")

        assert len(self.stages) <= len(self.keys)

    def _flowly_eval_(self, obj):
        what = self.what
        for idx, stage in enumerate(self.stages):
            obj = +(pipe(obj) | this.groupby(self.keys[idx:])[what] | stage)
            what = obj.name
            obj = obj if (idx + 1) == len(self.stages) else obj.reset_index()

        if self.name is not None:
            obj.name = self.name

        return obj.reset_index() if self.df else obj


class foreach_group(flowly_base):
    def __init__(self, expr):
        self.expr = expr

    def _flowly_eval_(self, obj):
        for key, df in obj:
            +(pipe(Group(key, df)) | self.expr)

        return obj

    def _flowly_items_(self):
        return [("expr", self.expr)]


class plot(callable_base):
    def call(self, obj, *args, **kwargs):
        obj.plot(*args, **kwargs)
        return obj


class groupby(callable_base):
    def call(self, obj, *args, **kwargs):
        return obj.groupby(*args, **kwargs)

    def foreach(self, expr):
        return groupby_foreach(groupby_expr=self.expr, foreach_expr=expr)


class groupby_foreach(flowly_base):
    def __init__(self, groupby_expr, foreach_expr):
        self.groupby_expr = groupby_expr
        self.foreach_expr = foreach_expr

    def _flowly_eval_(self, obj):
        for key, df in +(pipe(obj) | self.groupby_expr):
            +(pipe(Group(key, df)) | self.foreach_expr)

        return obj

    def _flowly_items_(self):
        return [
            ("groupby_expr", self.groupby_expr),
            ("foreach_expr", self.foreach_expr)
        ]


class Group(object):
    def __init__(self, key, df):
        self.key = key
        self.df = df

    def __getattr__(self, name):
        return getattr(self.df, name)

    def __getitem__(self, name):
        return self.df[name]
