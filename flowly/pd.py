from ._base import flowly_base, eval_expr, callable_base, expr, pipe, lit

import collections
import functools as ft
import itertools as it

import pandas as pd
import six

# TODO: add proper __all__

cut = lit(pd.cut)
qcut = lit(pd.qcut)


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
