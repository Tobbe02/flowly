from __future__ import print_function, division, absolute_import

import functools as ft
import operator as op

from ._base import is_flowly_expr
from ._interface import predicate, unary

__all__ = [
    'filter',

    'drop_index',
    'drop_columns',
]


# dplyr verbs

class filter(object):
    """Filter a dataframe by a set of conditions.

    The expressions can either be callables or flowly expressions.

    Examples::

        from toolz import pipe
        from flowly import df as fdf

        # no filter
        pipe(df, fdf.filter())

        # flowly expressions
        pipe(df, fdf.filter(this.a == this.b, this.a % 2 == 0))

        # callables
        pipe(
            df,
            fdf.filter(lambda df: (df['a'] == df['b']) & (this['a'] % 2 == 0))
        )

    """
    def __init__(self, *exprs):
        self.exprs = list(exprs)

    def __call__(self, df):
        if not self.exprs:
            return df

        mask = ft.reduce(op.and_, (predicate(expr)(df) for expr in self.exprs))
        return df[mask]


def drop_index(df):
    """Drop the index from a dataframe.

    Example::

        from toolz import pipe
        from flowly import df as fdf

        pipe(df, fdf.drop_index)
    """
    return df.reset_index(drop=True)


class drop_columns(object):
    """Drop given columns from a dataframe.

    Column names can be given as strings or as get-column-like flowly
    expression. Nonexisting columns are ignored.

    Examples::

        from toolz import pipe
        from flowly import this, df as fdf

        pipe(df, fdf.drop_columns('a', 'b'))

        pipe(df, fdf.drop_columns(this.a, this['b']))
    """
    def __init__(self, *cols):
        self.cols = list(cols)

    def __call__(self, df):
        cols = set(self._get_col_name(col) for col in self.cols)
        selected_columns = [col for col in df.columns if col not in cols]
        return df[selected_columns]

    def _get_col_name(self, col):
        if not is_flowly_expr(col):
            return col

        return unary(col)(_helper_get_column_names())


class _helper_get_column_names(object):
    def __getattr__(self, name):
        return name

    def __getitem__(self, name):
        return name


# class _assign_base(object):
#     def __init__(self, *args, **kwargs):
#         self.expressions = {}
#         self.expressions.update(dict(enumerate(args)))
#         self.expressions.update(kwargs)
#
#     def _eval(self, df):
#         return {k: eval_expr(df, v) for (k, v) in self.expressions.items()}
#
#
# class assign(_assign_base):
#     def __call__(self, df):
#         return df.assign(**self._eval(df))
#
#
# class select(_assign_base):
#     def __call__(self, df):
#         return pd.DataFrame(self._eval(df))
#
#
# def eval_(expr):
#     return ft.partial(eval_expr, expr=expr)
#
#
# class foreach_column(object):
#     def __init__(self, expr):
#         self.expr = expr
#
#     def _flowly_eval_(self, obj):
#         data = {
#             col: +(pipe(obj[col]) | self.expr)
#             for col in obj.columns
#         }
#
#         max_dim = max(len(np.shape(v)) for v in data.values())
#
#         if max_dim == 0:
#             return pd.Series(data)
#
#         elif max_dim == 1:
#             return pd.DataFrame(data)
#
#         else:
#             raise ValueError("cannot combine results with max dim {}".format(max_dim))
#
#
# class foreach_group(flowly_base):
#     def __init__(self, expr):
#         self.expr = expr
#
#     def _flowly_eval_(self, obj):
#         for key, df in obj:
#             +(pipe(Group(key, df)) | self.expr)
#
#         return obj
#
#     def _flowly_items_(self):
#         return [("expr", self.expr)]
#
#
# class plot(callable_base):
#     def call(self, obj, *args, **kwargs):
#         obj.plot(*args, **kwargs)
#         return obj
#
#
# class groupby(callable_base):
#     def call(self, obj, *args, **kwargs):
#         return obj.groupby(*args, **kwargs)
#
#     def foreach(self, expr):
#         return groupby_foreach(groupby_expr=self.expr, foreach_expr=expr)
#
#
# class groupby_foreach(flowly_base):
#     def __init__(self, groupby_expr, foreach_expr):
#         self.groupby_expr = groupby_expr
#         self.foreach_expr = foreach_expr
#
#     def _flowly_eval_(self, obj):
#         for key, df in +(pipe(obj) | self.groupby_expr):
#             +(pipe(Group(key, df)) | self.foreach_expr)
#
#         return obj
#
#     def _flowly_items_(self):
#         return [
#             ("groupby_expr", self.groupby_expr),
#             ("foreach_expr", self.foreach_expr)
#         ]
#
#
# class Group(object):
#     def __init__(self, key, df):
#         self.key = key
#         self.df = df
#
#     def __getattr__(self, name):
#         return getattr(self.df, name)
#
#     def __getitem__(self, name):
#         return self.df[name]
