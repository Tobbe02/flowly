from __future__ import print_function, division, absolute_import

import pandas as pd
import toolz
import builtins

from ._dispatch import Dispatch
from .tz import chained

__all__ = [
    'as_frame',
    'add_dfply_support',
]


def as_frame(**kwargs):
    """Helper function to simplify code generating dataframes.
    """
    return pd.DataFrame(kwargs)


def add_dfply_support(transform):
    """Add support for calling [dfply][dfply] functions inside pipes.

    This function is a rewrite rule and best be used with
    :func:`flowly.tz.apply` and :func:`flowly.tz.pipe`.

    Example::

        from flowly.tz import pipe
        from flowly.df import add_dfply_support

        from dfply import transmute, X

        pipe(
            df,
            transmute(c=X.a + X.b),
            rewrites=[add_dfply_support],
        )

    """
    return dfply_support(transform, dfply_support)


dfply_support = Dispatch()


@dfply_support.default
def dfply_support_default(obj, _):
    return obj


@dfply_support.bind(chained)
def dfply_support_chained(chain, dfply_support):
    return chained(*[
        dfply_support(func, dfply_support)
        for func in chain
    ])


@dfply_support.bind_conditional('dfply')
def dfply_support_dfply(dfply_support):
    import dfply

    @dfply_support.bind(dfply.pipe)
    def dfply_support_dfply_pipe(p, _):
        return lambda df: df >> p


@dfply_support.bind_rule(toolz.curry, lambda p, _: p.func == builtins.map)
def dfply_support_curry_map(p, dfply_support):
    return toolz.curry(
        p.func,
        *[dfply_support(f, dfply_support) for f in p.args],
        **p.keywords
    )


@dfply_support.bind_rule_default(toolz.curry)
def dfply_support_curry_default(p, _):
    return p
