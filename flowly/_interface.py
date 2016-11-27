from __future__ import print_function, division, absolute_import

from ._base import eval_expr


class unary(object):
    """Expose a unary flowly expression as a callable.
    """
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, obj):
        return eval_expr(obj, self.expr)


predicate = unary


class binary(object):
    """Expose a binary flowly expression as a callable.

    Example::

        binary(this.a + this.b)(2, 3)

    """
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, a, b):
        return eval_expr(binary_object(a, b), self.expr)


class binary_object(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

        self.left = a
        self.right = b

    def __getitem__(self, idx):
        return [self.a, self.b][idx]
