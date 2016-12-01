from __future__ import print_function, division, absolute_import

from ._base import eval_expr, is_flowly_expr


class unary(object):
    """Expose a unary flowly expression as a callable.

    When the unary object is called with a single argument, the ``this`` object
    is replaced by that argument. If a non flowly expressions is wrapped, the
    object is simply called with the given argument.

    To specify intent, also ``unary`` can be replaced by ``predicate``.

    Examples::

        unary(this.foo())(obj)
        unary(this.is_valid())(obj)
        predicate(this.is_valid())(obj)
    """
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, obj):
        if is_flowly_expr(self.expr):
            return eval_expr(obj, self.expr)

        else:
            return self.expr(obj)


predicate = unary


class binary(object):
    """Expose a binary flowly expression as a callable.

    When the binary object is called with two arguments, the ``this`` object is
    replaced by a temporary object with members ``a``, ``left`` (both bound to
    the first argument) and ``b``, ``right`` (both bound to the second
    argument). Furthermore, the arguments can be access by indexing.

    For example, the follwing calls are all equivalent::

        binary(this.a + this.b)(2, 3)
        binary(this.left + this.right)(2, 3)
        binary(this[0] + this[1])(2, 3)

    If a non flowly expressions is wrapped, the object is simply called with the
    given arguments.
    """
    def __init__(self, expr):
        self.expr = expr

    def __call__(self, a, b):
        if is_flowly_expr(self.expr):
            return eval_expr(binary_object(a, b), self.expr)

        else:
            return self.expr(a, b)


class binary_object(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

        self.left = a
        self.right = b

    def __getitem__(self, idx):
        return [self.a, self.b][idx]
