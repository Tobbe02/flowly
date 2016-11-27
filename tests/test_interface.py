from __future__ import print_function, division, absolute_import
from flowly import this, binary, unary


def test_unary__example():
    assert unary(this * 3)(4) == 12


def test_binary__examples():
    assert binary(this.a - this.b)(1, 2) == -1
    assert binary(this.left - this.right)(1, 2) == -1
    assert binary(this[0] - this[1])(1, 2) == -1
