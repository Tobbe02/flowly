from __future__ import division
from flowly import pipe, this, lit


def test_pipe_callables():
    assert 8 == +(pipe(2) | (lambda x: x * 2) | (lambda x: x + 4))


def test_pipe_this():
    assert 8 == +(pipe(2) | this * 2 | this + 4)


def test_unbound_pipe():
    transform = pipe() | this * 2 | this + 4
    assert 8 == +(pipe(2) | transform)


def test_pipe_logical_expressions():
    assert 0 == +(pipe(1) | (this | 0) | (this & 0))


def test_this_add():
    assert 3 == +(pipe(2) | this + 1)
    assert 3 == +(pipe(2) | 1 + this)


def test_this_sub():
    assert 7 == +(pipe(21) | this - 14)
    assert 7 == +(pipe(14) | 21 - this)


def test_this_mul():
    assert 21 == +(pipe(7) | this * 3)
    assert 21 == +(pipe(7) | 3 * this)


def test_this_div():
    assert 3 == +(pipe(21) | this / 7)
    assert 3 == +(pipe(7) | 21 / this)


def test_this_floordir():
    assert 3 == +(pipe(24) | this // 7)
    assert 3 == +(pipe(7) | 24 // this)


def test_this_pow():
    assert 8 == +(pipe(2) | this ** 3)
    assert 8 == +(pipe(3) | 2 ** this)


def test_this_abs():
    assert 1 == +(pipe(-1) | abs(this))


def test_this_call():
    assert ord('a') == +(pipe(ord) | this('a'))


def test_this_attr_call():
    assert [1, 2, 3] == +(pipe(dict(a=1, b=2, c=3)) | this.values() | sorted)


def test_this_getitem():
    assert 2 == +(pipe(dict(a=1, b=2)) | this['b'])


def test_lit():
    assert 1 == +(pipe(2) | lit(1))


def test_lit_call():
    assert ord('a') == +(pipe('a') | lit(ord)(this))
