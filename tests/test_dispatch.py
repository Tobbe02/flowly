from flowly._dispatch import Dispatch

import pytest


def test_parent():
    parent, _ = create_dispatchers()

    assert parent(3) == 'parent-int'
    assert parent(1.0) == 'parent-float'
    assert parent('') == 'parent-default'
    assert parent(class1()) == 'parent-class1'
    assert parent(class2()) == 'parent-class2'
    assert parent(class3()) == 'parent-class3'


def test_child():
    _, child = create_dispatchers()

    assert child(3) == 'parent-int'
    assert child(1.0) == 'child-float'
    assert child('') == 'child-default'
    assert child(class1()) == 'parent-class1'
    assert child(class2()) == 'child-class2'
    assert child(class3()) == 'child-class3'


def test_default_chaining():
    parent = Dispatch()
    child = parent.inherit()
    parent.default(lambda _: 'parent-default')

    assert parent('') == 'parent-default'
    assert child('') == 'parent-default'


def test_no_default():
    parent = Dispatch()
    child = parent.inherit()
    child.default(lambda _: 'child-default')

    with pytest.raises(ValueError):
        assert parent('') == 'parent-default'

    assert child('') == 'child-default'


def test_double_binding_raises():
    parent = Dispatch()
    parent.bind(int, lambda _: 'foo')

    with pytest.raises(ValueError):
        parent.bind(int, lambda _: 'bar')


def create_dispatchers():
    parent = Dispatch()
    child = parent.inherit()

    parent.bind(int, lambda _: 'parent-int')
    parent.bind(float, lambda _: 'parent-float')
    parent.bind_conditional('foo', lambda d: (
        d
        .bind(class1, lambda _: 'parent-class1')
        .bind(class2, lambda _: 'parent-class2')
    ))
    parent.bind(class3, lambda _: 'parent-class3')
    parent.default(lambda _: 'parent-default')

    child.bind(float, lambda _: 'child-float')
    child.bind_conditional('foo', lambda d: d.bind(class3, lambda _: 'child-class3'))
    child.bind(class2, lambda _: 'child-class2')
    child.default(lambda _: 'child-default')

    return parent, child


class class1(object):
    __module__ = 'foo'


class class2(object):
    __module__ = 'foo'


class class3(object):
    __module__ = 'foo'
