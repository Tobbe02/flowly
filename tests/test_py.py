from __future__ import print_function, division
import unittest
from flowly import py as fpy, pipe, this, lit


def test_filter():
    assert [0, 2, 4] == +(pipe(range(5)) | fpy.filter((this % 2) == 0) | list)


def test_map():
    assert [0, -1, -2] == +(pipe(range(3)) | fpy.map(-this) | list)


def test_reduce():
    assert 6 == +(pipe(range(4)) | fpy.reduce(this.left + this.right))
    assert 26 == +(pipe(range(4)) | fpy.reduce(this.left + this.right, 20))


def test_setattr():
    class class_(object):
        pass

    obj = class_()

    actual =  +(pipe(obj) | fpy.setattr(this, "foo", "bar") | this.foo)
    assert "bar" == actual


def test_setitem():
    actual =  +(
        pipe({}) |
        fpy.setitem(this, "foo", "bar") |
        this.keys() |
        set
    )
    assert {"foo"} == actual


def test_update():
    d = {}

    actual = +(
        pipe(d) |
        fpy.update(
            hello="world",
            foo="bar"
        )
    )

    assert {"hello": "world", "foo": "bar"} == actual
    assert {} == d


def test_assign():
    class class_(object):
        pass

    obj = class_()
    actual =  +(pipe(obj) | fpy.assign(foo="bar") | this.foo)
    assert "bar" == actual


class TestPipeFuncFor(unittest.TestCase):
    def test_example(self):
        from flowly.py import assign, range, for_

        factorial_fragment = (
            pipe.func("start", "end") |
            assign(current=1) |
            for_("i").in_(range(this.start, this.end + 1))(
                assign(current=this.current * this.i)
            ) |
            this.current
        )
        self.assertEqual(1, factorial_fragment(1, 1))

        self.assertEqual(2, factorial_fragment(1, 2))
        self.assertEqual(2, factorial_fragment(2, 2))

        self.assertEqual(6, factorial_fragment(1, 3))
        self.assertEqual(6, factorial_fragment(2, 3))
        self.assertEqual(3, factorial_fragment(3, 3))

        self.assertEqual(24, factorial_fragment(1, 4))
        self.assertEqual(24, factorial_fragment(2, 4))
        self.assertEqual(12, factorial_fragment(3, 4))
        self.assertEqual(4, factorial_fragment(4, 4))

    def test_for_literal(self):
        from flowly.py import assign, for_

        impl = (
            pipe.func() |
            for_("result").in_([1, 2, 3])(this) |
            this.result
        )

        self.assertEqual(3, impl())

    def test_if(self):
        from flowly.py import assign, if_

        impl = (
            pipe.func("arg") |
            assign(result="odd") |
            if_((this.arg % 2) == 0).do( assign(result="even") ) |
            this.result
        )

        self.assertEqual("even", impl(0))
        self.assertEqual("odd", impl(1))
        self.assertEqual("even", impl(2))
        self.assertEqual("odd", impl(3))

    def test_if_literal_cond(self):
        from flowly.py import assign, if_

        impl = (
            pipe.func("arg") |
            if_(True).do( assign(result="foo") ) |
            this.result
        )

        self.assertEqual("foo", impl(0))
        self.assertEqual("foo", impl(1))

    def test_if_else(self):
        from flowly.py import assign, if_

        impl = (
            pipe.func("arg") |
            if_((this.arg % 2) == 0).do( assign(result="even") )
            .else_.do( assign(result="odd") ) |
            this.result
        )

        self.assertEqual("even", impl(0))
        self.assertEqual("odd", impl(1))
        self.assertEqual("even", impl(2))
        self.assertEqual("odd", impl(3))

    def test_if_elif_else(self):
        from flowly.py import assign, if_

        impl = (
            pipe.func("arg") |
            if_(this.arg == 0).do( assign(result="zero") )
            .elif_(this.arg == 1).do( assign(result="one") )
            .else_.do( assign(result="number") ) |
            this.result
        )

        self.assertEqual("zero", impl(0))
        self.assertEqual("one", impl(1))
        self.assertEqual("number", impl(2))
        self.assertEqual("number", impl(3))

    def test_if_elif_elif_else(self):
        from flowly.py import assign, if_

        impl = (
            pipe.func("arg") |
            if_(this.arg == 0).do( assign(result="zero") )
            .elif_(this.arg == 1).do( assign(result="one") )
            .elif_(this.arg == 2).do( assign(result="two") )
            .else_.do( assign(result="number") ) |
            this.result
        )

        self.assertEqual("zero", impl(0))
        self.assertEqual("one", impl(1))
        self.assertEqual("two", impl(2))
        self.assertEqual("number", impl(3))

    def test_if_elif_elif_else_functional(self):
        from flowly.py import assign, if_

        impl = (
            pipe.func("arg") |
            if_(this.arg == 0)("zero")
            .elif_(this.arg == 1)("one")
            .elif_(this.arg == 2)("two")
            .else_("number")
        )

        self.assertEqual("zero", impl(0))
        self.assertEqual("one", impl(1))
        self.assertEqual("two", impl(2))
        self.assertEqual("number", impl(3))

    def test_try_catch(self):
        from flowly.py import do, try_, assign

        impl = (
            pipe.func("arg") |
            try_.do(pipe() | do(this.arg.items) | assign(result="has_items"))
            .except_(AttributeError).do(
                assign(result="has_no_items")
            ) |
            this.result
        )

        self.assertEqual("has_no_items", impl([]))
        self.assertEqual("has_items", impl({}))

    def test_try_catch_functional(self):
        from flowly.py import do, try_, assign

        impl = (
            pipe.func("arg") |
            try_(pipe() | this.arg.items | lit("has_items"))
            .except_(AttributeError)(lit("has_no_items"))
        )

        self.assertEqual("has_no_items", impl([]))
        self.assertEqual("has_items", impl({}))

    def test_try_catch_else(self):
        from flowly.py import try_, assign

        impl = (
            pipe.func("arg") |
            try_.do(this.arg.items)
            .except_(AttributeError).do(assign(result="has_no_items"))
            .else_.do(assign(result="has_items")) |
            this.result
        )

        self.assertEqual("has_no_items", impl([]))
        self.assertEqual("has_items", impl({}))

    def test_try_catch_else_functional(self):
        from flowly.py import try_, assign

        impl = (
            pipe.func("arg") |
            try_.do(this.arg.items)
            .except_(AttributeError)(lit("has_no_items"))
            .else_(lit("has_items"))
        )

        self.assertEqual("has_items", impl({}))
        self.assertEqual("has_no_items", impl([]))

    def test_try_finally(self):
        from flowly.py import try_, raise_, setitem

        impl = (
            pipe.func("arg") |
            try_.do(raise_(RuntimeError()))
            .finally_(setitem(this.arg, "result", "foo"))
        )

        obj = {}
        try:
            impl(obj)

        except RuntimeError:
            pass

        else:
            raise AssertionError()

        self.assertEqual({"result": "foo"}, obj)
