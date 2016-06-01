from __future__ import division
import pickle
import unittest

from flowly import pipe, this, lit, wrapped


class TestPipe(unittest.TestCase):
    def test_pipe_callables(self):
        self.assertEqual(8, +(pipe(2) | (lambda x: x * 2) | (lambda x: x + 4)))

    def test_pipe_this(self):
        self.assertEqual(8, +(pipe(2) | this * 2 | this + 4))

    def test_unbound_pipe(self):
        transform = pipe() | this * 2 | this + 4
        self.assertEqual(8, +(pipe(2) | transform))

    def test_pipe_logical_expressions(self):
        self.assertEqual(0, +(pipe(1) | (this | 0) | (this & 0)))

    def test_this_add(self):
        self.assertEqual(3, +(pipe(2) | this + 1))
        self.assertEqual(3, +(pipe(2) | 1 + this))

    def test_this_sub(self):
        self.assertEqual(7, +(pipe(21) | this - 14))
        self.assertEqual(7, +(pipe(14) | 21 - this))

    def test_this_mul(self):
        self.assertEqual(21, +(pipe(7) | this * 3))
        self.assertEqual(21, +(pipe(7) | 3 * this))

    def test_this_div(self):
        self.assertEqual(3, +(pipe(21) | this / 7))
        self.assertEqual(3, +(pipe(7) | 21 / this))

    def test_this_floordir(self):
        assert 3 == +(pipe(24) | this // 7)
        assert 3 == +(pipe(7) | 24 // this)

    def test_this_pow(self):
        assert 8 == +(pipe(2) | this ** 3)
        assert 8 == +(pipe(3) | 2 ** this)

    def test_this_abs(self):
        assert 1 == +(pipe(-1) | abs(this))

    def test_this_call(self):
        assert ord('a') == +(pipe(ord) | this('a'))

    def test_this_attr_call(self):
        assert [1, 2, 3] == +(pipe(dict(a=1, b=2, c=3)) | this.values() | sorted)

    def test_this_getitem(self):
        assert 2 == +(pipe(dict(a=1, b=2)) | this['b'])

    def test_lit(self):
        assert 1 == +(pipe(2) | lit(1))

    def test_lit_call(self):
        assert ord('a') == +(pipe('a') | lit(ord)(this))


class TestPipeFunc(unittest.TestCase):
    def test_pipe_func(self):
        add = pipe.func() | this[0] + this[1]
        assert 3 == add(1, 2)

        sub = pipe.func("left", "right") | this.left - this.right

        assert 1 == sub(3, 2)
        assert 1 == sub(right=2, left=3)

    def test_pipe_func_multi_step(self):
        add = pipe.func() | this[0] + this[1] | this ** 3 | (lambda o: o + 3)
        assert 30 == add(1, 2)

        sub = pipe.func("left", "right") | this.left - this.right

        assert 1 == sub(3, 2)
        assert 1 == sub(right=2, left=3)

    def test_pipe_func_args_empty_argspec(self):
        result = pipe.func()(1, 2, 3)
        self.assertEqual(set(), self._args(result))
        self.assertEqual(1, result[0])
        self.assertEqual(2, result[1])
        self.assertEqual(3, result[2])

    def test_pipe_func_args_single_arg(self):
        result = pipe.func("left")(1, 2, 3)
        self.assertEqual({"left"}, self._args(result))
        self.assertEqual(2, len(result))

        self.assertEqual(1, result.left)
        self.assertEqual(2, result[0])
        self.assertEqual(3, result[1])

    def test_pipe_func_two_args(self):
        result = pipe.func("left", "right")(1, 2, 3)

        self.assertEqual({"left", "right"}, self._args(result))
        self.assertEqual(1, len(result))
        self.assertEqual(1, result.left)
        self.assertEqual(2, result.right)
        self.assertEqual(3, result[0])

    def test_pipe_func_kwargs(self):
        result = pipe.func("left", "right")(left=1, right=2)
        self.assertEqual({"left", "right"}, self._args(result))
        self.assertEqual(0, len(result))
        self.assertEqual(1, result.left)
        self.assertEqual(2, result.right)

    def test_pipe_func_args_single_arg_kwargs(self):
        result = pipe.func("left")(1, 2, 3, foo="bar")

        self.assertEqual({"left", "foo"}, self._args(result))
        self.assertEqual(2, len(result))

        self.assertEqual(1, result.left)
        self.assertEqual(2, result[0])
        self.assertEqual(3, result[1])

        self.assertEqual("bar", result.foo)

    @staticmethod
    def _args(obj):
        return set(key for key in obj.__dict__.keys() if not key.startswith("_"))


class TestWraped(unittest.TestCase):
    def test_base_behavior(self):
        self.assertEqual(5, +(pipe(3) | foo()))

    def test_picle_unpickle(self):
        self.assertEqual(5, +(pipe(3) | pickle.loads(pickle.dumps(foo))()))


@wrapped
def foo(obj):
    return obj + 2


if __name__ == "__main__":
    unittest.main()
