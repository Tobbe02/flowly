from __future__ import print_function
import unittest


class TestBf(unittest.TestCase):
    def test_empty_program(self):
        memory, output = bf("", [], [])
        self.assertEqual([], memory)
        self.assertEqual([], output)

    def test_increment(self):
        memory, output = bf("+", [0], [])
        self.assertEqual([1], memory)
        self.assertEqual([], output)

    def test_decrement(self):
        memory, output = bf("-", [1], [])
        self.assertEqual([0], memory)
        self.assertEqual([], output)

    def test_increment_decrement(self):
        memory, output = bf("+-", [0], [])
        self.assertEqual([0], memory)
        self.assertEqual([], output)

    def test_move_pos(self):
        memory, output = bf("+>-", [0, 1], [])
        self.assertEqual([1, 0], memory)
        self.assertEqual([], output)

    def test_move_neg(self):
        memory, output = bf("+>-<+", [0, 1], [])
        self.assertEqual([2, 0], memory)
        self.assertEqual([], output)

    def test_simple_loop(self):
        memory, output = bf("[-]", [2], [])
        self.assertEqual([0], memory)
        self.assertEqual([], output)

    def test_subtract(self):
        memory, output = bf(">>[<->-]", [0, 3, 2], [])
        self.assertEqual([0, 1, 0], memory)
        self.assertEqual([], output)

    def test_subtract2(self):
        memory, output = bf(">>[<->-]", [0, 7, 4], [])
        self.assertEqual([0, 3, 0], memory)
        self.assertEqual([], output)

    def test_shift(self):
        memory, output = bf(">[-]<[>+<-]", [13, 10], [])
        self.assertEqual([0, 13], memory)
        self.assertEqual([], output)


def define_bf():
    """define a brainfuck interpreter fully with flowly objects.
    """
    from flowly import pipe, this
    from flowly.py import (
        assign, as_tuple, if_, setitem, while_, len, or_, raise_,
    )

    return (
        pipe.func("program", "memory", "input") |
        assign(
            output=[],
            memory=pipe() | this.memory | list,
            pidx=0,
            midx=0,
            depth=0,
        ) |
        while_(this.pidx < len(this.program)).do(
            if_(this.program[this.pidx] == '+').do(
                pipe() |
                setitem(this.memory, this.midx, this.memory[this.midx] + 1) |
                assign(pidx=this.pidx + 1)
            )
            .elif_(this.program[this.pidx] == '-').do(
                pipe() |
                setitem(this.memory, this.midx, this.memory[this.midx] - 1) |
                assign(pidx=this.pidx + 1)
            )
            .elif_(this.program[this.pidx] == '>').do(
                assign(midx=this.midx + 1, pidx=this.pidx + 1)
            )
            .elif_(this.program[this.pidx] == '<').do(
                assign(midx=this.midx - 1, pidx=this.pidx + 1)
            )
            .elif_(this.program[this.pidx] == '[').do(
                pipe() |
                if_(this.memory[this.midx] == 0).do(
                    pipe() |
                    assign(search_depth=1) |
                    while_(
                        or_(
                            this.program[this.pidx] != ']',
                            this.search_depth != 0
                        )
                    ).do(
                        pipe() |
                        assign(pidx=this.pidx + 1) |
                        if_(this.program[this.pidx] == '[').do(
                            assign(search_depth=this.search_depth + 1)
                        )
                        .elif_(this.program[this.pidx] == ']').do(
                            assign(search_depth=this.search_depth - 1)
                        )
                    )
                ) |
                assign(pidx=this.pidx + 1)
            )
            .elif_(this.program[this.pidx] == ']').do(
                pipe() |
                if_(this.memory[this.midx] != 0).do(
                    pipe() |
                    assign(search_depth=1) |
                    while_(
                        or_(
                            this.program[this.pidx] != '[',
                            this.search_depth != 0
                        )
                    ).do(
                        pipe() |
                        assign(pidx=this.pidx - 1) |
                        if_(this.program[this.pidx] == ']').do(
                            assign(search_depth=this.search_depth + 1)
                        )
                        .elif_(this.program[this.pidx] == '[').do(
                            assign(search_depth=this.search_depth - 1)
                        )
                    )
                ) |
                assign(pidx=this.pidx + 1)
            )
            .elif_(this.program[this.pidx] == '.').do(
                raise_(ValueError("IO not implemented"))
            )
            .elif_(this.program[this.pidx] == ',').do(
                raise_(ValueError("IO not implemented"))
            )
            .else_.do(raise_(ValueError()))
        ) |
        as_tuple(this.memory, this.output)
    )


bf = define_bf()
