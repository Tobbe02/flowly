from flowly import py as fpy, pipe, this, lit


def test_filter():
    assert [0, 2, 4] == +(pipe(range(5)) | fpy.filter((this % 2) == 0) | list)


def test_map():
    assert [0, -1, -2] == +(pipe(range(3)) | fpy.map(-this) | list)


def test_reduce():
    assert 6 == +(pipe(range(4)) | fpy.reduce(this.left + this.right))
    assert 26 == +(pipe(range(4)) | fpy.reduce(this.left + this.right, 20))
