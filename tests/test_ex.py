from flowly.ex import get, NotDone
from flowly.tz import raise_

import pytest


def test_get__not_done():
    with pytest.raises(NotDone):
        +get(Future())


def test_get__failed():
    with pytest.raises(ValueError):
        +get(Future(done=lambda: True, result=lambda: raise_(ValueError)))


def test_get__succes():
    assert +get(Future(done=lambda: True, result=lambda: 42)) == 42


class Future(object):
    def __init__(self, done=lambda: False, result=lambda: None):
        self.done = done
        self.result = result


