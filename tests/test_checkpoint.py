from flowly.checkpoint import checkpoint, add_checkpoints
from flowly.tz import chained, apply


def test_checkpoints__empty():
    transform = chained()
    assert apply(transform, 5, rewrites=[add_checkpoints]) == 5


def test_checkpoints__no_checkpoints():
    transform = chained(lambda x: x * 2, lambda x: x - 3)
    assert apply(transform, 5, rewrites=[add_checkpoints]) == 7


def test_checkpoints__single():
    _checkpoints = {}
    calls = {}

    transform = chained(
        _count(calls, 'step 1', lambda x: x * 2),
        checkpoint(target=_checkpoints),
        _count(calls, 'step 2', lambda x: x - 3)
    )

    assert apply(transform, 5, rewrites=[add_checkpoints]) == 7
    assert calls == {'step 1': 1, 'step 2': 1}

    assert apply(transform, 5, rewrites=[add_checkpoints]) == 7
    assert calls == {'step 1': 1, 'step 2': 2}

    assert apply(transform, 5, rewrites=[add_checkpoints]) == 7
    assert calls == {'step 1': 1, 'step 2': 3}

    _checkpoints.clear()

    assert apply(transform, 5, rewrites=[add_checkpoints]) == 7
    assert calls == {'step 1': 2, 'step 2': 4}


def test_checkpoints__single_no_rewrite():
    _checkpoints = {}
    calls = {}

    transform = chained(
        _count(calls, 'step 1', lambda x: x * 2),
        checkpoint(target=_checkpoints),
        _count(calls, 'step 2', lambda x: x - 3)
    )

    assert apply(transform, 5) == 7
    assert calls == {'step 1': 1, 'step 2': 1}

    assert apply(transform, 5) == 7
    assert calls == {'step 1': 2, 'step 2': 2}


def test_checkpoints__rewrites():
    """
    Test that only parts that were changed are executed if a pipeline is
    changed.
    """
    _checkpoints = {}
    calls = {}

    transform = chained(
        _count(calls, 'step 1', lambda x: x * 2),
        checkpoint(target=_checkpoints),
        _count(calls, 'step 2', lambda x: x - 3),
        checkpoint(target=_checkpoints),
        _count(calls, 'step 3', lambda x: x),
    )

    assert apply(transform, 5, rewrites=[add_checkpoints]) == 7
    assert calls == {'step 1': 1, 'step 2': 1, 'step 3': 1}

    assert apply(transform, 5, rewrites=[add_checkpoints]) == 7
    assert calls == {'step 1': 1, 'step 2': 1, 'step 3': 2}

    transform = chained(
        _count(calls, 'step 1', lambda x: x * 2),
        checkpoint(target=_checkpoints),
        _count(calls, 'alt-step 2', lambda x: x),
        checkpoint(target=_checkpoints),
        _count(calls, 'step 3', lambda x: x),
    )

    assert apply(transform, 5, rewrites=[add_checkpoints]) == 10
    assert calls == {'step 1': 1, 'step 2': 1, 'alt-step 2': 1, 'step 3': 3}

    assert apply(transform, 5, rewrites=[add_checkpoints]) == 10
    assert calls == {'step 1': 1, 'step 2': 1, 'alt-step 2': 1, 'step 3': 4}


def test_checkpoints__repr():
    """
    Test that only parts that were changed are executed if a pipeline is
    changed.
    """
    _checkpoints = {}
    transform = add_checkpoints(chained(
        lambda x: x * 2,
        checkpoint(target=_checkpoints),
        lambda x: x
    ))
    repr(transform)


class _count(object):
    """Count calls of the function.
    """
    def __init__(self, target, tag, func):
        self.target = target
        self.tag = tag
        self.func = func

    def __call__(self, *args, **kwargs):
        self.target[self.tag] = self.target.get(self.tag, 0) + 1
        return self.func(*args, **kwargs)

    def __reduce__(self):
        # NOTE: skip target to make closure proper pickable
        return _count, ({}, self.tag, self.func)
